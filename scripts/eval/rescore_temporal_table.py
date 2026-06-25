"""Re-score temporal eval result JSONs offline for the thesis 4.2 tables.

Some early temporal evaluations stored only the localization metrics (mean IoU,
endpoint error) and skipped caption BLEU / MOS MAE, leaving N.A. cells in the
results chapter. Those JSONs do, however, store the model's ``predicted_response``
and the gold ``response`` and ``mos`` per clip, so BLEU and MOS MAE can be recovered
offline without re-running inference on the cluster.

This script reuses the evaluation pipeline's own helpers (``extract_mos``,
``strip_time_tokens_for_caption``, ``sacrebleu.corpus_bleu``) so the recovered
numbers are identical to what an in-pipeline re-score would produce. It also
recomputes the unique-span count as a percentage of parsed clips, matching the
``Unique cap.`` percentage convention used in the global (4.1) tables.

Aggregation over FOR / LIVE / P501 is sample-weighted, because E_offset, MOS MAE
and the uniqueness percentage are per-sample quantities; BLEU is a corpus score, so
it is recomputed on the pooled hypotheses/references across the three sets rather
than averaged.

Usage:
    .venv/bin/python scripts/eval/rescore_temporal_table.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import sacrebleu  # noqa: E402

from evaluate import extract_mos  # noqa: E402
from evaluate_temporal import (  # noqa: E402
    strip_non_timestamp_special_tokens,
    strip_time_tokens_for_caption,
)

SETS = ["FOR", "LIVE", "P501"]

# Each arm: label -> (branch_or_None_for_local, path-template with {s} for the set).
# branch None means the file is in the current working tree. Every reported arm is
# committed locally under the new results layout, so all entries read from None.
B = "results/evaluation/temporal"

ARMS: dict[str, tuple[Optional[str], str]] = {
    # localization-only evals that currently read N.A. for BLEU / MOS MAE
    "zeroshot": (
        None,
        f"{B}/zeroshot/zeroshot__qwen2-instruct__t07/"
        "test_{s}_temporal_global_caption_results.json",
    ),
    "plain_tsfirst": (
        None,
        f"{B}/sft/gc-plain__h100__greedy/"
        "test_{s}_temporal_global_caption_results.json",
    ),
    "timeaudio_tsfirst": (
        None,
        f"{B}/sft/gc-anchoroffset__timeaudio-h100__greedy/"
        "test_{s}_temporal_global_caption_anchoroffset_results.json",
    ),
    # arms that already have BLEU/MOS, re-scored here so the uniqueness percentage
    # is computed the same way everywhere
    "timeaudio_timelast": (
        None,
        f"{B}/sft/gc-timelast__timeaudio-h100__greedy/"
        "test_{s}_temporal_global_caption_anchoroffset_results.json",
    ),
}
for n in ["500", "1000", "2500", "5105"]:
    ARMS[f"sweep{n}"] = (
        None,
        f"{B}/sft/gc-timelast__timeaudio-h100__greedy__n{n}/"
        "test_{s}_temporal_global_caption_anchoroffset_results.json",
    )
for label, dirname in [
    ("dpo_norm", "alld/armA-hardneg-iou06__h100__greedy"),
    ("dpo_nonorm", "alld/armA-hardneg-iou06-nonorm__h100__greedy"),
    ("dpo_lr5e6", "alld/armA-hardneg-iou06-lr5e6-beta1__h100__greedy"),
    ("dpo_medshift", "alld/armA-medshift__h100__greedy"),
]:
    ARMS[label] = (
        None,
        f"{B}/{dirname}/test_{{s}}_temporal_global_caption_anchoroffset_results.json",
    )


def load_json(branch: Optional[str], path: str) -> dict[str, Any]:
    if branch is None:
        with open(path) as f:
            return json.load(f)
    raw = subprocess.run(
        ["git", "show", f"{branch}:{path}"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(raw)


def rescore_set(doc: dict[str, Any]) -> dict[str, Any]:
    """Recompute BLEU, MOS MAE, E_offset and uniqueness for one test set."""
    results = doc["results"]
    hyps: list[str] = []
    refs: list[str] = []
    mos_abs_errs: list[float] = []
    endpoint_errs: list[float] = []
    parsed_intervals: list[tuple[float, float]] = []
    distinct_captions: set[str] = set()

    for r in results:
        pred_raw = str(r.get("predicted_response", ""))
        pred_text = strip_non_timestamp_special_tokens(pred_raw)
        gold_caption = str(r.get("response", ""))

        # caption BLEU: strip the timestamp clause from both sides (pipeline rule)
        hyp_caption = strip_time_tokens_for_caption(pred_text)
        hyps.append(hyp_caption)
        refs.append(strip_time_tokens_for_caption(gold_caption))
        distinct_captions.add(hyp_caption)

        # MOS MAE
        pred_mos = extract_mos(pred_text)
        gold_mos = r.get("mos")
        if pred_mos is not None and gold_mos is not None:
            mos_abs_errs.append(abs(float(gold_mos) - pred_mos))

        # E_offset = mean over clips of (|s_p - s_g| + |e_p - e_g|) / 2
        sa, ea = r.get("start_abs_err"), r.get("end_abs_err")
        if sa is not None and ea is not None:
            endpoint_errs.append((float(sa) + float(ea)) / 2.0)

        # uniqueness: distinct parsed intervals among clips that produced one
        ps, pe = r.get("pred_start"), r.get("pred_end")
        if ps is not None and pe is not None:
            parsed_intervals.append((round(float(ps), 2), round(float(pe), 2)))

    n = len(results)
    parsed = len(parsed_intervals)
    unique = len(set(parsed_intervals))
    return {
        "n": n,
        "hyps": hyps,
        "refs": refs,
        "mos_abs_errs": mos_abs_errs,
        "endpoint_errs": endpoint_errs,
        "parsed": parsed,
        "unique_intervals": unique,
        "unique_captions": len(distinct_captions),
        # interval uniqueness as a percentage of parsed clips: the temporal
        # collapse story is about the model emitting the same span over and over,
        # so this is the temporal analogue of the global "Unique cap." percentage
        "uniq_int_pct": (100.0 * unique / parsed) if parsed else None,
        # caption uniqueness as a percentage of all clips: the exact 4.1 metric
        # (share of distinct predicted captions), reported alongside for reference
        "uniq_cap_pct": (100.0 * len(distinct_captions) / n) if n else None,
    }


def aggregate(per_set: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sample-weighted aggregation; BLEU pooled over the three sets."""
    all_hyps: list[str] = []
    all_refs: list[str] = []
    mos_all: list[float] = []
    eoff_all: list[float] = []
    total_parsed = 0
    total_unique_int = 0
    total_unique_cap = 0
    total_n = 0
    for s, d in per_set.items():
        all_hyps += d["hyps"]
        all_refs += d["refs"]
        mos_all += d["mos_abs_errs"]
        eoff_all += d["endpoint_errs"]
        total_parsed += d["parsed"]
        total_unique_int += d["unique_intervals"]
        total_unique_cap += d["unique_captions"]
        total_n += d["n"]

    # BLEU on pooled hyps/refs (corpus-level, matches a single pooled re-score)
    bleu = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score if all_hyps else None
    # MOS MAE only meaningful if the model emitted parseable MOS at all
    mos_mae = (sum(mos_all) / len(mos_all)) if mos_all else None
    eoff = (sum(eoff_all) / len(eoff_all)) if eoff_all else None
    uniq_int_pct = (100.0 * total_unique_int / total_parsed) if total_parsed else None
    uniq_cap_pct = (100.0 * total_unique_cap / total_n) if total_n else None
    return {
        "N": total_n,
        "BLEU": bleu,
        "MOS_MAE": mos_mae,
        "E_offset": eoff,
        "uniq_int_pct": uniq_int_pct,
        "uniq_cap_pct": uniq_cap_pct,
        "parsed": total_parsed,
        "unique_intervals": total_unique_int,
    }


def fmt(x: Any, d: int = 3) -> str:
    if x is None:
        return "N.A."
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def main() -> None:
    hdr = (
        f"{'arm':20s} {'BLEU':>6s} {'MOSmae':>7s} {'Eoff':>6s} "
        f"{'UniqInt%':>9s} {'UniqCap%':>9s} {'N':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, (branch, tmpl) in ARMS.items():
        per_set = {}
        for s in SETS:
            doc = load_json(branch, tmpl.format(s=s))
            per_set[s] = rescore_set(doc)
        a = aggregate(per_set)
        print(
            f"{label:20s} {fmt(a['BLEU'], 2):>6s} {fmt(a['MOS_MAE']):>7s} "
            f"{fmt(a['E_offset']):>6s} {fmt(a['uniq_int_pct'], 1):>9s} "
            f"{fmt(a['uniq_cap_pct'], 1):>9s} {a['N']:>4d}"
        )


if __name__ == "__main__":
    main()
