"""Compute caption BLEU for the Gemini temporal runs, exactly as the temporal eval does.

The temporal Batch-API Gemini runs (results/evaluation/gemini/.../temporal_<SET>_batch_full/)
store the predicted text and the reference caption per clip, but the batch path did
not run the caption-metric block, so the result JSONs have no caption_bleu. This
script fills that gap using the *same* helpers the real evaluator uses, so the number
is directly comparable to the caption_bleu reported for our own temporal models:

  - strip_time_tokens_for_caption (from scripts/eval/evaluate_temporal.py): drops the
    localization clause / timestamp tokens from both prediction and reference.
  - compute_caption_metrics (from scripts/eval/evaluate.py): corpus BLEU (cased), the
    global eval's helper, applied to the stripped captions.

It reports per-set BLEU and a pooled BLEU over FOR/LIVE/P501 (corpus BLEU is computed
on the pooled caption set, which is the right "aggregated over the three sets" number,
not a mean of per-set BLEUs).

Run from the repo root with the project venv active:

    python scripts/eval/gemini_temporal_caption_bleu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse the exact eval helpers so the number matches our models' caption_bleu.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import compute_caption_metrics  # noqa: E402
from evaluate_temporal import strip_time_tokens_for_caption  # noqa: E402

GEMINI_ROOT = Path(
    "results/evaluation/gemini/gemini31_pro_preview"
)
SETS = ["FOR", "LIVE", "P501"]


def load_captions(set_name: str) -> tuple[list[str], list[str]]:
    """Return (stripped hyps, stripped refs) for one temporal set, the same way
    the temporal evaluator builds caption_hyps / caption_refs."""
    path = GEMINI_ROOT / f"temporal_{set_name}_batch_full" / "results.json"
    records = json.loads(path.read_text())["results"]
    hyps, refs = [], []
    for r in records:
        pred = str(r.get("predicted_response", ""))
        gold = str(r.get("response", ""))
        hyps.append(strip_time_tokens_for_caption(pred))
        refs.append(strip_time_tokens_for_caption(gold))
    return hyps, refs


def main() -> None:
    all_hyps: list[str] = []
    all_refs: list[str] = []
    print(f"{'set':6} {'n':>5} {'BLEU':>7}")
    for s in SETS:
        hyps, refs = load_captions(s)
        bleu = compute_caption_metrics(hyps, refs)["bleu"]
        print(f"{s:6} {len(hyps):>5} {bleu:>7.2f}")
        all_hyps += hyps
        all_refs += refs
    pooled = compute_caption_metrics(all_hyps, all_refs)["bleu"]
    print(f"{'POOL':6} {len(all_hyps):>5} {pooled:>7.2f}")
    # Simple mean of per-set BLEUs, for reference (not the headline number).
    per_set = [compute_caption_metrics(*load_captions(s))["bleu"] for s in SETS]
    print(f"mean-of-sets BLEU: {sum(per_set) / len(per_set):.2f}")


if __name__ == "__main__":
    main()
