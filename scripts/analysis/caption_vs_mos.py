"""Per-sample analysis: does caption quality predict MOS accuracy?

Tests the hypothesis that when the model describes the audio well (high caption
similarity to the reference), it also predicts the MOS more accurately (low
absolute MOS error). Reports per-test-set Spearman correlation, never pooled
across test sets (different MOS distributions would invite Simpson's paradox).

Design decisions (see docs/handoff or the thesis methods section):
- Caption similarity uses BERTScore-F1 and ROUGE-L only. Sentence-level BLEU on
  n=1 is degenerate (brevity penalty + a single missing 4-gram collapses it to
  ~0), so it is excluded from per-sample work. BLEU stays a corpus metric.
- The numeric MOS rating is stripped from BOTH reference and prediction before
  caption scoring. The reference caption states the MOS in prose ("...MOS of
  4.3..."), so an unstripped score would reward copying the number and
  manufacture a correlation by construction. Descriptive prose is preserved.
- ``mos_error`` is read from the stored per-sample field (computed at eval time
  from the original predicted MOS). It is never re-derived from stripped text.
- Spearman (rank) is used, not Pearson: BERTScore is compressed into a narrow
  band and the distributions are skewed.

Usage:
    python scripts/analysis/caption_vs_mos.py \
        --results-path results/evaluation/sft/sft_full_eval/test_FOR_results.json \
        --results-path results/evaluation/sft/sft_full_eval/test_LIVE_results.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import typer
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Correlate per-sample caption quality with MOS error.")

# Strip the numeric MOS rating while keeping the surrounding descriptive prose.
# Matches "MOS" / "MOS score" + optional connector (of/as/is/:) + optional hedge
# (only/just/around/...) + the number. Replaces the whole match with "MOS".
MOS_NUM_RE = re.compile(
    r"\bMOS(?:\s+score)?\b\s*"
    r"(?:of|as|is|:|score\s+(?:of|as|is))?\s*"
    r"(?:only|just|around|about|approximately|roughly)?\s*"
    r"\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def strip_mos_number(text: str) -> str:
    """Remove the numeric MOS rating, preserving descriptive prose.

    Args:
        text: A caption that may contain a MOS rating phrase.

    Returns:
        The caption with the MOS number (and bare connector) removed.
    """
    cleaned = MOS_NUM_RE.sub("MOS", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    return cleaned.strip()


def _describe(name: str, values: List[float]) -> str:
    """One-line distribution summary for sanity-checking variance."""
    if not values:
        return f"{name}: (empty)"
    srt = sorted(values)
    n = len(srt)
    mean = sum(srt) / n
    p = lambda q: srt[min(n - 1, int(q * n))]  # noqa: E731
    return (
        f"{name}: n={n} mean={mean:.4f} min={srt[0]:.4f} "
        f"p25={p(0.25):.4f} median={p(0.5):.4f} p75={p(0.75):.4f} "
        f"max={srt[-1]:.4f}"
    )


@app.command()
def analyze(
    results_paths: List[Path] = typer.Option(
        ...,
        "--results-path",
        help="Existing *_results.json file(s) to analyze (one per test set).",
    ),
    bertscore_model: str = typer.Option(
        "roberta-large", help="HuggingFace backbone for BERTScore."
    ),
    output_path: Optional[Path] = typer.Option(
        None, help="Optional JSON path to write the per-test-set summary."
    ),
    strip_mos: bool = typer.Option(
        True,
        "--strip-mos/--keep-mos",
        help="Strip the numeric MOS rating before caption scoring (default). "
        "--keep-mos leaves it in, for the strip-vs-no-strip robustness check.",
    ),
) -> None:
    """Per-test-set Spearman: caption similarity vs MOS absolute error."""
    from bert_score import score as bertscore_fn
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    transform = strip_mos_number if strip_mos else (lambda t: t)
    summary = []

    for results_path in results_paths:
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("results", [])

        hyps_stripped: List[str] = []
        refs_stripped: List[str] = []
        mos_errors: List[float] = []
        skipped = 0

        for record in records:
            hyp = record.get("predicted_response")
            ref = record.get("response")
            err = record.get("mos_error")
            if not isinstance(hyp, str) or not isinstance(ref, str):
                skipped += 1
                continue
            if not isinstance(err, (int, float)):
                skipped += 1
                continue
            hyps_stripped.append(transform(hyp.strip()))
            refs_stripped.append(transform(ref.strip()))
            mos_errors.append(float(err))

        if len(mos_errors) < 3:
            logging.warning("%s: too few scorable samples; skipping.", results_path)
            continue

        # Per-sample ROUGE-L F1.
        rougel = [
            scorer.score(ref, hyp)["rougeL"].fmeasure
            for hyp, ref in zip(hyps_stripped, refs_stripped)
        ]

        # Per-sample BERTScore F1 (genuinely per-sample, unlike corpus BLEU).
        logging.info(
            "%s: BERTScore on %d samples (%s)...",
            results_path.name,
            len(hyps_stripped),
            bertscore_model,
        )
        _, _, bs_f1 = bertscore_fn(
            hyps_stripped,
            refs_stripped,
            model_type=bertscore_model,
            lang="en",
            rescale_with_baseline=False,
            verbose=False,
        )
        bertscore_f1 = [v.item() for v in bs_f1]

        # Direction: higher similarity -> lower error -> expect negative rho.
        rho_bert, p_bert = spearmanr(bertscore_f1, mos_errors)
        rho_rouge, p_rouge = spearmanr(rougel, mos_errors)
        # #4 redundancy (appendix): ROUGE-L vs BERTScore per sample.
        rho_metric, _ = spearmanr(rougel, bertscore_f1)

        entry = {
            "results_path": str(results_path),
            "test_set": results_path.stem,
            "mos_stripped": strip_mos,
            "n": len(mos_errors),
            "skipped": skipped,
            "spearman_bertscore_vs_mos_error": rho_bert,
            "p_bertscore": p_bert,
            "spearman_rougeL_vs_mos_error": rho_rouge,
            "p_rougeL": p_rouge,
            "spearman_rougeL_vs_bertscore": rho_metric,
            "dist_mos_error": _describe("mos_error", mos_errors),
            "dist_bertscore_f1": _describe("bertscore_f1", bertscore_f1),
            "dist_rougeL": _describe("rougeL", rougel),
        }
        summary.append(entry)

        logging.info("=" * 60)
        logging.info("TEST SET: %s (n=%d, skipped=%d)", entry["test_set"], entry["n"], skipped)
        logging.info(entry["dist_mos_error"])
        logging.info(entry["dist_bertscore_f1"])
        logging.info(entry["dist_rougeL"])
        logging.info(
            "Spearman BERTScore-F1 vs MOS error: rho=%+.3f (p=%.3g)  "
            "[neg = better captions -> lower error]",
            rho_bert,
            p_bert,
        )
        logging.info(
            "Spearman ROUGE-L     vs MOS error: rho=%+.3f (p=%.3g)",
            rho_rouge,
            p_rouge,
        )
        logging.info(
            "Spearman ROUGE-L     vs BERTScore: rho=%+.3f  [metric redundancy]",
            rho_metric,
        )
        logging.info("=" * 60)

    if not summary:
        logging.warning("No test sets analyzed.")
        return

    # Cross-set robustness: is the sign consistent?
    signs = [e["spearman_bertscore_vs_mos_error"] for e in summary]
    consistent = all(s < 0 for s in signs) or all(s > 0 for s in signs)
    logging.info(
        "Cross-set sign consistency (BERTScore vs MOS error): %s",
        "CONSISTENT" if consistent else "MIXED",
    )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "sign_consistent": consistent}, f, indent=2
            )
        logging.info("Wrote summary to %s", output_path)


if __name__ == "__main__":
    app()
