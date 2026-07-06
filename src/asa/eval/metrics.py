"""Task-agnostic scoring: MOS parsing, caption metrics, diagnostics.

Every eval (global MOS, temporal, and both Gemini baselines) scores the model's
MOS score and descriptive caption with the functions here, so the numbers are
directly comparable across tasks and models. Keeping this in one module is what
prevents the four CLIs from each growing their own subtly-different copy.

The heavy caption-metric extras (``sacrebleu`` is a hard dep, but
``rouge-score`` and ``bert-score`` are imported lazily) so that a caller that
only needs MOS parsing, e.g. the interval-only temporal path, can import this
module on a node without them.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import sacrebleu

# Default BERTScore backbone. roberta-large is the bert-score library default and
# the most commonly cited choice for English; record it in the output for
# reproducibility since BERTScore values depend on the backbone.
BERTSCORE_MODEL = "roberta-large"


def mean_or_zero(values: Sequence[float]) -> float:
    """Return the mean of ``values``, or 0.0 for an empty sequence.

    The shared "mean-or-zero" used across the evals so metric aggregation never
    raises on an empty list (e.g. a set with no parsed intervals).
    """
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def extract_mos(text: str) -> Optional[float]:
    """Extract the model's MOS score from generated text.

    Returns the parsed score, or ``None`` when no score can be confidently
    located. ``None`` is deliberate: the old behaviour fell back to "the last
    number in the text", which on a zero-shot baseline that rambles or rates
    "3 out of 5" grabs the wrong digit (the "5" denominator) and fabricates a
    plausible-but-wrong MOS. An honest parse failure is better than a fake
    number, especially for the untrained baseline whose whole point is that it
    cannot do the task. Callers treat ``None`` as unparsed and report a parse
    rate alongside the error over parsed samples.

    Patterns are tried most-specific first:

    1. Explicit "MOS" mention, e.g. "MOS of 4.3", "overall MOS is 4.3". This is
       the format the fine-tuned models were trained to emit, so this branch is
       unchanged from the original implementation and their parsed scores are
       byte-for-byte identical.
    2. Out-of-5 ratings, e.g. "3 out of 5", "rated as 4 out of 5", "3/5". Takes
       the numerator, not the "5" denominator.
    3. "score/rating of X" and "rate ... as X" phrasings.

    There is intentionally NO blind last-number fallback.
    """
    # 1. Explicit MOS mention (fine-tuned format) — unchanged.
    match = re.search(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # 2. "X out of 5" / "X/5" — take the numerator (the rating), not the 5.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out\s+of|/)\s*5\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # 3. "score of X" / "rating of X" / "rating is X" / "rating: X" / "rate ...
    #    as X" / "rate it (a) X".
    match = re.search(
        r"(?:score|rating)\s*(?:of|is|:|=)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    match = re.search(
        r"rate\s+(?:it|the\s+\w+(?:\s+\w+)?)\s+(?:as\s+)?(?:a\s+)?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    # No confident match — honest parse failure.
    return None


def compute_caption_metrics(
    hyps: List[str],
    refs: List[str],
    bertscore_model: str = BERTSCORE_MODEL,
) -> Dict[str, Any]:
    """Compute lexical and semantic caption-quality metrics.

    Three complementary views of how close each predicted caption is to its
    reference:

    - BLEU (sacrebleu corpus): n-gram precision, the surface phrasing overlap.
      Reported cased and lowercased. Corpus-level aggregation.
    - ROUGE-1/2/L (rouge-score): n-gram recall plus longest-common-subsequence.
      The recall-side mirror of BLEU. Per-sample F1, then averaged.
    - BERTScore P/R/F1 (bert-score): token-embedding cosine similarity, the only
      semantic (synonym-aware) view. Per-sample, then averaged.

    Args:
        hyps: Predicted captions.
        refs: Reference captions, aligned 1:1 with ``hyps``.
        bertscore_model: HuggingFace backbone for BERTScore (recorded in output).

    Returns:
        Flat dict of metric name to score. BLEU on the 0-100 sacrebleu scale;
        ROUGE and BERTScore on 0-1.
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps/refs length mismatch: {len(hyps)} vs {len(refs)}")

    # Imported lazily so the module loads even when these extras are absent
    # (e.g. on a node where only MOS/BLEU is needed).
    from rouge_score import rouge_scorer
    from bert_score import score as bertscore_fn

    # BLEU: corpus-level n-gram precision, cased and lowercased.
    bleu_cased = sacrebleu.corpus_bleu(hyps, [refs]).score
    bleu_lc = sacrebleu.corpus_bleu(
        [h.lower() for h in hyps], [[r.lower() for r in refs]]
    ).score

    # ROUGE: per-sample F1 for unigram, bigram, and LCS overlap, then averaged.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for hyp, ref in zip(hyps, refs):
        scores = scorer.score(ref, hyp)  # (target, prediction) order
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)

    n = max(len(hyps), 1)
    rouge1_f = sum(rouge1) / n
    rouge2_f = sum(rouge2) / n
    rougel_f = sum(rougel) / n

    # BERTScore: semantic similarity via contextual embeddings, averaged.
    logging.info("Computing BERTScore with backbone %s...", bertscore_model)
    bs_p, bs_r, bs_f1 = bertscore_fn(
        hyps,
        refs,
        model_type=bertscore_model,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )

    return {
        "bleu": bleu_cased,
        "bleu_lowercased": bleu_lc,
        "rouge1_f": rouge1_f,
        "rouge2_f": rouge2_f,
        "rougeL_f": rougel_f,
        "bertscore_precision": bs_p.mean().item(),
        "bertscore_recall": bs_r.mean().item(),
        "bertscore_f1": bs_f1.mean().item(),
        "bertscore_model": bertscore_model,
        # BLEU is corpus-level (sacrebleu); ROUGE and BERTScore are per-sample
        # then averaged. Recorded so cited numbers stay comparable across runs.
        "caption_metric_aggregation": {
            "bleu": "corpus",
            "rouge": "sample_mean_f1",
            "bertscore": "sample_mean",
        },
    }


def log_caption_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print the caption metrics produced by ``compute_caption_metrics``."""
    logging.info("BLEU (corpus, cased):       %.2f", metrics["bleu"])
    logging.info("BLEU (corpus, lowercased):  %.2f", metrics["bleu_lowercased"])
    logging.info(
        "ROUGE-1 / -2 / -L F1 (mean): %.4f / %.4f / %.4f",
        metrics["rouge1_f"],
        metrics["rouge2_f"],
        metrics["rougeL_f"],
    )
    logging.info(
        "BERTScore P / R / F1 (mean, %s): %.4f / %.4f / %.4f",
        metrics["bertscore_model"],
        metrics["bertscore_precision"],
        metrics["bertscore_recall"],
        metrics["bertscore_f1"],
    )


def mos_regression_metrics(errors: Sequence[float], total: int) -> Dict[str, Any]:
    """Aggregate MOS absolute errors into parse-rate-aware MAE/MSE.

    MAE/MSE are computed over parsed samples only (``errors`` holds one absolute
    error per sample whose MOS parsed). An unparsed prediction is an honest
    failure surfaced by the parse rate, never a silent zero. When every sample
    parses, parsed == total and these match a naive over-all-samples mean.

    Args:
        errors: Absolute MOS errors, one per sample whose MOS parsed.
        total: Total number of samples considered (parsed + unparsed).

    Returns:
        Dict with ``mae``, ``mse``, ``parsed`` and ``parse_rate``. ``mae``/``mse``
        are ``nan`` when nothing parsed.
    """
    errors = list(errors)
    parsed = len(errors)
    parse_rate = parsed / max(total, 1)
    mae = sum(errors) / parsed if parsed else float("nan")
    mse = sum(error**2 for error in errors) / parsed if parsed else float("nan")
    return {"mae": mae, "mse": mse, "parsed": parsed, "parse_rate": parse_rate}


def diversity_metrics(hyps: Sequence[str]) -> Dict[str, Any]:
    """Output-diversity diagnostics: is the model answering or emitting a canon?

    ``unique_predictions`` and ``top_prediction_frequency`` (share of the single
    most common output) catch caption mode-collapse, the textual parallel to the
    temporal "unique intervals" honesty check.
    """
    hyps = list(hyps)
    return {
        "unique_predictions": len(set(hyps)),
        "top_prediction_frequency": (
            max(Counter(hyps).values()) / max(len(hyps), 1) if hyps else 0.0
        ),
    }
