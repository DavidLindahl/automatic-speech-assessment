"""Evaluation metrics for zero-shot SALMONN outputs."""

from __future__ import annotations

import re
from typing import Sequence

import nltk
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import pearsonr, spearmanr


def ensure_nltk_tokenizer() -> None:
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def bleu_score(reference: str, hypothesis: str) -> float:
    """Compute sentence BLEU for two texts."""
    ensure_nltk_tokenizer()
    try:
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    except Exception:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
    return float(sentence_bleu([ref_tokens], hyp_tokens))


def extract_mos(text: str) -> float:
    """Extract numeric MOS score from a generated response."""
    match = re.search(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if matches:
        return float(matches[-1])
    return 0.0


def extract_winner(text: str) -> str:
    """Extract winner label from A/B response text."""
    lowered = text.lower()

    pref = re.search(
        r"\b(?:select|prefer|choose|better|winner)\b.{0,30}speech\s*([ab])\b",
        lowered,
    )
    if pref:
        return pref.group(1).upper()

    shortpref = re.search(
        r"\b(?:select|prefer|choose|better|winner)\b.{0,10}\b([ab])\b",
        lowered,
    )
    if shortpref:
        return shortpref.group(1).upper()

    if re.search(r"\b(tie|draw|equal|same quality)\b", lowered):
        return "Tie"

    labels = re.findall(r"\bspeech\s*([ab])\b", lowered)
    if labels:
        return labels[-1].upper()

    last = re.findall(r"\b([ab])\b", lowered)
    if last:
        return last[-1].upper()

    return "Tie"


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean for non-empty sequences."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def lcc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    return float(pearsonr(y_true, y_pred).statistic)


def srcc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(y_true) < 2:
        return 0.0
    return float(spearmanr(y_true, y_pred).statistic)
