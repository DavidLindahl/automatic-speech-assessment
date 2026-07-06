"""Temporal-localization scoring: interval parsing, t-IoU, and baselines.

The temporal model emits a quality caption followed by a localized degradation
window. This module owns everything about the window: parsing it out of free
text or timestamp tokens, comparing it to the construction-time ground truth via
t-IoU, and the audio-blind baselines that bound the no-information regime. The
caption/MOS half is scored by :mod:`asa.eval.metrics`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from asa.audio import AUDIO_PLACEHOLDER, AUDIO_SPECIAL
from asa.eval.metrics import mean_or_zero
from asa.prompts import PROMPT_TEMPLATE

# Matches a discrete anchor or offset time token, e.g. ``<a2>`` or ``<f9>``, used
# only to strip the localization clause out of a caption before caption-quality
# scoring (see strip_time_tokens_for_caption). The interval itself is parsed
# separately, by value, via the timestamp regexes below.
ANCHOR_OFFSET_TOKEN_RE = re.compile(r"<[af]\d+>")

TIMESTAMP_TOKEN_RE = re.compile(r"<\|(-?\d+(?:\.\d+)?)\|>")
NON_TIMESTAMP_SPECIAL_TOKEN_RE = re.compile(r"<\|(?!-?\d+(?:\.\d+)?\|>)[^|]*\|>")
PLAIN_FLOAT_RE = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])")
RANGE_PATTERNS = [
    re.compile(
        r"(?:between|from)\s+(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?"
        r"\s*(?:and|to|-)\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*(?:to|-)\s*"
        r"(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?",
        re.IGNORECASE,
    ),
]


@dataclass(frozen=True)
class Interval:
    """Time interval in seconds."""

    start: float
    end: float


def _safe_float(value: Any) -> Optional[float]:
    """Parse float-like values safely.

    Args:
        value: Any numeric-like value.

    Returns:
        Parsed float, or ``None`` when parsing fails.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _sanitize_interval(
    start: float,
    end: float,
    duration_seconds: Optional[float],
) -> Optional[Interval]:
    """Normalize, clamp, and validate an interval.

    Args:
        start: Interval start.
        end: Interval end.
        duration_seconds: Optional clip duration for clamping.

    Returns:
        A valid interval, or ``None`` if invalid.
    """
    if end < start:
        start, end = end, start

    if duration_seconds is not None and duration_seconds > 0:
        start = max(0.0, min(start, duration_seconds))
        end = max(0.0, min(end, duration_seconds))
    else:
        start = max(0.0, start)
        end = max(0.0, end)

    if end <= start:
        return None
    return Interval(start=start, end=end)


def _extract_interval_from_anchor_offset(
    text: str,
    duration_seconds: Optional[float],
) -> Optional[Interval]:
    """Extract interval from TimeAudio-style ``<aN><fK>`` token pairs."""
    from asa.temporal_tokens import decode_all_times

    matches = decode_all_times(text)
    if len(matches) < 2:
        return None
    return _sanitize_interval(matches[0], matches[1], duration_seconds)


def _extract_interval_from_tags(
    text: str,
    duration_seconds: Optional[float],
) -> Optional[Interval]:
    """Extract interval from ``<|...|>`` timestamp tokens."""
    matches = [float(m) for m in TIMESTAMP_TOKEN_RE.findall(text)]
    if len(matches) < 2:
        return None
    return _sanitize_interval(matches[0], matches[1], duration_seconds)


def extract_interval(
    text: str,
    duration_seconds: Optional[float],
    allow_plain: bool = True,
) -> tuple[Optional[Interval], str]:
    """Extract one timestamp interval from model text.

    Args:
        text: Input text to parse.
        duration_seconds: Optional clip duration for clamping and validity checks.
        allow_plain: When ``True`` (default, used for fine-tuned models), fall
            back to "any two plain floats" if no explicit timestamp token or
            range phrasing is found. When ``False`` (zero-shot baseline), this
            fallback is suppressed. The plain fallback manufactures an interval
            from any two numbers in the text, so a free-text answer like "I'd
            rate this 3 out of 5; the 6 second clip..." would yield a bogus
            (3.0, 5.0) interval and a non-zero t-IoU. For the untrained baseline
            that turns "the model emitted no localizable range" into a fake hit,
            inflating both parse rate and t-IoU. A low parse rate is the honest,
            defensible result for a baseline, so zero-shot parses only via the
            explicit ``range`` path. This mirrors the None-on-failure choice in
            the MOS ``extract_mos``.

    Returns:
        Tuple ``(interval, source)`` where source indicates parse strategy.
    """
    from_anchor_offset = _extract_interval_from_anchor_offset(text, duration_seconds)
    if from_anchor_offset is not None:
        return from_anchor_offset, "anchor_offset"

    from_tags = _extract_interval_from_tags(text, duration_seconds)
    if from_tags is not None:
        return from_tags, "token"



    return None, "none"


def strip_non_timestamp_special_tokens(text: str) -> str:
    """Remove generated control tokens while keeping timestamp tokens.

    Args:
        text: Decoded model text that may contain Qwen control tokens.

    Returns:
        Text with non-timestamp ``<|...|>`` tokens removed.
    """
    cleaned = NON_TIMESTAMP_SPECIAL_TOKEN_RE.sub("", text)
    return " ".join(cleaned.split())


def strip_time_tokens_for_caption(text: str) -> str:
    """Remove timestamp tokens so caption metrics score the prose, not the time.

    The joint temporal target is a quality caption followed by a localization
    clause, e.g. ``"... overall quality. The degradation in the clip is between
    <a0><f9> and <a2><f0>."``. Caption BLEU/ROUGE/BERTScore should reflect the
    descriptive content only, independent of whether the model placed the
    interval correctly (that is what temporal IoU measures). This strips both
    timestamp encodings, the discrete anchor/offset tokens ``<aN><fK>`` and the
    free-text ``<|float|>`` tokens, from a caption. It is applied identically to
    the prediction and the reference, so the residual "... is between and."
    scaffolding contributes the same n-grams to both sides and does not bias the
    comparison; what is removed is exactly the part IoU already scores.

    Args:
        text: A caption that may contain a trailing localization clause.

    Returns:
        The caption with timestamp tokens removed and whitespace collapsed.
    """
    cleaned = ANCHOR_OFFSET_TOKEN_RE.sub("", text)
    cleaned = TIMESTAMP_TOKEN_RE.sub("", cleaned)
    # Tidy the punctuation left dangling by the removed tokens ("between  and .")
    # so it does not introduce spurious tokens; cosmetic and symmetric.
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    return " ".join(cleaned.split())


def interval_iou(pred: Interval, truth: Interval) -> float:
    """Compute temporal IoU.

    Args:
        pred: Predicted interval.
        truth: Ground-truth interval.

    Returns:
        Temporal intersection-over-union score.
    """
    intersection_start = max(pred.start, truth.start)
    intersection_end = min(pred.end, truth.end)
    if intersection_end <= intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (pred.end - pred.start) + (truth.end - truth.start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def interval_offset_error(pred: Interval, truth: Interval) -> float:
    """Compute signed expected endpoint offset in seconds.

    Positive values mean the predicted endpoints are late on average; negative
    values mean they are early on average.
    """
    return ((pred.start - truth.start) + (pred.end - truth.end)) / 2


def whole_clip_baseline_mean_tiou(
    truths: List[Interval],
    durations: List[Optional[float]],
) -> float:
    """Mean t-IoU of the audio-blind strategy that predicts the whole clip.

    For every sample the prediction is ``[0, duration]``. This strategy never
    reads the audio; any model scoring at or below it has not demonstrated
    audio-conditioned localization. Samples without a known duration are
    skipped.

    Args:
        truths: Ground-truth intervals, one per evaluable sample.
        durations: Clip durations aligned with ``truths``.

    Returns:
        Mean t-IoU of the whole-clip prediction over samples with a duration.
    """
    ious: List[float] = []
    for truth, duration in zip(truths, durations):
        if duration is None or duration <= 0:
            continue
        ious.append(interval_iou(Interval(start=0.0, end=duration), truth))
    return mean_or_zero(ious)


def best_constant_baseline(
    truths: List[Interval],
    start_step: float = 0.25,
    length_min: float = 0.5,
    length_max: float = 4.0,
    length_step: float = 0.25,
) -> tuple[Optional[Interval], float]:
    """Grid-search the strongest constant interval for a truth distribution.

    Finds the single fixed ``[start, end]`` guess that maximizes mean t-IoU
    when applied unchanged to every sample. Like the whole-clip rule it never
    reads the audio, but it is fit on the evaluated set itself, so it is the
    oracle ceiling of audio-blind play: the upper edge of the no-information
    regime. A model below this number has learned less than a lookup of the
    interval prior.

    Args:
        truths: Ground-truth intervals to fit against.
        start_step: Grid resolution for the candidate start time in seconds.
        length_min: Smallest candidate window length in seconds.
        length_max: Largest candidate window length in seconds.
        length_step: Grid resolution for the candidate window length.

    Returns:
        Tuple of the best constant interval (``None`` when no truths are
        given) and its mean t-IoU.
    """
    if not truths:
        return None, 0.0

    max_start = max(truth.end for truth in truths)
    best_interval: Optional[Interval] = None
    best_score = -1.0

    start = 0.0
    while start <= max_start:
        length = length_min
        while length <= length_max + 1e-9:
            candidate = Interval(start=start, end=start + length)
            score = mean_or_zero([interval_iou(candidate, truth) for truth in truths])
            if score > best_score:
                best_score = score
                best_interval = candidate
            length += length_step
        start += start_step

    return best_interval, max(best_score, 0.0)


def query_to_prompt(query: Any) -> str:
    """Convert a dataset query string into a Qwen2-Audio prompt.

    Args:
        query: Stored query field from processed records.

    Returns:
        Prompt with the expected audio special tokens.
    """
    if not isinstance(query, str):
        return PROMPT_TEMPLATE

    text = " ".join(query.strip().split())
    if not text:
        return PROMPT_TEMPLATE
    if AUDIO_PLACEHOLDER in text:
        return text.replace(AUDIO_PLACEHOLDER, AUDIO_SPECIAL)
    if "<|AUDIO|>" in text:
        return text
    return f"{AUDIO_SPECIAL}{text}"


def extract_ground_truth_interval(
    record: dict[str, Any],
) -> tuple[Optional[Interval], str]:
    """Read ground-truth interval from a temporal record.

    Args:
        record: Processed JSON/JSONL record.

    Returns:
        Tuple ``(interval, source)`` where source names extraction method.
    """
    duration = _safe_float(record.get("duration_seconds"))
    segments = record.get("mix_deg_segments")
    if isinstance(segments, list):
        valid: list[Interval] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = _safe_float(segment.get("start"))
            end = _safe_float(segment.get("end"))
            if start is None or end is None:
                continue
            interval = _sanitize_interval(start, end, duration)
            if interval is not None:
                valid.append(interval)

        if valid:
            longest = max(valid, key=lambda item: item.end - item.start)
            return longest, "mix_deg_segments"

    response_text = str(record.get("response", ""))
    from_text, source = extract_interval(response_text, duration)
    if from_text is not None:
        return from_text, f"response_{source}"
    return None, "none"
