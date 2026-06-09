"""Temporal localization evaluation for Qwen2-Audio checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, List, Optional

import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from asa.data import AUDIO_PLACEHOLDER, AUDIO_SPECIAL, PROMPT_TEMPLATE
from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records, resolve_audio_path
from asa.prompts import build_zeroshot_prompt_temporal
from asa.temporal_tokens import decode_all_times

EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.9
EVAL_MAX_NEW_TOKENS = 150

# Off-the-shelf (untrained) Qwen2-Audio chat model, used as the zero-shot
# temporal baseline. The source paper reports off-the-shelf audio LLMs cannot do
# speech quality assessment without fine-tuning; the --zero-shot row reproduces
# that finding for the temporal-localization task (t-IoU floor before training).
ZEROSHOT_BASELINE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(
    help="Evaluate fine-tuned Qwen2-Audio models on temporal localization."
)


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


def _extract_interval_from_ranges(
    text: str,
    duration_seconds: Optional[float],
) -> Optional[Interval]:
    """Extract interval from range-style expressions."""
    for pattern in RANGE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start = _safe_float(match.group(1))
        end = _safe_float(match.group(2))
        if start is None or end is None:
            continue
        candidate = _sanitize_interval(start, end, duration_seconds)
        if candidate is not None:
            return candidate
    return None


def _extract_interval_from_plain_numbers(
    text: str,
    duration_seconds: Optional[float],
) -> Optional[Interval]:
    """Extract interval from plain numeric text when explicit ranges are absent."""
    text_without_timestamp_tokens = TIMESTAMP_TOKEN_RE.sub(" ", text)
    values = [float(m) for m in PLAIN_FLOAT_RE.findall(text_without_timestamp_tokens)]
    if len(values) < 2:
        return None

    for left, right in zip(values, values[1:]):
        candidate = _sanitize_interval(left, right, duration_seconds)
        if candidate is not None:
            return candidate

    for i, left in enumerate(values[:-1]):
        for right in values[i + 1 :]:
            candidate = _sanitize_interval(left, right, duration_seconds)
            if candidate is not None:
                return candidate
    return None


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

    from_range = _extract_interval_from_ranges(text, duration_seconds)
    if from_range is not None:
        return from_range, "range"

    if allow_plain:
        from_plain = _extract_interval_from_plain_numbers(text, duration_seconds)
        if from_plain is not None:
            return from_plain, "plain"

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


def _mean(values: list[float]) -> float:
    """Mean helper that returns zero for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


@app.command()
def eval_temporal(
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Paths to the test JSONL datasets."
    ),
    model_path: Optional[str] = typer.Option(
        None,
        help=(
            "Hub repo ID or local checkpoint path. Defaults to the fine-tuned "
            "SFT model, or to the off-the-shelf Instruct baseline under "
            "--zero-shot."
        ),
    ),
    zero_shot: bool = typer.Option(
        False,
        "--zero-shot",
        help=(
            "Evaluate the off-the-shelf (untrained) Qwen2-Audio-7B-Instruct "
            "baseline. Uses a ChatML chat-template prompt instead of each "
            "record's bare query prompt, and suppresses the plain-number "
            "interval fallback so a rambling answer cannot manufacture a bogus "
            "interval. This is the defensible 'before fine-tuning' t-IoU floor."
        ),
    ),
    data_root: Path = typer.Option(
        Path("data"),
        help="Root directory used to resolve audio paths.",
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to evaluate (for testing)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Dir to save results. Defaults to results/evaluation/<model_name>_temporal.",
    ),
    batch_size: int = typer.Option(4, help="Inference batch size."),
    use_query_prompt: bool = typer.Option(
        True,
        "--use-query-prompt/--use-default-prompt",
        help="Use each record's temporal query prompt instead of the default prompt.",
    ),
    do_sample: bool = typer.Option(
        False,
        "--do-sample/--greedy",
        help="Sample with temperature/top_p or use greedy decoding.",
    ),
    temperature: float = typer.Option(
        EVAL_TEMPERATURE,
        help="Sampling temperature (used when --do-sample).",
    ),
    top_p: float = typer.Option(
        EVAL_TOP_P,
        help="Nucleus top-p (used when --do-sample).",
    ),
    max_new_tokens: int = typer.Option(
        EVAL_MAX_NEW_TOKENS,
        help="Max new tokens to generate per sample.",
    ),
) -> None:
    """Run temporal inference and report localization quality metrics."""
    # In --zero-shot mode the model defaults to the off-the-shelf Instruct
    # baseline; otherwise it defaults to the fine-tuned SFT checkpoint.
    if model_path is None:
        model_path = ZEROSHOT_BASELINE_MODEL if zero_shot else ASAModel.SFT

    if output_dir is None:
        model_name = Path(model_path).name or "model"
        output_dir = Path(f"results/evaluation/{model_name}_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s", model_path)
    processor, model, device = load_model(model_path)

    # The zero-shot baseline uses one ChatML-templated, non-leaking prompt for
    # every row (overriding the per-record query prompt), and parses predictions
    # without the plain-number fallback so a non-temporal ramble cannot fake an
    # interval. allow_plain stays True for fine-tuned models (unchanged path).
    zeroshot_prompt: Optional[str] = None
    if zero_shot:
        zeroshot_prompt = build_zeroshot_prompt_temporal(processor)
        audio_token_count = zeroshot_prompt.count("<|AUDIO|>")
        if audio_token_count != 1:
            raise ValueError(
                "Zero-shot temporal prompt must contain exactly one <|AUDIO|> "
                f"token for run_inference alignment, found {audio_token_count}."
            )
        logging.info(
            "Zero-shot temporal baseline: ChatML prompt, plain-number fallback "
            "suppressed (parse via explicit range only)."
        )

    for dataset_path in dataset_paths:
        logging.info("Loading dataset from %s", dataset_path)
        rows = load_processed_records(dataset_path)
        if max_samples is not None:
            rows = rows[:max_samples]
            logging.info("Limited evaluation to %d samples", len(rows))

        resolved_rows: list[dict[str, Any]] = []
        missing_audio_ref = 0
        missing_audio_file = 0

        for item in rows:
            audios = item.get("audios")
            if not isinstance(audios, list) or not audios:
                missing_audio_ref += 1
                continue

            raw_audio = str(audios[0])
            resolved_audio = resolve_audio_path(raw_audio, data_root)
            if not resolved_audio.exists():
                missing_audio_file += 1
                continue

            duration_seconds = _safe_float(item.get("duration_seconds"))
            truth_interval, truth_source = extract_ground_truth_interval(item)
            if zeroshot_prompt is not None:
                prompt = zeroshot_prompt
            elif use_query_prompt:
                prompt = query_to_prompt(item.get("query"))
            else:
                prompt = PROMPT_TEMPLATE
            resolved_rows.append(
                {
                    "record": item,
                    "audio_path": str(resolved_audio),
                    "duration_seconds": duration_seconds,
                    "truth_interval": truth_interval,
                    "truth_source": truth_source,
                    "prompt": prompt,
                }
            )

        if not resolved_rows:
            raise ValueError(
                "No evaluable rows found after audio path resolution. "
                f"dataset={dataset_path}, data_root={data_root}"
            )

        logging.info(
            "Running inference on %d rows (skipped: missing_audios=%d, missing_files=%d)",
            len(resolved_rows),
            missing_audio_ref,
            missing_audio_file,
        )
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=[row["audio_path"] for row in resolved_rows],
            prompt_texts=[row["prompt"] for row in resolved_rows],
            device=device,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=False,
        )

        details: list[dict[str, Any]] = []
        ious: list[float] = []
        start_errors: list[float] = []
        end_errors: list[float] = []
        parsed_count = 0
        ground_truth_count = 0
        # Tally which parse strategy produced each prediction interval. For the
        # zero-shot baseline this is the audit surface: "plain" should be absent
        # (it is suppressed), and a healthy baseline parses via "range" or not
        # at all. A surprising distribution is the smoke-gate signal.
        pred_source_counts: dict[str, int] = {}
        # Set of distinct (start, end) predicted intervals (rounded to 0.01 s).
        # This is the honesty diagnostic, the temporal parallel to the MOS
        # "unique predictions N/N" line. A near-constant interval (count of 1-2
        # across many samples) means the model emits a canned answer and any
        # nonzero t-IoU is chance overlap, NOT localization. A genuinely
        # localizing model varies its interval with the audio.
        unique_pred_intervals: set[tuple[float, float]] = set()

        for row, prediction in zip(resolved_rows, predictions):
            record = row["record"]
            duration_seconds = row["duration_seconds"]
            prediction_text = strip_non_timestamp_special_tokens(prediction)

            pred_interval, pred_source = extract_interval(
                prediction_text, duration_seconds, allow_plain=not zero_shot
            )
            truth_interval = row["truth_interval"]
            truth_source = row["truth_source"]

            pred_source_counts[pred_source] = (
                pred_source_counts.get(pred_source, 0) + 1
            )
            if pred_interval is not None:
                unique_pred_intervals.add(
                    (round(pred_interval.start, 2), round(pred_interval.end, 2))
                )
            if pred_interval is not None:
                parsed_count += 1
            if truth_interval is not None:
                ground_truth_count += 1

            tiou = 0.0
            if pred_interval is not None and truth_interval is not None:
                tiou = interval_iou(pred_interval, truth_interval)
                start_errors.append(abs(pred_interval.start - truth_interval.start))
                end_errors.append(abs(pred_interval.end - truth_interval.end))
            if truth_interval is not None:
                ious.append(tiou)

            detail = dict(record)
            detail["audio_path_resolved"] = row["audio_path"]
            detail["predicted_response"] = prediction_text
            detail["gt_interval_source"] = truth_source
            detail["pred_interval_source"] = pred_source
            detail["gt_start"] = (
                truth_interval.start if truth_interval is not None else None
            )
            detail["gt_end"] = (
                truth_interval.end if truth_interval is not None else None
            )
            detail["pred_start"] = (
                pred_interval.start if pred_interval is not None else None
            )
            detail["pred_end"] = (
                pred_interval.end if pred_interval is not None else None
            )
            detail["tiou"] = tiou
            detail["start_abs_err"] = (
                abs(pred_interval.start - truth_interval.start)
                if pred_interval is not None and truth_interval is not None
                else None
            )
            detail["end_abs_err"] = (
                abs(pred_interval.end - truth_interval.end)
                if pred_interval is not None and truth_interval is not None
                else None
            )
            details.append(detail)

        metrics = {
            "samples_total": len(resolved_rows),
            "samples_with_ground_truth_interval": ground_truth_count,
            "samples_with_parsed_prediction_interval": parsed_count,
            "skipped_missing_audios_field": missing_audio_ref,
            "skipped_missing_audio_file": missing_audio_file,
            "mean_tiou": _mean(ious),
            "median_tiou": median(ious) if ious else 0.0,
            "hit_iou_ge_0_1": _mean([1.0 if value >= 0.1 else 0.0 for value in ious]),
            "hit_iou_ge_0_3": _mean([1.0 if value >= 0.3 else 0.0 for value in ious]),
            "hit_iou_ge_0_5": _mean([1.0 if value >= 0.5 else 0.0 for value in ious]),
            "mean_start_abs_err": _mean(start_errors),
            "mean_end_abs_err": _mean(end_errors),
            "parse_rate": parsed_count / len(resolved_rows),
            "pred_interval_source_counts": pred_source_counts,
            "unique_pred_intervals": len(unique_pred_intervals),
            "prompt_mode": (
                "zero_shot_chatml"
                if zero_shot
                else ("query" if use_query_prompt else "default")
            ),
            "zero_shot": zero_shot,
            "model_path": model_path,
            "prompt": zeroshot_prompt if zero_shot else None,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        logging.info("========================================")
        logging.info("TEMPORAL EVALUATION: %s", dataset_path.name)
        logging.info("Samples evaluated: %d", metrics["samples_total"])
        logging.info("Prediction parse rate: %.4f", parsed_count / len(resolved_rows))
        logging.info("Pred interval sources: %s", pred_source_counts)
        logging.info(
            "Unique predicted intervals: %d of %d parsed (honesty diagnostic; "
            "near-1 means canned answer, not localization)",
            len(unique_pred_intervals),
            parsed_count,
        )
        logging.info("Mean t-IoU: %.4f", metrics["mean_tiou"])
        logging.info("Median t-IoU: %.4f", metrics["median_tiou"])
        logging.info("Hit@0.1: %.4f", metrics["hit_iou_ge_0_1"])
        logging.info("Hit@0.3: %.4f", metrics["hit_iou_ge_0_3"])
        logging.info("Hit@0.5: %.4f", metrics["hit_iou_ge_0_5"])
        logging.info("Mean |start error|: %.4f", metrics["mean_start_abs_err"])
        logging.info("Mean |end error|: %.4f", metrics["mean_end_abs_err"])
        logging.info("========================================")

        dataset_name = dataset_path.stem
        out_json = output_dir / f"{dataset_name}_results.json"
        out_csv = output_dir / f"{dataset_name}_results.csv"

        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {"metrics": metrics, "results": details},
                handle,
                indent=2,
                ensure_ascii=False,
            )

        csv_columns = [
            "id",
            "filename_deg",
            "mix_filename",
            "audio_path_resolved",
            "duration_seconds",
            "gt_start",
            "gt_end",
            "pred_start",
            "pred_end",
            "tiou",
            "start_abs_err",
            "end_abs_err",
            "gt_interval_source",
            "pred_interval_source",
            "mos",
            "predicted_response",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=csv_columns, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(details)

        logging.info("Saved detailed results to %s", out_json)
        logging.info("Saved tabular results to %s", out_csv)


if __name__ == "__main__":
    app()
