"""Prepare and score temporal zero-shot runs for external audio LLMs.

The external models are not loaded here. This CLI only handles the stable
ASA-facing contract:

1. Convert ASA temporal JSON/JSONL into an external annotation JSON.
2. Score an external model's prediction JSON against the original ASA records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any

import typer

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    sys.path.append(str(repo_root / "src"))

from asa.processed_data import load_processed_records, resolve_audio_path
from scripts.eval.evaluate_temporal import (
    _safe_float as safe_float,
    extract_ground_truth_interval,
    extract_interval,
    interval_iou,
    strip_non_timestamp_special_tokens,
)

DEFAULT_QUESTION = (
    "Please describe and evaluate the synthetic speech, and find timestamps "
    "for the degradation."
)
PREDICTION_FIELDS = (
    "model_prediction",
    "prediction",
    "predicted_response",
    "response",
    "answer",
)

app = typer.Typer(
    help="Prepare and score temporal zero-shot inference for external audio LLMs."
)


def _dataset_output_path(output_dir: Path, dataset_path: Path, suffix: str) -> Path:
    return output_dir / f"{dataset_path.stem}_{suffix}.json"


def _interval_answer(record: dict[str, Any]) -> str:
    interval, _ = extract_ground_truth_interval(record)
    if interval is None:
        return "No localized degradation interval is available."
    return (
        "The localized degradation occurs between "
        f"{interval.start:.2f} - {interval.end:.2f} seconds."
    )


def _record_id(record: dict[str, Any], dataset_path: Path, index: int) -> str:
    raw_id = record.get("id")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    return f"{dataset_path.stem}_{index:05d}"


def query_to_plain_question(query: Any) -> str | None:
    if not isinstance(query, str):
        return None
    text = query.replace("<audio>", "").replace("<|AUDIO|>", "").strip()
    return " ".join(text.split())


def _question_for_record(record: dict[str, Any], question: str | None) -> str:
    if question is not None:
        return question
    query_question = query_to_plain_question(record.get("query"))
    return query_question or DEFAULT_QUESTION


def _prediction_text(record: dict[str, Any], prediction_field: str | None) -> str:
    if prediction_field is not None:
        return str(record.get(prediction_field, ""))
    for field in PREDICTION_FIELDS:
        value = record.get(field)
        if value is not None:
            return str(value)
    return ""


def score_temporal_predictions(
    records: list[dict[str, Any]],
    prediction_by_id: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ious: list[float] = []
    offset_errors: list[float] = []
    parsed_count = 0
    ground_truth_count = 0
    details: list[dict[str, Any]] = []

    for record in records:
        record_id = str(record.get("id", ""))
        duration = safe_float(record.get("duration_seconds"))

        # Extract ground truth
        truth_interval, truth_source = extract_ground_truth_interval(record)
        if truth_interval is not None:
            ground_truth_count += 1

        # Match prediction
        pred_text = prediction_by_id.get(record_id, "")
        pred_text = strip_non_timestamp_special_tokens(pred_text)

        # Extract prediction
        pred_interval, pred_source = extract_interval(pred_text, duration)
        if pred_interval is not None:
            parsed_count += 1

        # Compute metrics
        tiou = 0.0
        if pred_interval is not None and truth_interval is not None:
            tiou = interval_iou(pred_interval, truth_interval)
            start_err = abs(pred_interval.start - truth_interval.start)
            end_err = abs(pred_interval.end - truth_interval.end)
            offset_errors.append((start_err + end_err) / 2.0)
        if truth_interval is not None:
            ious.append(tiou)

        # Build detail dict
        detail = dict(record)
        detail["predicted_response"] = pred_text
        detail["gt_interval_source"] = truth_source
        detail["pred_interval_source"] = pred_source
        detail["gt_start"] = (
            truth_interval.start if truth_interval is not None else None
        )
        detail["gt_end"] = truth_interval.end if truth_interval is not None else None
        detail["pred_start"] = (
            pred_interval.start if pred_interval is not None else None
        )
        detail["pred_end"] = pred_interval.end if pred_interval is not None else None
        detail["tiou"] = tiou
        if pred_interval is not None and truth_interval is not None:
            start_err = abs(pred_interval.start - truth_interval.start)
            end_err = abs(pred_interval.end - truth_interval.end)
            detail["offset_abs_err"] = (start_err + end_err) / 2.0
        else:
            detail["offset_abs_err"] = None
        details.append(detail)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "samples_total": len(records),
        "samples_with_ground_truth_interval": ground_truth_count,
        "samples_with_parsed_prediction_interval": parsed_count,
        "mean_tiou": _mean(ious),
        "median_tiou": median(ious) if ious else 0.0,
        "hit_iou_ge_0_1": _mean([1.0 if v >= 0.1 else 0.0 for v in ious]),
        "hit_iou_ge_0_3": _mean([1.0 if v >= 0.3 else 0.0 for v in ious]),
        "hit_iou_ge_0_5": _mean([1.0 if v >= 0.5 else 0.0 for v in ious]),
        "mean_offset_abs_err": _mean(offset_errors),
    }

    return metrics, details


@app.command()
def prepare(
    dataset_paths: list[Path] = typer.Option(
        ..., "--dataset-path", help="ASA temporal JSON/JSONL dataset path."
    ),
    output_dir: Path = typer.Option(
        Path("results/evaluation/temporal/external_zero_shot/inputs"),
        help="Directory for external annotation JSON files.",
    ),
    data_root: Path = typer.Option(
        Path("data"),
        help="Root directory used to resolve ASA audio paths.",
    ),
    model_format: str = typer.Option(
        "timeaudio",
        help="External annotation flavor. Currently: timeaudio or salmonn.",
    ),
    question: str | None = typer.Option(
        None,
        help="Override the question sent to the external model.",
    ),
    max_samples: int | None = typer.Option(
        None,
        help="Optional smoke-test limit per dataset.",
    ),
    relative_audio_paths: bool = typer.Option(
        False,
        "--relative-audio-paths/--absolute-audio-paths",
        help="Write repo-relative audio paths instead of absolute paths.",
    ),
) -> None:
    """Convert ASA temporal datasets into external-model annotation JSON."""
    if model_format not in {"timeaudio", "salmonn"}:
        raise ValueError("model_format must be one of: timeaudio, salmonn")

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_path in dataset_paths:
        rows = load_processed_records(dataset_path)
        if max_samples is not None:
            rows = rows[:max_samples]

        external_rows: list[dict[str, Any]] = []
        skipped_missing_audio = 0
        for index, record in enumerate(rows):
            audios = record.get("audios")
            if not isinstance(audios, list) or not audios:
                skipped_missing_audio += 1
                continue

            resolved_audio = resolve_audio_path(str(audios[0]), data_root)
            if not resolved_audio.exists():
                skipped_missing_audio += 1
                continue

            audio_value = (
                str(audios[0])
                if relative_audio_paths
                else str(resolved_audio.resolve())
            )
            duration = safe_float(record.get("duration_seconds"))
            external_rows.append(
                {
                    "id": _record_id(record, dataset_path, index),
                    "audio": audio_value,
                    "question": _question_for_record(record, question),
                    "answer": _interval_answer(record),
                    "duration": duration if duration is not None else 0.0,
                    "source_dataset": str(dataset_path),
                    "source_model_format": model_format,
                }
            )

        if rows and not external_rows:
            raise ValueError(
                "No records had resolvable audio files. "
                f"dataset={dataset_path}, data_root={data_root}, "
                f"skipped_missing_audio={skipped_missing_audio}"
            )

        output_path = _dataset_output_path(output_dir, dataset_path, model_format)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(external_rows, handle, indent=2, ensure_ascii=False)

        typer.echo(
            f"Wrote {len(external_rows)} records to {output_path} "
            f"(skipped_missing_audio={skipped_missing_audio})"
        )


@app.command()
def score(
    dataset_path: Path = typer.Option(
        ..., "--dataset-path", help="Original ASA temporal JSON/JSONL dataset path."
    ),
    prediction_path: Path = typer.Option(
        ..., "--prediction-path", help="External model prediction JSON/JSONL path."
    ),
    output_json: Path = typer.Option(
        ..., "--output-json", help="Metrics and detailed result JSON path."
    ),
    output_csv: Path | None = typer.Option(
        None, "--output-csv", help="Optional detailed result CSV path."
    ),
    prediction_field: str | None = typer.Option(
        None,
        help="Prediction field name. Auto-detects common fields when omitted.",
    ),
) -> None:
    """Score external temporal predictions against one ASA temporal dataset."""
    records = load_processed_records(dataset_path)
    predictions = load_processed_records(prediction_path)
    prediction_by_id = {
        str(item.get("id", item.get("id_audio", ""))): _prediction_text(
            item, prediction_field
        )
        for item in predictions
    }

    metrics, details = score_temporal_predictions(records, prediction_by_id)
    metrics.update(
        {
            "dataset_path": str(dataset_path),
            "prediction_path": str(prediction_path),
            "predictions_loaded": len(predictions),
            "predictions_matched_by_id": sum(
                1 for record in records if str(record.get("id", "")) in prediction_by_id
            ),
        }
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metrics": metrics, "results": details},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_columns = [
            "id",
            "filename_deg",
            "mix_filename",
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
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=csv_columns, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(details)

    typer.echo(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
