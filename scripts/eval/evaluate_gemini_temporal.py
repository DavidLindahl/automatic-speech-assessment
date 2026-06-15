"""Run and score the Appendix-B zero-shot temporal prompt on Gemini."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records, resolve_audio_path
from asa.prompts import ZEROSHOT_USER_TEXT_TEMPORAL
from evaluate import extract_mos
from evaluate_gemini_mos import (
    AUDIO_LABEL,
    DailyQuotaExhausted,
    MODEL_NAME,
    SEED,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
    append_prediction,
    calculate_cost_usd,
    is_daily_quota_error,
    latest_predictions,
    load_predictions,
    usage_to_dict,
)
from evaluate_gemini_mos_batch import (
    BATCH_COST_MULTIPLIER,
    load_jsonl,
    make_client,
    save_model,
    upload_with_retries,
)
from evaluate_temporal import (
    Interval,
    best_constant_baseline,
    extract_ground_truth_interval,
    extract_interval,
    interval_iou,
    whole_clip_baseline_mean_tiou,
)

DEFAULT_DATASET = Path("data/processed/temporal/test_FOR_temporal_global_caption.json")
DEFAULT_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/temporal_FOR_greedy"
)
DEFAULT_BATCH_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/temporal_FOR_batch_full"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
app = typer.Typer(help=__doc__)


def prompt_sha256() -> str:
    """Return a stable fingerprint for the exact prompt sent to Gemini."""
    payload = (
        f"{SYSTEM_INSTRUCTION}\n{AUDIO_LABEL}\n{ZEROSHOT_USER_TEXT_TEMPORAL}".encode()
    )
    return hashlib.sha256(payload).hexdigest()


def generation_config() -> types.GenerateContentConfig:
    """Return the shared temperature-zero Gemini generation configuration."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        seed=SEED,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


def build_run_config(dataset_path: Path, data_root: Path) -> dict[str, Any]:
    """Build the immutable temporal run configuration."""
    return {
        "task": "zero_shot_temporal",
        "model": MODEL_NAME,
        "dataset_path": str(dataset_path),
        "data_root": str(data_root),
        "system_instruction": SYSTEM_INSTRUCTION,
        "audio_label": AUDIO_LABEL,
        "user_prompt": ZEROSHOT_USER_TEXT_TEMPORAL,
        "prompt_sha256": prompt_sha256(),
        "decoding": {
            "mode": "greedy_like",
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": None,
            "thinking_level": "LOW",
        },
        "interval_parsing": {
            "allow_plain_numbers": False,
            "unparsed_prediction_tiou": 0.0,
        },
        "prompt_parity": (
            "Shared corrected non-leaking zero-shot temporal instruction; audio is "
            "attached through Gemini's native API instead of Qwen ChatML."
        ),
    }


def validate_rows(
    rows: list[dict[str, Any]],
    data_root: Path,
    sample_indices: Optional[list[int]] = None,
) -> list[tuple[int, dict[str, Any], Path, Interval]]:
    """Resolve WAVs and require one valid construction-time interval per row."""
    selected = set(sample_indices) if sample_indices else None
    validated: list[tuple[int, dict[str, Any], Path, Interval]] = []
    for sample_index, row in enumerate(rows):
        if selected is not None and sample_index not in selected:
            continue
        audios = row.get("audios")
        if not isinstance(audios, list) or not audios:
            raise ValueError(f"Sample {sample_index} has no audio reference.")
        audio_path = resolve_audio_path(str(audios[0]), data_root)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Sample {sample_index} audio missing: {audio_path}"
            )
        truth, source = extract_ground_truth_interval(row)
        if truth is None or source != "mix_deg_segments":
            raise ValueError(
                f"Sample {sample_index} lacks a construction-time ground truth interval."
            )
        validated.append((sample_index, row, audio_path, truth))
    if selected is not None:
        missing = selected - {item[0] for item in validated}
        if missing:
            raise ValueError(f"Sample indices outside dataset: {sorted(missing)}")
    return validated


def score_record(
    sample_index: int,
    row: dict[str, Any],
    audio_path: Path,
    truth: Interval,
    response_text: str,
    usage: dict[str, int],
    cost_multiplier: float = 1.0,
    api_mode: str = "interactive",
) -> dict[str, Any]:
    """Parse and score one successful Gemini temporal response."""
    duration = float(row["duration_seconds"])
    pred, source = extract_interval(response_text, duration, allow_plain=False)
    tiou = interval_iou(pred, truth) if pred is not None else 0.0
    predicted_mos = extract_mos(response_text)
    gold_mos = float(row["mos"])
    return {
        **row,
        "sample_index": sample_index,
        "audio_path_resolved": str(audio_path),
        "status": "ok",
        "api_mode": api_mode,
        "predicted_response": response_text.strip(),
        "gt_start": truth.start,
        "gt_end": truth.end,
        "pred_start": pred.start if pred is not None else None,
        "pred_end": pred.end if pred is not None else None,
        "pred_interval_source": source,
        "tiou": tiou,
        "start_abs_err": abs(pred.start - truth.start) if pred is not None else None,
        "end_abs_err": abs(pred.end - truth.end) if pred is not None else None,
        "predicted_mos": predicted_mos,
        "mos_error": (
            abs(predicted_mos - gold_mos) if predicted_mos is not None else None
        ),
        "usage": usage,
        "cost_usd": calculate_cost_usd(usage) * cost_multiplier,
    }


def write_results(
    output_path: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    """Write temporal metrics using the same strict parser contract as Qwen."""
    predictions = latest_predictions(predictions)
    successful = [item for item in predictions if item.get("status") == "ok"]
    interval_parsed = [
        item for item in successful if item.get("pred_start") is not None
    ]
    mos_parsed = [item for item in successful if item.get("predicted_mos") is not None]
    ious = [float(item["tiou"]) for item in successful]
    start_errors = [float(item["start_abs_err"]) for item in interval_parsed]
    end_errors = [float(item["end_abs_err"]) for item in interval_parsed]
    mos_errors = [float(item["mos_error"]) for item in mos_parsed]
    truths = [
        Interval(float(item["gt_start"]), float(item["gt_end"])) for item in successful
    ]
    durations = [float(item["duration_seconds"]) for item in successful]
    constant, constant_tiou = best_constant_baseline(truths)
    unique_intervals = {
        (round(float(item["pred_start"]), 2), round(float(item["pred_end"]), 2))
        for item in interval_parsed
    }
    responses = [str(item["predicted_response"]) for item in successful]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "samples_requested": len(predictions),
        "samples_successful": len(successful),
        "completion_rate": len(successful) / max(len(predictions), 1),
        "intervals_parsed": len(interval_parsed),
        "interval_parse_rate": len(interval_parsed) / max(len(successful), 1),
        "mean_tiou": mean(ious),
        "median_tiou": median(ious) if ious else 0.0,
        "hit_iou_ge_0_1": mean([float(value >= 0.1) for value in ious]),
        "hit_iou_ge_0_3": mean([float(value >= 0.3) for value in ious]),
        "hit_iou_ge_0_5": mean([float(value >= 0.5) for value in ious]),
        "mean_start_abs_err": mean(start_errors),
        "mean_end_abs_err": mean(end_errors),
        "pred_interval_source_counts": dict(
            Counter(str(item["pred_interval_source"]) for item in successful)
        ),
        "unique_pred_intervals": len(unique_intervals),
        "baseline_whole_clip_mean_tiou": whole_clip_baseline_mean_tiou(
            truths, durations
        ),
        "baseline_best_constant_mean_tiou": constant_tiou,
        "baseline_best_constant_interval": (
            [constant.start, constant.end] if constant is not None else None
        ),
        "mos_parsed": len(mos_parsed),
        "mos_parse_rate": len(mos_parsed) / max(len(successful), 1),
        "mos_mae": mean(mos_errors),
        "mos_mse": mean([error**2 for error in mos_errors]),
        "unique_responses": len(set(responses)),
        "cost_usd": sum(float(item.get("cost_usd", 0.0)) for item in predictions),
    }
    output_path.write_text(
        json.dumps(
            {"metrics": metrics, "run": config, "results": predictions},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def generate_one(
    client: genai.Client, audio_path: Path, max_retries: int
) -> tuple[str, dict[str, int]]:
    """Send one temporal request, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    AUDIO_LABEL,
                    types.Part.from_bytes(
                        data=audio_path.read_bytes(), mime_type="audio/wav"
                    ),
                    ZEROSHOT_USER_TEXT_TEMPORAL,
                ],
                config=generation_config(),
            )
            return (response.text or "").strip(), usage_to_dict(response.usage_metadata)
        except Exception as exc:
            if is_daily_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(
        f"Gemini request failed after retries: {last_error}"
    ) from last_error


def build_batch_request(sample_index: int, file_uri: str) -> types.InlinedRequest:
    """Build one keyed temporal Batch request."""
    return types.InlinedRequest(
        contents=[
            AUDIO_LABEL,
            types.Part.from_uri(file_uri=file_uri, mime_type="audio/wav"),
            ZEROSHOT_USER_TEXT_TEMPORAL,
        ],
        metadata={"sample_index": str(sample_index)},
        config=generation_config(),
    )


@app.command()
def run(
    dataset_path: Path = typer.Option(DEFAULT_DATASET),
    data_root: Path = typer.Option(Path("data")),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR),
    sample_indices: Optional[list[int]] = typer.Option(None, "--sample-index"),
    max_cost_usd: float = typer.Option(2.0),
    max_retries: int = typer.Option(4),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run resumable interactive temporal inference, optionally on selected rows."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    rows = load_processed_records(dataset_path)
    validated = validate_rows(rows, data_root, sample_indices)
    config = build_run_config(dataset_path, data_root)
    config["sample_indices"] = [item[0] for item in validated]
    logging.info("Validated %d temporal FOR samples.", len(validated))
    logging.info("Prompt SHA-256: %s", config["prompt_sha256"])
    if dry_run:
        logging.info("Dry run complete: no Gemini API request was sent.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment or root .env.")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Run configuration differs from existing {config_path}.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    predictions_path = output_dir / "predictions.jsonl"
    existing = load_predictions(predictions_path)
    complete = {
        int(item["sample_index"]) for item in existing if item["status"] == "ok"
    }
    accrued = sum(float(item.get("cost_usd", 0.0)) for item in existing)
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=120_000)
    )
    for sample_index, row, audio_path, truth in validated:
        if sample_index in complete or accrued >= max_cost_usd:
            continue
        logging.info("Sending sample %d: %s", sample_index, audio_path.name)
        try:
            text, usage = generate_one(client, audio_path, max_retries)
            record = score_record(sample_index, row, audio_path, truth, text, usage)
        except DailyQuotaExhausted:
            logging.warning("Daily Gemini request quota exhausted; stopping.")
            break
        except Exception as exc:
            record = {
                **row,
                "sample_index": sample_index,
                "audio_path_resolved": str(audio_path),
                "status": "error",
                "error": str(exc),
                "cost_usd": 0.0,
            }
        append_prediction(predictions_path, record)
        existing.append(record)
        accrued += float(record["cost_usd"])
        logging.info(
            "interval=%s--%s source=%s t-IoU=%.3f cost=$%.4f",
            record.get("pred_start"),
            record.get("pred_end"),
            record.get("pred_interval_source"),
            float(record.get("tiou", 0.0)),
            float(record["cost_usd"]),
        )
        time.sleep(4.1)
    write_results(output_dir / "results.json", config, existing)


@app.command()
def submit(
    dataset_path: Path = typer.Option(DEFAULT_DATASET),
    data_root: Path = typer.Option(Path("data")),
    output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR),
) -> None:
    """Upload the 179 FOR WAVs and submit one temporal Batch job."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "batch_job.json"
    if job_path.exists():
        raise ValueError(f"{job_path} already exists; refusing duplicate submission.")
    rows = load_processed_records(dataset_path)
    validated = validate_rows(rows, data_root)
    config = build_run_config(dataset_path, data_root)
    config.update(
        api_mode="batch",
        batch_cost_multiplier=BATCH_COST_MULTIPLIER,
        samples_submitted=len(validated),
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    client = make_client()
    uploads_path = output_dir / "uploads.jsonl"
    uploads = {int(item["sample_index"]): item for item in load_jsonl(uploads_path)}
    for sample_index, _row, audio_path, _truth in validated:
        if sample_index in uploads:
            continue
        logging.info(
            "Uploading [%d/%d] %s", len(uploads) + 1, len(validated), audio_path.name
        )
        uploaded = upload_with_retries(
            client, audio_path, f"FOR-temporal-{sample_index:03d}-{audio_path.name}"
        )
        record = {
            "sample_index": sample_index,
            "audio_path": str(audio_path),
            "file_name": uploaded.name,
            "file_uri": uploaded.uri,
            "mime_type": uploaded.mime_type,
            "size_bytes": uploaded.size_bytes,
            "expiration_time": (
                uploaded.expiration_time.isoformat()
                if uploaded.expiration_time is not None
                else None
            ),
        }
        append_prediction(uploads_path, record)
        uploads[sample_index] = record
    requests = [
        build_batch_request(index, uploads[index]["file_uri"])
        for index, _row, _audio, _truth in validated
    ]
    job = client.batches.create(
        model=MODEL_NAME,
        src=requests,
        config=types.CreateBatchJobConfig(display_name="FOR temporal zero-shot full"),
    )
    save_model(job_path, job)
    logging.info("Submitted Batch job %s in state %s.", job.name, job.state)


@app.command()
def status(output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR)) -> None:
    """Fetch and save the latest Batch job state."""
    record = json.loads((output_dir / "batch_job.json").read_text())
    client = make_client()
    job = client.batches.get(name=record["name"])
    save_model(output_dir / "batch_job_status.json", job)
    logging.info("Batch job %s: %s", job.name, job.state)


@app.command("import-results")
def import_results(
    dataset_path: Path = typer.Option(DEFAULT_DATASET),
    data_root: Path = typer.Option(Path("data")),
    output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR),
) -> None:
    """Import and score a completed temporal Batch job."""
    rows = load_processed_records(dataset_path)
    validated = validate_rows(rows, data_root)
    by_index = {item[0]: item[1:] for item in validated}
    job_record = json.loads((output_dir / "batch_job.json").read_text())
    client = make_client()
    job = client.batches.get(name=job_record["name"])
    save_model(output_dir / "batch_job_status.json", job)
    if not job.done or job.dest is None or not job.dest.inlined_responses:
        raise ValueError(f"Batch job is not ready: {job.state} {job.error}")
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise ValueError(f"{predictions_path} exists; refusing duplicate import.")
    for item in job.dest.inlined_responses:
        sample_index = int(item.metadata["sample_index"])
        row, audio_path, truth = by_index[sample_index]
        if item.error is not None or item.response is None:
            record = {
                **row,
                "sample_index": sample_index,
                "audio_path_resolved": str(audio_path),
                "status": "error",
                "api_mode": "batch",
                "error": str(item.error),
                "cost_usd": 0.0,
            }
        else:
            usage = usage_to_dict(item.response.usage_metadata)
            record = score_record(
                sample_index,
                row,
                audio_path,
                truth,
                item.response.text or "",
                usage,
                BATCH_COST_MULTIPLIER,
                "batch",
            )
        append_prediction(predictions_path, record)
    config = json.loads((output_dir / "run_config.json").read_text())
    write_results(output_dir / "results.json", config, load_jsonl(predictions_path))
    logging.info("Imported %d Batch responses.", len(load_jsonl(predictions_path)))


if __name__ == "__main__":
    app()
