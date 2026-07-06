"""Zero-shot MOS baseline on Gemini: interactive + Batch, resumable and costed.

Thin entrypoint over :mod:`asa.eval.gemini_api` (client, quota, cost, JSONL
resume, upload/batch helpers) and :mod:`asa.eval.metrics` (MOS parse + caption
metrics). This file supplies only the MOS-specific parts: the Appendix-B prompt,
the run-config, the per-record scoring, and the correlation metrics. Prompt
parity with the Qwen zero-shot row makes the Gemini number a comparable baseline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import typer
from google import genai
from google.genai import types
from scipy.stats import pearsonr, spearmanr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.eval.gemini_api import (
    AUDIO_LABEL,
    BATCH_COST_MULTIPLIER,
    MODEL_NAME,
    SEED,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
    DailyQuotaExhausted,
    append_prediction,
    calculate_cost_usd,
    generation_config,
    latest_predictions,
    load_jsonl,
    load_predictions,
    make_client,
    save_model,
    upload_with_retries,
    usage_to_dict,
)
from asa.eval.metrics import compute_caption_metrics, extract_mos
from asa.processed_data import load_processed_records, resolve_audio_path
from asa.prompts import ZEROSHOT_USER_TEXT_MOS

INPUT_USD_PER_MILLION_TOKENS = 2.0
OUTPUT_USD_PER_MILLION_TOKENS = 12.0

DEFAULT_DATASET = Path("data/processed/eval/test_FOR.json")
DEFAULT_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/mos_FOR_greedy"
)
DEFAULT_BATCH_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/mos_FOR_batch_full"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
app = typer.Typer(help=__doc__)


def prompt_sha256() -> str:
    """Return a stable fingerprint for the exact system and user prompts."""
    payload = f"{SYSTEM_INSTRUCTION}\n{AUDIO_LABEL}\n{ZEROSHOT_USER_TEXT_MOS}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_run_config(dataset_path: Path, data_root: Path) -> dict[str, Any]:
    """Build the immutable configuration recorded beside a run."""
    return {
        "task": "zero_shot_mos",
        "model": MODEL_NAME,
        "dataset_path": str(dataset_path),
        "data_root": str(data_root),
        "system_instruction": SYSTEM_INSTRUCTION,
        "audio_label": AUDIO_LABEL,
        "user_prompt": ZEROSHOT_USER_TEXT_MOS,
        "prompt_sha256": prompt_sha256(),
        "decoding": {
            "mode": "greedy_like",
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": None,
            "thinking_level": "LOW",
        },
        "pricing_usd_per_million_tokens": {
            "input": INPUT_USD_PER_MILLION_TOKENS,
            "output_including_thoughts": OUTPUT_USD_PER_MILLION_TOKENS,
        },
        "prompt_parity": (
            "Shared corrected non-leaking zero-shot instruction; audio is attached "
            "through Gemini's native API instead of Qwen ChatML."
        ),
    }


def validate_rows(
    rows: list[dict[str, Any]], data_root: Path
) -> list[tuple[int, dict[str, Any], Path]]:
    """Resolve and validate every audio path before any API request is sent."""
    validated: list[tuple[int, dict[str, Any], Path]] = []
    for sample_index, row in enumerate(rows):
        audios = row.get("audios")
        if not isinstance(audios, list) or not audios:
            raise ValueError(f"Sample {sample_index} has no audio reference.")
        audio_path = resolve_audio_path(str(audios[0]), data_root)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Sample {sample_index} audio missing: {audio_path}"
            )
        validated.append((sample_index, row, audio_path))
    return validated


def score_record(
    sample_index: int,
    row: dict[str, Any],
    audio_path: Path,
    response_text: str,
    usage: dict[str, int],
    cost_multiplier: float = 1.0,
    api_mode: str = "interactive",
) -> dict[str, Any]:
    """Parse and score one successful Gemini MOS response into a record."""
    predicted_mos = extract_mos(response_text)
    true_mos = float(row["mos"])
    return {
        **row,
        "sample_index": sample_index,
        "audio_path_resolved": str(audio_path),
        "status": "ok",
        "api_mode": api_mode,
        "predicted_response": response_text.strip(),
        "predicted_mos": predicted_mos,
        "mos_error": (
            abs(true_mos - predicted_mos) if predicted_mos is not None else None
        ),
        "usage": usage,
        "cost_usd": calculate_cost_usd(usage) * cost_multiplier,
    }


def write_results(
    output_path: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    include_caption_metrics: bool,
) -> None:
    """Score saved predictions through the shared MOS evaluation functions."""
    predictions = latest_predictions(predictions)
    successful = [item for item in predictions if item.get("status") == "ok"]
    parsed = [item for item in successful if item.get("predicted_mos") is not None]
    errors = [float(item["mos_error"]) for item in parsed]
    true_mos = [float(item["mos"]) for item in parsed]
    predicted_mos = [float(item["predicted_mos"]) for item in parsed]
    hyps = [str(item["predicted_response"]) for item in successful]
    refs = [str(item["response"]) for item in successful]
    can_correlate = (
        len(parsed) >= 2 and len(set(true_mos)) >= 2 and len(set(predicted_mos)) >= 2
    )
    pearson = pearsonr(true_mos, predicted_mos).statistic if can_correlate else None
    spearman = spearmanr(true_mos, predicted_mos).statistic if can_correlate else None

    metrics: dict[str, Any] = {
        "samples_requested": len(predictions),
        "samples_successful": len(successful),
        "completion_rate": len(successful) / max(len(predictions), 1),
        "mos_parsed": len(parsed),
        "mos_parse_rate": len(parsed) / max(len(successful), 1),
        "mae": sum(errors) / len(errors) if errors else None,
        "mse": sum(error**2 for error in errors) / len(errors) if errors else None,
        "pearson_r": float(pearson) if pearson is not None else None,
        "spearman_rho": float(spearman) if spearman is not None else None,
        "unique_predictions": len(set(hyps)),
        "top_prediction_frequency": (
            max(Counter(hyps).values()) / len(hyps) if hyps else None
        ),
        "cost_usd": sum(float(item.get("cost_usd", 0.0)) for item in predictions),
    }
    if include_caption_metrics and hyps:
        metrics.update(compute_caption_metrics(hyps, refs))

    output_path.write_text(
        json.dumps(
            {"metrics": metrics, "run": config, "results": predictions},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def generate_one(
    client: genai.Client,
    audio_path: Path,
    max_retries: int,
) -> tuple[str, dict[str, int]]:
    """Send one Gemini MOS request, retrying transient failures."""
    from asa.eval.gemini_api import generate_text

    contents = [
        AUDIO_LABEL,
        types.Part.from_bytes(data=audio_path.read_bytes(), mime_type="audio/wav"),
        ZEROSHOT_USER_TEXT_MOS,
    ]
    return generate_text(client, contents, max_retries)


def build_batch_request(sample_index: int, file_uri: str) -> types.InlinedRequest:
    """Build one keyed MOS Batch request using a previously uploaded WAV."""
    return types.InlinedRequest(
        contents=[
            AUDIO_LABEL,
            types.Part.from_uri(file_uri=file_uri, mime_type="audio/wav"),
            ZEROSHOT_USER_TEXT_MOS,
        ],
        metadata={"sample_index": str(sample_index)},
        config=generation_config(),
    )


@app.command()
def run(
    dataset_path: Path = typer.Option(DEFAULT_DATASET, help="MOS evaluation JSONL."),
    data_root: Path = typer.Option(Path("data"), help="Root used to resolve audio."),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Run output directory."),
    max_samples: Optional[int] = typer.Option(
        None, help="Limit samples for a smoke test."
    ),
    max_cost_usd: float = typer.Option(
        10.0, help="Stop before another request once accumulated cost reaches this."
    ),
    max_retries: int = typer.Option(4, help="Retries per transient API request."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and print configuration without API calls."
    ),
    include_caption_metrics: bool = typer.Option(
        True,
        "--caption-metrics/--no-caption-metrics",
        help="Compute BLEU, ROUGE, and BERTScore after inference.",
    ),
) -> None:
    """Evaluate Gemini greedily on the FOR MOS task, resumable and cost-capped."""
    rows = load_processed_records(dataset_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    validated = validate_rows(rows, data_root)
    config = build_run_config(dataset_path, data_root)

    predictions_path = output_dir / "predictions.jsonl"
    results_path = output_dir / "results.json"
    config_path = output_dir / "run_config.json"
    existing = load_predictions(predictions_path)
    completed_indices = {
        int(item["sample_index"]) for item in existing if item.get("status") == "ok"
    }
    accrued_cost = sum(float(item.get("cost_usd", 0.0)) for item in existing)

    logging.info("Dataset: %s (%d samples)", dataset_path, len(validated))
    logging.info("Prompt SHA-256: %s", config["prompt_sha256"])
    logging.info("Existing completed samples: %d", len(completed_indices))
    logging.info("Accrued cost: $%.4f / $%.2f ceiling", accrued_cost, max_cost_usd)
    logging.info("Output: %s", output_dir)
    if dry_run:
        logging.info("Dry run complete: no Gemini API request was sent.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config != config:
            raise ValueError(
                f"Run configuration differs from existing {config_path}; "
                "use a new output directory."
            )
    else:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    client = make_client()
    for sample_index, row, audio_path in validated:
        if sample_index in completed_indices:
            continue
        if accrued_cost >= max_cost_usd:
            logging.warning(
                "Cost ceiling reached; stopping before sample %d.", sample_index
            )
            break

        logging.info("[%d/%d] %s", sample_index + 1, len(validated), audio_path.name)
        try:
            response_text, usage = generate_one(client, audio_path, max_retries)
            record = score_record(sample_index, row, audio_path, response_text, usage)
        except DailyQuotaExhausted as exc:
            logging.warning(
                "Daily Gemini request quota exhausted at sample %d; stopping "
                "without recording an error placeholder. Resume this output "
                "directory after the quota resets. Details: %s",
                sample_index,
                exc,
            )
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
        accrued_cost += float(record["cost_usd"])
        logging.info(
            "Predicted MOS=%s | request=$%.4f | accrued=$%.4f",
            record.get("predicted_mos"),
            record["cost_usd"],
            accrued_cost,
        )
        time.sleep(4.1)

    write_results(results_path, config, existing, include_caption_metrics)
    logging.info("Saved results: %s", results_path)


@app.command()
def submit(
    dataset_path: Path = typer.Option(DEFAULT_DATASET, help="MOS evaluation JSONL."),
    data_root: Path = typer.Option(Path("data"), help="Root used to resolve audio."),
    output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR, help="Batch run dir."),
) -> None:
    """Upload all WAVs and submit exactly one all-sample MOS Batch job."""
    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "batch_job.json"
    if job_path.exists():
        raise ValueError(
            f"{job_path} already exists. Batch creation is not idempotent; "
            "inspect the existing job instead of submitting another."
        )

    rows = load_processed_records(dataset_path)
    validated = validate_rows(rows, data_root)
    config = build_run_config(dataset_path, data_root)
    config["api_mode"] = "batch"
    config["batch_cost_multiplier"] = BATCH_COST_MULTIPLIER
    config["samples_submitted"] = len(validated)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Run configuration differs from existing {config_path}.")
    else:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    client = make_client()
    uploads_path = output_dir / "uploads.jsonl"
    uploads = {int(row["sample_index"]): row for row in load_jsonl(uploads_path)}
    for sample_index, _row, audio_path in validated:
        if sample_index in uploads:
            continue
        logging.info(
            "Uploading [%d/%d] %s", sample_index + 1, len(rows), audio_path.name
        )
        uploaded = upload_with_retries(
            client, audio_path, f"FOR-{sample_index:03d}-{audio_path.name}"
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
        build_batch_request(sample_index, uploads[sample_index]["file_uri"])
        for sample_index, _row, _audio_path in validated
    ]
    logging.info("Submitting one Batch job with %d requests.", len(requests))
    job = client.batches.create(
        model=MODEL_NAME,
        src=requests,
        config=types.CreateBatchJobConfig(display_name="FOR MOS corrected prompt full"),
    )
    save_model(job_path, job)
    logging.info("Submitted Batch job %s in state %s.", job.name, job.state)


@app.command()
def status(
    output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR, help="Batch run dir."),
) -> None:
    """Fetch and save the latest state of a submitted Batch job."""
    job_path = output_dir / "batch_job.json"
    job_record = json.loads(job_path.read_text(encoding="utf-8"))
    client = make_client()
    job = client.batches.get(name=job_record["name"])
    save_model(output_dir / "batch_job_status.json", job)
    logging.info("Batch job %s: %s", job.name, job.state)
    if job.error is not None:
        logging.error("Batch job error: %s", job.error)


@app.command("import-results")
def import_results(
    dataset_path: Path = typer.Option(DEFAULT_DATASET, help="MOS evaluation JSONL."),
    data_root: Path = typer.Option(Path("data"), help="Root used to resolve audio."),
    output_dir: Path = typer.Option(DEFAULT_BATCH_OUTPUT_DIR, help="Batch run dir."),
    include_caption_metrics: bool = typer.Option(
        True,
        "--caption-metrics/--no-caption-metrics",
        help="Compute BLEU, ROUGE, and BERTScore after importing responses.",
    ),
) -> None:
    """Import a completed inline MOS Batch job into the standard result format."""
    rows = load_processed_records(dataset_path)
    validated = validate_rows(rows, data_root)
    job_record = json.loads((output_dir / "batch_job.json").read_text(encoding="utf-8"))
    client = make_client()
    job = client.batches.get(name=job_record["name"])
    save_model(output_dir / "batch_job_status.json", job)
    if not job.done:
        raise ValueError(f"Batch job is not complete: {job.state}")
    if job.dest is None or not job.dest.inlined_responses:
        raise ValueError(f"Batch job has no inline responses: {job.error}")

    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists():
        raise ValueError(
            f"{predictions_path} already exists; refusing to import responses twice."
        )

    by_index = {index: (row, audio_path) for index, row, audio_path in validated}
    for item in job.dest.inlined_responses:
        if item.metadata is None or "sample_index" not in item.metadata:
            raise ValueError("Batch response is missing sample_index metadata.")
        sample_index = int(item.metadata["sample_index"])
        row, audio_path = by_index[sample_index]
        if item.error is not None or item.response is None:
            record = {
                **row,
                "sample_index": sample_index,
                "audio_path_resolved": str(audio_path),
                "status": "error",
                "error": str(item.error),
                "api_mode": "batch",
                "cost_usd": 0.0,
            }
        else:
            usage = usage_to_dict(item.response.usage_metadata)
            record = score_record(
                sample_index,
                row,
                audio_path,
                item.response.text or "",
                usage,
                BATCH_COST_MULTIPLIER,
                "batch",
            )
        append_prediction(predictions_path, record)

    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    predictions = load_jsonl(predictions_path)
    write_results(
        output_dir / "results.json", config, predictions, include_caption_metrics
    )
    logging.info("Imported %d Batch responses.", len(predictions))


if __name__ == "__main__":
    app()
