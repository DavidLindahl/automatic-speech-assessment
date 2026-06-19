"""Submit, inspect, and import a full FOR MOS evaluation through Gemini Batch."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records
from evaluate import extract_mos
from evaluate_gemini_mos import (
    AUDIO_LABEL,
    DEFAULT_DATASET,
    MODEL_NAME,
    SEED,
    SYSTEM_INSTRUCTION,
    TEMPERATURE,
    ZEROSHOT_USER_TEXT_MOS,
    append_prediction,
    build_run_config,
    calculate_cost_usd,
    usage_to_dict,
    validate_rows,
    write_results,
)

BATCH_COST_MULTIPLIER = 0.5
DEFAULT_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/mos_FOR_batch_full"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
app = typer.Typer(help=__doc__)


def make_client() -> genai.Client:
    """Create an authenticated Gemini Developer API client."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment or root .env.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records when the file exists."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def batch_config() -> types.GenerateContentConfig:
    """Return the same generation configuration used by interactive inference."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        seed=SEED,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


def build_batch_request(
    sample_index: int,
    file_uri: str,
) -> types.InlinedRequest:
    """Build one keyed Batch request using a previously uploaded WAV."""
    return types.InlinedRequest(
        contents=[
            AUDIO_LABEL,
            types.Part.from_uri(file_uri=file_uri, mime_type="audio/wav"),
            ZEROSHOT_USER_TEXT_MOS,
        ],
        metadata={"sample_index": str(sample_index)},
        config=batch_config(),
    )


def save_model(path: Path, model: Any) -> None:
    """Serialize a Google Gen AI SDK response beside the run."""
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def upload_with_retries(
    client: genai.Client,
    audio_path: Path,
    display_name: str,
    max_retries: int = 5,
) -> types.File:
    """Upload one WAV, retrying transient service failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.files.upload(
                file=audio_path,
                config=types.UploadFileConfig(
                    mime_type="audio/wav",
                    display_name=display_name,
                ),
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = min(2**attempt, 30)
            logging.warning(
                "Upload failed (%s); retrying in %ds.",
                type(exc).__name__,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"File upload failed after retries: {last_error}"
    ) from last_error


@app.command()
def submit(
    dataset_path: Path = typer.Option(DEFAULT_DATASET, help="MOS evaluation JSONL."),
    data_root: Path = typer.Option(Path("data"), help="Root used to resolve audio."),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Batch run directory."),
) -> None:
    """Upload all WAVs and submit exactly one all-sample Batch job."""
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
            client,
            audio_path,
            f"FOR-{sample_index:03d}-{audio_path.name}",
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
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Batch run directory."),
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
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Batch run directory."),
    include_caption_metrics: bool = typer.Option(
        True,
        "--caption-metrics/--no-caption-metrics",
        help="Compute BLEU, ROUGE, and BERTScore after importing responses.",
    ),
) -> None:
    """Import a completed inline Batch job into the standard result format."""
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
            response_text = item.response.text or ""
            predicted_mos = extract_mos(response_text)
            true_mos = float(row["mos"])
            usage = usage_to_dict(item.response.usage_metadata)
            record = {
                **row,
                "sample_index": sample_index,
                "audio_path_resolved": str(audio_path),
                "status": "ok",
                "api_mode": "batch",
                "predicted_response": response_text.strip(),
                "predicted_mos": predicted_mos,
                "mos_error": (
                    abs(true_mos - predicted_mos) if predicted_mos is not None else None
                ),
                "usage": usage,
                "cost_usd": calculate_cost_usd(usage) * BATCH_COST_MULTIPLIER,
            }
        append_prediction(predictions_path, record)

    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    predictions = load_jsonl(predictions_path)
    write_results(
        output_dir / "results.json",
        config,
        predictions,
        include_caption_metrics,
    )
    logging.info("Imported %d Batch responses.", len(predictions))


if __name__ == "__main__":
    app()
