"""Run the Appendix-B zero-shot MOS prompt on Gemini audio inputs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.stats import pearsonr, spearmanr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import load_processed_records, resolve_audio_path
from asa.prompts import ZEROSHOT_USER_TEXT_MOS
from evaluate import compute_caption_metrics, extract_mos

MODEL_NAME = "gemini-3.1-pro-preview"
SYSTEM_INSTRUCTION = "You are a helpful assistant."
AUDIO_LABEL = "Audio 1:"
TEMPERATURE = 0.0
SEED = 42
INPUT_USD_PER_MILLION_TOKENS = 2.0
OUTPUT_USD_PER_MILLION_TOKENS = 12.0

DEFAULT_DATASET = Path("data/processed/eval/test_FOR.json")
DEFAULT_OUTPUT_DIR = Path(
    "results/evaluation/gemini/gemini31_pro_preview/mos_FOR_greedy"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
app = typer.Typer(help=__doc__)


class DailyQuotaExhausted(RuntimeError):
    """Raised when Gemini's per-model daily request quota is exhausted."""


def is_daily_quota_error(exc: Exception) -> bool:
    """Return whether an API error reports the per-model daily request quota."""
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        and "generate_requests_per_model_per_day" in message
    )


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


def usage_to_dict(usage: Any) -> dict[str, int]:
    """Normalize Gemini usage metadata to the token counts used for costing."""
    if usage is None:
        return {}
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
    )
    return {field: int(getattr(usage, field, 0) or 0) for field in fields}


def calculate_cost_usd(usage: dict[str, int]) -> float:
    """Calculate standard-tier cost from Gemini usage metadata."""
    input_tokens = usage.get("prompt_token_count", 0)
    output_tokens = usage.get("candidates_token_count", 0) + usage.get(
        "thoughts_token_count", 0
    )
    return (
        input_tokens * INPUT_USD_PER_MILLION_TOKENS
        + output_tokens * OUTPUT_USD_PER_MILLION_TOKENS
    ) / 1_000_000


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load an incremental prediction JSONL if it exists."""
    if not path.exists():
        return []
    return load_processed_records(path)


def latest_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest attempt for each sample index."""
    latest: dict[int, dict[str, Any]] = {}
    for prediction in predictions:
        latest[int(prediction["sample_index"])] = prediction
    return [latest[index] for index in sorted(latest)]


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    """Append one completed request immediately for interruption-safe resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


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


def write_results(
    output_path: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    include_caption_metrics: bool,
) -> None:
    """Score saved predictions through the existing MOS evaluation functions."""
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
    """Send one Gemini request, retrying transient failures."""
    audio_bytes = audio_path.read_bytes()
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    AUDIO_LABEL,
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    ZEROSHOT_USER_TEXT_MOS,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                ),
            )
            return (response.text or "").strip(), usage_to_dict(response.usage_metadata)
        except Exception as exc:
            if is_daily_quota_error(exc):
                raise DailyQuotaExhausted(str(exc)) from exc
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = min(2**attempt, 60)
            logging.warning(
                "Request failed (%s); retrying in %ds.",
                type(exc).__name__,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Gemini request failed after retries: {last_error}"
    ) from last_error


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
    """Evaluate Gemini 3.1 Pro greedily on the FOR MOS task."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")

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

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment or root .env.")

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

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )
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
            predicted_mos = extract_mos(response_text)
            true_mos = float(row["mos"])
            mos_error = (
                abs(true_mos - predicted_mos) if predicted_mos is not None else None
            )
            record = {
                **row,
                "sample_index": sample_index,
                "audio_path_resolved": str(audio_path),
                "status": "ok",
                "predicted_response": response_text,
                "predicted_mos": predicted_mos,
                "mos_error": mos_error,
                "usage": usage,
                "cost_usd": calculate_cost_usd(usage),
            }
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


if __name__ == "__main__":
    app()
