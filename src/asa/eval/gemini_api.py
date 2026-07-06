"""Gemini API driver shared by the MOS and temporal zero-shot baselines.

Both Gemini evals (``evaluate_gemini_mos.py`` and
``evaluate_gemini_temporal.py``) send audio + a fixed instruction to the same
model under the same decoding config, then score the reply. Everything that is
not the prompt string or the task-specific scorer is here: client construction,
per-model daily-quota detection, token→USD costing, interruption-safe JSONL
resume, and the file-upload / Batch helpers. The scripts stay thin: they supply
the prompt and the scorer, this module runs the plumbing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Model + decoding contract, identical across both Gemini evals so the baseline
# rows stay comparable. Costing is the published standard-tier price per the
# model card; Batch requests are billed at half (see BATCH_COST_MULTIPLIER).
MODEL_NAME = "gemini-3.1-pro-preview"
SYSTEM_INSTRUCTION = "You are a helpful assistant."
AUDIO_LABEL = "Audio 1:"
TEMPERATURE = 0.0
SEED = 42
INPUT_USD_PER_MILLION_TOKENS = 2.0
OUTPUT_USD_PER_MILLION_TOKENS = 12.0
BATCH_COST_MULTIPLIER = 0.5


class DailyQuotaExhausted(RuntimeError):
    """Raised when Gemini's per-model daily request quota is exhausted."""


def is_daily_quota_error(exc: Exception) -> bool:
    """Return whether an API error reports the per-model daily request quota."""
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        and "generate_requests_per_model_per_day" in message
    )


def generation_config() -> types.GenerateContentConfig:
    """Return the shared temperature-zero Gemini generation configuration.

    The single decoding contract used by interactive and Batch requests for both
    tasks: greedy-like (temperature 0), fixed seed, low thinking budget.
    """
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        seed=SEED,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


def make_client() -> genai.Client:
    """Create an authenticated Gemini Developer API client from the root .env."""
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from the environment or root .env.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records when the file exists, else an empty list."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load an incremental prediction JSONL if it exists."""
    if not path.exists():
        return []
    return load_jsonl(path)


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


def save_model(path: Path, model: Any) -> None:
    """Serialize a Google Gen AI SDK response beside the run."""
    path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def generate_text(
    client: genai.Client,
    contents: list[Any],
    max_retries: int,
) -> tuple[str, dict[str, int]]:
    """Send one generate_content request, retrying transient failures.

    ``contents`` is the fully-built request payload (audio label, audio Part,
    and the task prompt), so this stays task-agnostic. A per-model daily quota
    error is re-raised as :class:`DailyQuotaExhausted` so callers can stop and
    resume later instead of burning retries.

    Returns:
        Tuple ``(response_text, usage_dict)``.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=generation_config(),
            )
            return (response.text or "").strip(), usage_to_dict(
                response.usage_metadata
            )
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


def upload_with_retries(
    client: genai.Client,
    audio_path: Path,
    display_name: str,
    max_retries: int = 5,
) -> types.File:
    """Upload one WAV to the Files API, retrying transient service failures."""
    last_error: Optional[Exception] = None
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
