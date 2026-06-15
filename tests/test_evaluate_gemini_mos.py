"""Tests for the Gemini MOS runner; no test makes an API request."""

import json
from pathlib import Path
from types import SimpleNamespace

import evaluate_gemini_mos as gemini_mos
import pytest
from asa.prompts import ZEROSHOT_USER_TEXT_MOS


def test_run_config_uses_exact_zero_shot_prompt() -> None:
    config = gemini_mos.build_run_config(Path("test_FOR.json"), Path("data"))

    assert config["system_instruction"] == "You are a helpful assistant."
    assert config["audio_label"] == "Audio 1:"
    assert config["user_prompt"] == ZEROSHOT_USER_TEXT_MOS
    assert config["decoding"]["temperature"] == 0.0
    assert config["decoding"]["max_output_tokens"] is None
    assert config["decoding"]["thinking_level"] == "LOW"


def test_calculate_cost_includes_thinking_tokens() -> None:
    usage = {
        "prompt_token_count": 1_000,
        "candidates_token_count": 100,
        "thoughts_token_count": 400,
    }

    assert gemini_mos.calculate_cost_usd(usage) == 0.008


def test_append_and_load_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    gemini_mos.append_prediction(path, {"sample_index": 0, "status": "ok"})
    gemini_mos.append_prediction(path, {"sample_index": 1, "status": "ok"})

    assert gemini_mos.load_predictions(path) == [
        {"sample_index": 0, "status": "ok"},
        {"sample_index": 1, "status": "ok"},
    ]


def test_latest_predictions_keeps_retry_result() -> None:
    predictions = [
        {"sample_index": 0, "status": "error"},
        {"sample_index": 1, "status": "ok"},
        {"sample_index": 0, "status": "ok"},
    ]

    assert gemini_mos.latest_predictions(predictions) == [
        {"sample_index": 0, "status": "ok"},
        {"sample_index": 1, "status": "ok"},
    ]


def test_generate_one_sends_appendix_b_prompt(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF-test")

    class FakeModels:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                text="The overall MOS score is 3.0.",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    thoughts_token_count=2,
                    total_token_count=17,
                ),
            )

    models = FakeModels()
    client = SimpleNamespace(models=models)
    text, usage = gemini_mos.generate_one(client, audio_path, max_retries=0)

    assert text == "The overall MOS score is 3.0."
    assert usage["total_token_count"] == 17
    assert models.kwargs["model"] == gemini_mos.MODEL_NAME
    assert models.kwargs["contents"][0] == "Audio 1:"
    assert models.kwargs["contents"][2] == ZEROSHOT_USER_TEXT_MOS
    assert models.kwargs["config"].system_instruction == "You are a helpful assistant."
    assert models.kwargs["config"].temperature == 0.0
    assert models.kwargs["config"].max_output_tokens is None


def test_generate_one_stops_immediately_on_daily_quota(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF-test")

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: "
                "generativelanguage.googleapis.com/"
                "generate_requests_per_model_per_day"
            )

    models = FakeModels()
    client = SimpleNamespace(models=models)

    with pytest.raises(gemini_mos.DailyQuotaExhausted):
        gemini_mos.generate_one(client, audio_path, max_retries=8)

    assert models.calls == 1


def test_write_results_scores_mos_without_caption_download(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    predictions = [
        {
            "sample_index": 0,
            "status": "ok",
            "response": "Reference one.",
            "predicted_response": "The overall MOS score is 4.0.",
            "mos": 4.5,
            "predicted_mos": 4.0,
            "mos_error": 0.5,
            "cost_usd": 0.01,
        },
        {
            "sample_index": 1,
            "status": "ok",
            "response": "Reference two.",
            "predicted_response": "No score stated.",
            "mos": 3.0,
            "predicted_mos": None,
            "mos_error": None,
            "cost_usd": 0.02,
        },
    ]

    gemini_mos.write_results(output, {}, predictions, include_caption_metrics=False)
    metrics = json.loads(output.read_text())["metrics"]

    assert metrics["completion_rate"] == 1.0
    assert metrics["mos_parse_rate"] == 0.5
    assert metrics["mae"] == 0.5
    assert metrics["mse"] == 0.25
    assert metrics["pearson_r"] is None
    assert metrics["spearman_rho"] is None
    assert metrics["cost_usd"] == 0.03
