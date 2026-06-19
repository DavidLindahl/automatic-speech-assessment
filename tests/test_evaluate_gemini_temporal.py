"""Tests for Gemini temporal evaluation; no test makes an API request."""

import json
from pathlib import Path
from types import SimpleNamespace

import evaluate_gemini_temporal as temporal
from asa.prompts import ZEROSHOT_USER_TEXT_TEMPORAL
from evaluate_temporal import Interval


def test_configuration_uses_exact_temporal_prompt() -> None:
    config = temporal.build_run_config(Path("temporal.json"), Path("data"))
    generation = temporal.generation_config()

    assert config["user_prompt"] == ZEROSHOT_USER_TEXT_TEMPORAL
    assert config["interval_parsing"]["allow_plain_numbers"] is False
    assert generation.temperature == 0.0
    assert generation.seed == 42
    assert generation.max_output_tokens is None
    assert generation.thinking_config.thinking_level.value == "LOW"


def test_batch_request_uses_exact_temporal_prompt() -> None:
    request = temporal.build_batch_request(17, "https://example.test/audio.wav")

    assert request.metadata == {"sample_index": "17"}
    assert request.contents[0] == "Audio 1:"
    assert request.contents[2] == ZEROSHOT_USER_TEXT_TEMPORAL


def test_score_record_requires_explicit_range(tmp_path: Path) -> None:
    row = {"duration_seconds": 8.0, "mos": 2.0, "response": "gold"}
    record = temporal.score_record(
        3,
        row,
        tmp_path / "audio.wav",
        Interval(2.0, 4.0),
        "The MOS is 3 out of 5 for this 8 second clip.",
        {},
    )

    assert record["pred_start"] is None
    assert record["pred_interval_source"] == "none"
    assert record["tiou"] == 0.0


def test_score_record_scores_explicit_range(tmp_path: Path) -> None:
    row = {"duration_seconds": 8.0, "mos": 2.0, "response": "gold"}
    record = temporal.score_record(
        3,
        row,
        tmp_path / "audio.wav",
        Interval(2.0, 4.0),
        "The degradation occurs between 3 and 5 seconds. Overall MOS score is 3.",
        {},
    )

    assert record["pred_start"] == 3.0
    assert record["pred_end"] == 5.0
    assert record["pred_interval_source"] == "range"
    assert record["tiou"] == 1 / 3
    assert record["offset_err"] == 1.0
    assert record["start_offset_err"] == 1.0
    assert record["end_offset_err"] == 1.0


def test_write_results_counts_unparsed_as_zero_tiou(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    predictions = [
        {
            "sample_index": 0,
            "status": "ok",
            "duration_seconds": 10.0,
            "gt_start": 2.0,
            "gt_end": 4.0,
            "pred_start": 2.0,
            "pred_end": 4.0,
            "pred_interval_source": "range",
            "tiou": 1.0,
            "offset_err": 0.0,
            "start_offset_err": 0.0,
            "end_offset_err": 0.0,
            "start_abs_err": 0.0,
            "end_abs_err": 0.0,
            "predicted_mos": 2.0,
            "mos_error": 0.0,
            "predicted_response": "one",
            "cost_usd": 0.01,
        },
        {
            "sample_index": 1,
            "status": "ok",
            "duration_seconds": 10.0,
            "gt_start": 6.0,
            "gt_end": 8.0,
            "pred_start": None,
            "pred_end": None,
            "pred_interval_source": "none",
            "tiou": 0.0,
            "start_abs_err": None,
            "end_abs_err": None,
            "predicted_mos": None,
            "mos_error": None,
            "predicted_response": "two",
            "cost_usd": 0.02,
        },
    ]

    temporal.write_results(output, {}, predictions)
    metrics = json.loads(output.read_text())["metrics"]

    assert metrics["interval_parse_rate"] == 0.5
    assert metrics["mean_tiou"] == 0.5
    assert metrics["median_tiou"] == 0.5
    assert metrics["mos_parse_rate"] == 0.5
    assert metrics["cost_usd"] == 0.03


def test_write_results_reports_signed_expected_offset(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    predictions = [
        {
            "sample_index": 0,
            "status": "ok",
            "duration_seconds": 10.0,
            "gt_start": 2.0,
            "gt_end": 4.0,
            "pred_start": 3.0,
            "pred_end": 5.0,
            "pred_interval_source": "range",
            "tiou": 1 / 3,
            "offset_err": 1.0,
            "start_offset_err": 1.0,
            "end_offset_err": 1.0,
            "start_abs_err": 1.0,
            "end_abs_err": 1.0,
            "predicted_mos": None,
            "mos_error": None,
            "predicted_response": "late",
            "cost_usd": 0.0,
        },
        {
            "sample_index": 1,
            "status": "ok",
            "duration_seconds": 10.0,
            "gt_start": 6.0,
            "gt_end": 8.0,
            "pred_start": 5.0,
            "pred_end": 7.0,
            "pred_interval_source": "range",
            "tiou": 1 / 3,
            "offset_err": -1.0,
            "start_offset_err": -1.0,
            "end_offset_err": -1.0,
            "start_abs_err": 1.0,
            "end_abs_err": 1.0,
            "predicted_mos": None,
            "mos_error": None,
            "predicted_response": "early",
            "cost_usd": 0.0,
        },
    ]

    temporal.write_results(output, {}, predictions)
    metrics = json.loads(output.read_text())["metrics"]

    assert metrics["expected_offset_error"] == 0.0
    assert metrics["mean_start_offset_err"] == 0.0
    assert metrics["mean_end_offset_err"] == 0.0
    assert metrics["mean_start_abs_err"] == 1.0
    assert metrics["mean_end_abs_err"] == 1.0


def test_generate_one_sends_temporal_prompt(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF-test")

    class FakeModels:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(text="between 1 and 2 seconds", usage_metadata=None)

    models = FakeModels()
    text, _usage = temporal.generate_one(SimpleNamespace(models=models), audio, 0)

    assert text == "between 1 and 2 seconds"
    assert models.kwargs["contents"][2] == ZEROSHOT_USER_TEXT_TEMPORAL
