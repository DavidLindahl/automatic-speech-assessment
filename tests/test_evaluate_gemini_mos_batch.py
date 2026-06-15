"""Tests for the Gemini MOS Batch workflow; no test makes an API request."""

from pathlib import Path
from types import SimpleNamespace

import evaluate_gemini_mos_batch as batch
from asa.prompts import ZEROSHOT_USER_TEXT_MOS


def test_batch_request_matches_interactive_configuration() -> None:
    request = batch.build_batch_request(7, "https://example.test/audio.wav")

    assert request.metadata == {"sample_index": "7"}
    assert request.contents[0] == "Audio 1:"
    assert request.contents[1].file_data.file_uri.endswith("audio.wav")
    assert request.contents[2] == ZEROSHOT_USER_TEXT_MOS
    assert request.config.system_instruction == "You are a helpful assistant."
    assert request.config.temperature == 0.0
    assert request.config.seed == 42
    assert request.config.max_output_tokens is None
    assert request.config.thinking_config.thinking_level.value == "LOW"


def test_batch_cost_is_half_interactive_cost() -> None:
    usage = {
        "prompt_token_count": 1_000,
        "candidates_token_count": 100,
        "thoughts_token_count": 400,
    }

    assert batch.calculate_cost_usd(usage) * batch.BATCH_COST_MULTIPLIER == 0.004


def test_upload_retries_transient_failure(monkeypatch) -> None:
    class FakeFiles:
        calls = 0

        def upload(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary")
            return "uploaded"

    files = FakeFiles()
    monkeypatch.setattr(batch.time, "sleep", lambda _seconds: None)

    assert (
        batch.upload_with_retries(
            SimpleNamespace(files=files),
            Path("audio.wav"),
            "audio",
        )
        == "uploaded"
    )
    assert files.calls == 2
