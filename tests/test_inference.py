"""Smoke test for inference on a trained checkpoint.

Verifies that the model loads from ``results/sft`` and generates non-empty
text for a handful of NISQA_TEST_FOR audio files.

Marked ``@pytest.mark.slow`` so that fast CI runs can skip it
(``pytest -m "not slow"``).  Works on both GPU (HPC) and CPU (home).
"""

import random
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from asa.inference import load_model, run_inference

# ---------------------------------------------------------------------------
# Paths (absolute, resolved from this file — no os.chdir needed)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "results" / "sft"
_AUDIO_ROOT = _PROJECT_ROOT / "data" / "raw" / "NISQA_Corpus" / "NISQA_TEST_FOR"


def _collect_audio_files(max_files: int = 3) -> list[Path]:
    """Return up to *max_files* randomly sampled .wav paths from the test corpus."""
    if not _AUDIO_ROOT.is_dir():
        return []
    files: list[Path] = []
    for subdir in ("deg", "ref"):
        d = _AUDIO_ROOT / subdir
        if d.is_dir():
            files.extend(d.glob("*.wav"))
    if len(files) <= max_files:
        return files
    return random.sample(files, max_files)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@dataclass
class _FakeFeatureExtractor:
    """Minimal feature extractor stub for inference unit tests."""

    sampling_rate: int = 16_000


class _FakeProcessor:
    """Minimal processor stub that records decoding options."""

    def __init__(self) -> None:
        self.feature_extractor = _FakeFeatureExtractor()
        self.skip_special_tokens_seen: bool | None = None

    def __call__(self, text, audios, sampling_rate, return_tensors, padding):
        batch_size = len(text)
        return {
            "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }

    def batch_decode(self, response_ids, skip_special_tokens=True):
        self.skip_special_tokens_seen = skip_special_tokens
        response = "between  and ."
        if not skip_special_tokens:
            response = "between <|1.00|> and <|2.00|><|im_end|>"
        return [response for _ in range(response_ids.shape[0])]


class _FakeModel:
    """Minimal model stub for generation-only inference tests."""

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, **batch):
        input_ids = batch["input_ids"]
        suffix = torch.full((input_ids.shape[0], 3), 9, dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


def test_run_inference_can_preserve_special_tokens(monkeypatch) -> None:
    """Temporal evaluation can opt into timestamp-preserving decoding."""
    monkeypatch.setattr("asa.inference.load_audio", lambda *args, **kwargs: [0.0])
    processor = _FakeProcessor()

    outputs = run_inference(
        model=_FakeModel(),
        processor=processor,
        audio_paths=["dummy.wav"],
        skip_special_tokens=False,
    )

    assert processor.skip_special_tokens_seen is False
    assert outputs == ["between <|1.00|> and <|2.00|><|im_end|>"]


def test_run_inference_skips_special_tokens_by_default(monkeypatch) -> None:
    """Existing non-temporal inference keeps its prior decoding behavior."""
    monkeypatch.setattr("asa.inference.load_audio", lambda *args, **kwargs: [0.0])
    processor = _FakeProcessor()

    outputs = run_inference(
        model=_FakeModel(),
        processor=processor,
        audio_paths=["dummy.wav"],
    )

    assert processor.skip_special_tokens_seen is True
    assert outputs == ["between  and ."]


@pytest.mark.slow
def test_inference_smoke():
    """Load checkpoint → run inference → check we get plausible text back."""
    if not _MODEL_DIR.exists():
        pytest.skip(f"Model directory not found: {_MODEL_DIR}")

    audio_files = _collect_audio_files()
    if not audio_files:
        pytest.skip(f"No .wav files found under: {_AUDIO_ROOT}")

    processor, model, device = load_model(_MODEL_DIR)
    outputs = run_inference(
        model, processor, audio_files, device=device, max_new_tokens=256
    )

    assert len(outputs) == len(audio_files)
    for i, text in enumerate(outputs):
        assert (
            isinstance(text, str) and text.strip()
        ), f"Empty output for {audio_files[i].name}"
        # Print for manual inspection when running with -s
        print(f"  [{audio_files[i].name}] → {text.strip()}")
