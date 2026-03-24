"""Smoke test for inference on a trained checkpoint.

Verifies that the model loads from ``results/sft`` and generates non-empty
text for a handful of NISQA_TEST_FOR audio files.

Marked ``@pytest.mark.slow`` so that fast CI runs can skip it
(``pytest -m "not slow"``).  Works on both GPU (HPC) and CPU (home).
"""

import random
from pathlib import Path

import pytest

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
