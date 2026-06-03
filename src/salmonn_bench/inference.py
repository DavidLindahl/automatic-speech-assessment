"""Zero-shot inference wrapper for SALMONN quality evaluation."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

from salmonn_bench.config import BenchmarkConfig


def _ensure_salmonn_on_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    salmonn_root = project_root / "third_party" / "salmonn"
    path_value = str(salmonn_root)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)


class SalmonnZeroShotRunner:
    """Model runner for zero-shot SALMONN inference."""

    def __init__(self, config: BenchmarkConfig) -> None:
        _ensure_salmonn_on_path()

        from models.salmonn import SALMONN

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")

        self.device = torch.device(config.device)
        self.generate_cfg = config.generate
        self.prompt_template = str(
            config.model.get("prompt_template", "USER: {}\\nASSISTANT:")
        )
        self.use_amp = config.use_amp
        self.silence_samples_ab = config.silence_samples_ab

        self.model = SALMONN.from_config(config.model)
        self.model.to(self.device)
        self.model.eval()

        whisper_path = str(config.model["whisper_path"])
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.sample_rate = int(self.wav_processor.sampling_rate)

    def _format_prompt(self, task_prompt: str) -> str:
        prompt = task_prompt.strip()
        if "<SpeechHere>" not in prompt:
            prompt = f"<Speech><SpeechHere></Speech> {prompt}"
        return self.prompt_template.format(prompt)

    def _load_audio_mono(self, audio_path: Path) -> np.ndarray:
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = np.asarray(audio, dtype=np.float32)
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio.astype(np.float32)

    def _finalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.sample_rate:
            pad = np.zeros(self.sample_rate - len(audio), dtype=np.float32)
            audio = np.concatenate((audio, pad), axis=0)
        max_len = self.sample_rate * 30
        return audio[:max_len]

    def _prepare_sample(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
        spectrogram = self.wav_processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )["input_features"]

        sample = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        return {k: v.to(self.device) for k, v in sample.items()}

    def infer_mos(self, audio_path: Path, task_prompt: str) -> str:
        """Run zero-shot MOS inference for a single audio sample."""
        audio = self._load_audio_mono(audio_path)
        audio = self._finalize_audio(audio)
        sample = self._prepare_sample(audio)
        prompt = [self._format_prompt(task_prompt)]

        with torch.no_grad():
            if self.use_amp and self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = self.model.generate(
                        sample, self.generate_cfg, prompts=prompt
                    )
            else:
                output = self.model.generate(sample, self.generate_cfg, prompts=prompt)
        return output[0].strip()

    def infer_ab(self, audio_a_path: Path, audio_b_path: Path, task_prompt: str) -> str:
        """Run zero-shot A/B inference by concatenating two audio clips."""
        audio_a = self._load_audio_mono(audio_a_path)
        audio_b = self._load_audio_mono(audio_b_path)
        silence = np.zeros(self.silence_samples_ab, dtype=np.float32)
        audio = np.concatenate((audio_a, silence, audio_b), axis=0)
        audio = self._finalize_audio(audio)
        sample = self._prepare_sample(audio)
        prompt = [self._format_prompt(task_prompt)]

        with torch.no_grad():
            if self.use_amp and self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = self.model.generate(
                        sample, self.generate_cfg, prompts=prompt
                    )
            else:
                output = self.model.generate(sample, self.generate_cfg, prompts=prompt)
        return output[0].strip()


def dump_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write prediction records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
