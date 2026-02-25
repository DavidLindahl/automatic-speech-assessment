"""
data.py — SFT data pipeline for Qwen2-Audio fine-tuning.

Contains:
  - SFTDataset: PyTorch Dataset that loads JSONL + WAV files on-the-fly
  - Qwen2AudioCollator: batches samples and calls the Qwen2-Audio processor
"""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import librosa

TARGET_SR = 16_000  # Qwen2-Audio expects 16 kHz mono

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>Please describe and evaluate the synthetic speech."


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read a WAV file and return mono float32 array at target_sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Convert to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to target_sr if needed
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Loads train_nisqa_llama_10k.json (JSONL) and serves samples for SFT.

    Each sample returned by __getitem__:
        {
            "prompt":        str   — e.g. "<|audio_bos|><|AUDIO|><|audio_eos|>Evaluate the quality of this speech."
            "response":      str   — the target text the model should produce
            "audio":         np.ndarray (float32, mono, 16 kHz)
            "sampling_rate": int
        }
    """

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.samples = self._load_jsonl(Path(json_path))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"SFTDataset: loaded {len(self.samples)} samples from {json_path}")

    # ── private ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """Parse line-delimited JSON (one JSON object per '{...}' block)."""
        text = path.read_text(encoding="utf-8")

        items = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            # Skip whitespace
            while idx < len(text) and text[idx] in " \t\n\r":
                idx += 1
            if idx >= len(text):
                break
            obj, end_idx = decoder.raw_decode(text, idx)
            items.append(obj)
            idx = end_idx
        return items

    def _resolve_audio_path(self, raw_path: str) -> Path:
        """Map the JSON path (e.g. '/data/raw/NISQA_Corpus/...') to a local path."""
        if "NISQA_Corpus" in raw_path:
            rel = raw_path[raw_path.find("NISQA_Corpus"):]
            return self.data_root / "raw" / rel
        # Fallback: treat as relative to data_root
        return self.data_root / raw_path.lstrip("/")

    # ── public ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Resolve and load audio
        audio_path = self._resolve_audio_path(item["audios"][0])
        audio_array = load_audio(str(audio_path))

        return {
            "prompt": PROMPT_TEMPLATE,
            "response": item["response"],
            "audio": audio_array,
            "sampling_rate": TARGET_SR,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class Qwen2AudioCollator:
    """
    Collates a list of SFTDataset samples into a batch for the trainer.

    1. Concatenates prompt + response into a single text string per sample
    2. Calls processor(text=..., audios=...) to get input_ids + audio features
    3. Creates labels: prompt tokens masked with -100, response tokens kept
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate prompts and full texts, to 
        # distinguish between prompt and response tokens
        prompts = [f["prompt"] for f in features]
        full_texts = [f["prompt"] + f["response"] for f in features]
        audios = [f["audio"] for f in features]

        # Tokenize full sequences (prompt + response)
        batch = self.processor(
            text=full_texts,
            audio=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize prompts alone to find where the response starts
        prompt_batch = self.processor(
            text=prompts,
            audio=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: mask prompt tokens with -100
        labels = batch["input_ids"].clone()
        for i in range(len(features)):
            prompt_len = prompt_batch["input_ids"][i].ne(self.processor.tokenizer.pad_token_id).sum()
            # Mask prompt tokens with -100 (ignore loss on prompt tokens)
            labels[i, :prompt_len] = -100
        # Also mask padding
        labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch
