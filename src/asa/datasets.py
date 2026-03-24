"""
datasets.py - PyTorch datasets for ALLD training.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from asa.audio import load_audio, TARGET_SR
from asa.prompts import build_expert_prompt_MOS, build_expert_prompt_ab
from asa.processed_data import (
    DPO_METADATA_FIELDS,
    DPO_METADATA_FIELDS_AB,
    load_processed_records,
    resolve_audio_path,
)

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>Please describe and evaluate the synthetic speech."
PROMPT_TEMPLATE_AB = (
    "Please perform A/B preference test between<audio>and<audio>, including a tie."
)
AUDIO_PLACEHOLDER = "<audio>"
AUDIO_SPECIAL = "<|audio_bos|><|AUDIO|><|audio_eos|>"


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

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"SFTDataset: loaded {len(self.samples)} samples from {json_path}")

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
        """Map stored audio paths to a local path."""
        return resolve_audio_path(raw_path, self.data_root)

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


class SFTDatasetAB(SFTDataset):
    """
    A/B preference variant of SFTDataset.

    Each JSONL row has *two* audio paths in ``audios`` and a query with two
    ``<audio>`` placeholders.  Replaces each ``<audio>`` with the Qwen2-Audio
    special-token sequence and returns both waveforms.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        prompt = item["query"].replace(AUDIO_PLACEHOLDER, AUDIO_SPECIAL)

        audio_a = load_audio(str(self._resolve_audio_path(item["audios"][0])))
        audio_b = load_audio(str(self._resolve_audio_path(item["audios"][1])))

        return {
            "prompt": prompt,
            "response": item["response"],
            "audio_a": audio_a,
            "audio_b": audio_b,
            "sampling_rate": TARGET_SR,
        }


class DPODataset(Dataset):
    """
    Loads DPO JSONL format and extracts both audio (for the policy model)
    and metadata scores (for the reference text model).
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

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"DPODataset: loaded {len(self.samples)} samples from {json_path}")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        return load_processed_records(path)

    def _resolve_audio_path(self, raw_path: str) -> Path:
        return resolve_audio_path(raw_path, self.data_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_path = self._resolve_audio_path(item["audios"][0])
        audio_array = load_audio(str(audio_path))

        metadata = {field: float(item[field]) for field in DPO_METADATA_FIELDS}
        meta_prompt = build_expert_prompt_MOS(**metadata)

        return {
            "audio_prompt": PROMPT_TEMPLATE,  # For Policy Model
            "meta_prompt": meta_prompt,  # For Reference Model
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "audio": audio_array,
            "sampling_rate": TARGET_SR,
        }


class DPODatasetAB(DPODataset):
    """
    A/B preference variant of DPODataset.
    Each dual-audio sample is used for the policy model, while the reference
    text model receives the A/B metadata to evaluate the chosen/rejected texts.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # dual audios
        audio_a = load_audio(str(self._resolve_audio_path(item["audios"][0])))
        audio_b = load_audio(str(self._resolve_audio_path(item["audios"][1])))

        # dual metadata
        metadata = {field: float(item[field]) for field in DPO_METADATA_FIELDS_AB}
        meta_prompt = build_expert_prompt_ab(**metadata)

        # Qwen2-Audio placeholder replacement
        prompt = item["query"].replace(AUDIO_PLACEHOLDER, AUDIO_SPECIAL)

        return {
            "audio_prompt": prompt,  # For Policy Model
            "meta_prompt": meta_prompt,  # For Reference Model
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "audio_a": audio_a,
            "audio_b": audio_b,
            "sampling_rate": TARGET_SR,
        }
