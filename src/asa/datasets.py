"""PyTorch Datasets for SFT and DPO training on Qwen2-Audio."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from asa.audio import AUDIO_PLACEHOLDER, AUDIO_SPECIAL, TARGET_SR, load_audio
from asa.processed_data import (
    DPO_METADATA_FIELDS,
    load_processed_records,
    resolve_audio_path,
)
from asa.prompts import PROMPT_TEMPLATE, build_expert_prompt_MOS


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
        use_query_prompt: bool = False,
    ):
        self.data_root = Path(data_root)
        self.use_query_prompt = use_query_prompt
        self.samples = self._load_jsonl(Path(json_path))
        self.samples = [item for item in self.samples if self._is_valid(item)]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"SFTDataset: loaded {len(self.samples)} samples from {json_path}")

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """Parse line-delimited JSON (one JSON object per line)."""
        items: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def _is_valid(self, item):
        if (
            not item.get("audios")
            or not isinstance(item["audios"], list)
            or len(item["audios"]) == 0
        ):
            return False
        raw_path = item["audios"][0]
        return self._resolve_audio_path(raw_path).exists()

    def _resolve_audio_path(self, raw_path: str) -> Path:
        return resolve_audio_path(raw_path, self.data_root)

    @staticmethod
    def _query_to_prompt(query: Any) -> str:
        """Convert a record query string into a Qwen2-Audio prompt."""
        if not isinstance(query, str):
            return PROMPT_TEMPLATE

        text = " ".join(query.strip().split())
        if not text:
            return PROMPT_TEMPLATE
        if AUDIO_PLACEHOLDER in text:
            return text.replace(AUDIO_PLACEHOLDER, AUDIO_SPECIAL)
        if "<|AUDIO|>" in text:
            return text
        return f"{AUDIO_SPECIAL}{text}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        audio_path = self._resolve_audio_path(item["audios"][0])
        audio_array = load_audio(str(audio_path))
        prompt = PROMPT_TEMPLATE
        if self.use_query_prompt:
            prompt = self._query_to_prompt(item.get("query"))

        return {
            "prompt": prompt,
            "response": item["response"],
            "audio": audio_array,
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
