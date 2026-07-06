"""PyTorch Datasets for SFT and DPO training on Qwen2-Audio."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from asa.audio import AUDIO_PLACEHOLDER, AUDIO_SPECIAL, TARGET_SR, load_audio
from asa.processed_data import (
    DPO_METADATA_FIELDS,
    DPO_METADATA_FIELDS_DIS,
    load_processed_records,
    resolve_audio_path,
)
from asa.prompts import (
    PROMPT_TEMPLATE,
    build_expert_prompt_MOS,
    build_expert_prompt_MOS_DIS,
    build_expert_prompt_TEMPORAL,
)


def query_to_prompt(query: Any) -> str:
    """Convert a record ``query`` string into a Qwen2-Audio policy prompt.

    Falls back to ``PROMPT_TEMPLATE`` when the query is missing or empty. When a
    query is present, the audio placeholder is normalized to the real
    ``<|audio_bos|><|AUDIO|><|audio_eos|>`` block so the audio token is always
    present exactly once. Shared by both SFTDataset and DPODataset so the policy
    model sees the same task-specific prompt (e.g. the temporal query) at SFT and
    DPO time.
    """
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
        return query_to_prompt(query)

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
        dims_source_json: str | Path | None = None,
        use_discontinuity: bool = False,
    ):
        self.data_root = Path(data_root)
        self.samples = self._load_jsonl(Path(json_path))
        # Discontinuity-deviation ablation (global ALLD only): when True, the
        # MOS reference prompt is built from 5 scores incl. `dis` instead of 4.
        # Requires each record to carry a `dis` field. Default 4-dim path
        # (False) is byte-identical to the paper-faithful reference stream.
        self.use_discontinuity = use_discontinuity

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Temporal records (global-caption format) carry mos but NOT noi/col/loud
        # (build_nisqa_temporal_json drops them). For the ALLD temporal reference
        # prompt we need the full quality palette, so join noi/col/loud back from
        # the source caption file by degraded-filename basename. Only used for
        # temporal records; the MOS path never touches this.
        self.dims_index: dict[str, dict[str, float]] = {}
        if dims_source_json is not None:
            self.dims_index = self._build_dims_index(Path(dims_source_json))

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"DPODataset: loaded {len(self.samples)} samples from {json_path}")
            if self.dims_index:
                print(
                    f"DPODataset: loaded {len(self.dims_index)} quality-dimension "
                    f"rows from {dims_source_json} for the temporal reference prompt"
                )

    @staticmethod
    def _build_dims_index(path: Path) -> dict[str, dict[str, float]]:
        """Index {deg_basename: {mos, noi, col, loud}} from a caption JSONL."""
        index: dict[str, dict[str, float]] = {}
        for record in load_processed_records(path):
            audios = record.get("audios")
            if not isinstance(audios, list) or not audios:
                continue
            deg = Path(str(audios[0])).name
            try:
                index[deg] = {
                    field: float(record[field]) for field in DPO_METADATA_FIELDS
                }
            except (KeyError, TypeError, ValueError):
                continue
        return index

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        return load_processed_records(path)

    def _resolve_audio_path(self, raw_path: str) -> Path:
        return resolve_audio_path(raw_path, self.data_root)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_meta_prompt(self, item: Dict[str, Any]) -> str:
        """Build the ALLD reference prompt, MOS or temporal per record shape.

        Temporal records carry ``mix_deg_segments`` (the degradation interval)
        and lack ``noi/col/loud``; they use the temporal expert with the full
        quality palette joined from ``dims_index`` plus the interval. All other
        records take the original MOS expert path unchanged.
        """
        segments = item.get("mix_deg_segments")
        is_temporal = isinstance(segments, list) and len(segments) > 0
        if not is_temporal:
            if self.use_discontinuity:
                metadata = {
                    field: float(item[field]) for field in DPO_METADATA_FIELDS_DIS
                }
                return build_expert_prompt_MOS_DIS(**metadata)
            metadata = {field: float(item[field]) for field in DPO_METADATA_FIELDS}
            return build_expert_prompt_MOS(**metadata)

        deg = item.get("filename_deg") or Path(str(item["audios"][0])).name
        dims = self.dims_index.get(deg)
        if dims is None:
            raise KeyError(
                f"No quality dimensions (noi/col/loud) for temporal record {deg!r}. "
                "Pass dims_source_json pointing at the caption file (e.g. "
                "train_nisqa_llama_10k.json) so the temporal reference prompt "
                "can be built."
            )
        seg = segments[0]
        return build_expert_prompt_TEMPORAL(
            mos=float(item["mos"]),
            noi=dims["noi"],
            col=dims["col"],
            loud=dims["loud"],
            start=round(float(seg["start"]), 2),
            end=round(float(seg["end"]), 2),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_path = self._resolve_audio_path(item["audios"][0])
        audio_array = load_audio(str(audio_path))

        meta_prompt = self._build_meta_prompt(item)

        # Policy prompt mirrors the SFT stage: temporal records carry a task
        # `query` ("...identify when the degradation occurs"), so the policy model
        # is prompted the same way it was fine-tuned. MOS records have no `query`
        # and fall back to the bare PROMPT_TEMPLATE (unchanged behavior).
        audio_prompt = query_to_prompt(item.get("query"))

        return {
            "audio_prompt": audio_prompt,  # For Policy Model
            "meta_prompt": meta_prompt,  # For Reference Model
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "audio": audio_array,
            "sampling_rate": TARGET_SR,
        }
