"""
data.py — SFT data pipeline for Qwen2-Audio fine-tuning.

Contains:
  - SFTDataset: PyTorch Dataset that loads JSONL + WAV files on-the-fly
  - Qwen2AudioCollator: batches samples and calls the Qwen2-Audio processor
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import librosa

import typer

TARGET_SR = 16_000  # Qwen2-Audio expects 16 kHz mono

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>Please describe and evaluate the synthetic speech."

app = typer.Typer()


@app.command()
def download(
    bucket_name: str = "nisqa-dataset",
    source_blob_name: str = ".",
    destination_path: Path = Path("data/raw"),
):
    """Downloads data from Google Cloud Storage to a local directory."""
    from google.cloud import storage

    print(
        f"Downloading from gs://{bucket_name}/{source_blob_name} to {destination_path}..."
    )

    # Ensure destination exists
    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        client = storage.Client()
    except Exception:
        print("No credentials found. Using anonymous access...")
        client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(
        prefix=source_blob_name if source_blob_name != "." else None
    )
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = os.path.relpath(blob.name, source_blob_name)
        local_path = destination_path / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(str(local_path))


@app.command()
def generate_captions(
    data_path: Path = typer.Argument(Path("data/raw")),
    output_folder: Path = typer.Argument(Path("data/processed")),
) -> None:
    """Stage 1: Preprocesses dataset for Supervised Fine-Tuning (SFT) and DPO."""
    print("Preprocessing data for SFT/DPO...")

    # 1. Run Sampler
    print("\n--- Step 1: Sampling Data ---")
    from src.asa.sampler import sample_data

    nisqa_corpus_path = data_path / "NISQA_Corpus"

    sample_data(nisqa_corpus_path, output_folder)

    # 2. Run Caption Generator
    print("\n--- Step 2: Generating Captions ---")
    from src.asa.caption_generator import process_single_file

    # Process MOS Dataset
    mos_input = output_folder / "mos_dataset.json"
    mos_output = output_folder / "train_nisqa_llama_10k.json"
    if mos_input.exists():
        if not mos_output.exists():
            print(f"Generating captions for MOS dataset: {mos_input} -> {mos_output}")
            process_single_file(str(mos_input), str(mos_output))
        else:
            print(
                f"Captions for MOS dataset already exist at {mos_output}, skipping generation."
            )

    # Process A/B Dataset
    ab_input = output_folder / "ab_dataset.json"
    ab_output = output_folder / "train_nisqa_abtest_llama_10k.json"
    if ab_input.exists():
        if not ab_output.exists():
            print(f"Generating captions for A/B dataset: {ab_input} -> {ab_output}")
            process_single_file(str(ab_input), str(ab_output))
        else:
            print(
                f"Captions for A/B dataset already exist at {ab_output}, skipping generation."
            )


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

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
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
            rel = raw_path[raw_path.find("NISQA_Corpus") :]
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
            prompt_len = (
                prompt_batch["input_ids"][i]
                .ne(self.processor.tokenizer.pad_token_id)
                .sum()
            )
            # Mask prompt tokens with -100 (ignore loss on prompt tokens)
            labels[i, :prompt_len] = -100
        # Also mask padding
        labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    app()
