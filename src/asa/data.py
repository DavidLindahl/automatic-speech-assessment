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

from asa.processed_data import (
    DPO_METADATA_FIELDS,
    load_processed_records,
    resolve_audio_path,
)


TARGET_SR = 16_000  # Qwen2-Audio expects 16 kHz mono

AUDIO_PLACEHOLDER = "<audio>"
AUDIO_SPECIAL = "<|audio_bos|><|AUDIO|><|audio_eos|>"
# The trailing newline is a deliberate prompt/response delimiter. Without it,
# Qwen BPE merges the prompt tail with the first response word ("speech.This"
# -> ".This" as one token), so the prompt-length label mask hides the first
# response token and the model is never trained to produce position 0 at
# inference. That distribution shift drives the DPO EOS-collapse (the model
# defaults to <|im_end|> at the first generated position). The "\n" breaks the
# merge: "speech.\nThis" tokenizes with "This" as a clean standalone token.
# Shared by SFT, DPO and inference so all three see the identical prompt.
PROMPT_TEMPLATE = (
    f"{AUDIO_SPECIAL}Please describe and evaluate the synthetic speech.\n"
)


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

    # ── private ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """Parse line-delimited JSON (one JSON object per line).

        Keeps memory usage reasonable by reading line-by-line.
        """
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
        # Check if the path exists
        raw_path = item["audios"][0]
        return self._resolve_audio_path(raw_path).exists()

    def _resolve_audio_path(self, raw_path: str) -> Path:
        """Map stored audio paths to a local path."""
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

    # ── public ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Resolve and load audio
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

    def _prepare_inputs(self, features):
        prompts = [f["prompt"] for f in features]

        # ADD THE EOS TOKEN HERE:
        eos_token = self.processor.tokenizer.eos_token
        full_texts = [f["prompt"] + f["response"] + eos_token for f in features]
        audios = [f["audio"] for f in features]
        return prompts, full_texts, audios

    def _build_labels(self, batch, prompt_batch, features):
        """Mask prompt and padding tokens with -100 in labels."""
        labels = batch["input_ids"].clone()
        for i in range(len(features)):
            prompt_len = (
                prompt_batch["input_ids"][i]
                .ne(self.processor.tokenizer.pad_token_id)
                .sum()
            )
            labels[i, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts, full_texts, audios = self._prepare_inputs(features)

        batch = self.processor(
            text=full_texts,
            audios=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        prompt_batch = self.processor(
            text=prompts,
            audios=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["labels"] = self._build_labels(batch, prompt_batch, features)
        return batch


# --- ALLD Expert Text Prompt Templates ---
# ==========================================
# 1. SINGLE MOS PROMPTS (No 'dis' feature)
# ==========================================

DIMENSION_DEFINITIONS_MOS = """I will give you a tuple of meta information for speech quality evaluation, it contains 4 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

EXPERT_TASK_MOS = """I need you to generate a descriptive evaluation for this speech, including a description according to
the score from noise, coloration, and loudness, analyze how they influence the overall quality, and add the mos in the end.
"""

EXPERT_FEW_SHOT_EXAMPLES_MOS = """
--- Example 1 ---
Input: {mos: 4.5, noi: 5.0, col: 4.5, loud: 4.8}
Output: This speech is highly intelligible and perfectly loud. There is no background noise, and there is only a very slight coloration that is barely noticeable. Taking all factors into account, the overall MOS is 4.5.

--- Example 2 ---
Input: {mos: 2.1, noi: 3.0, col: 2.5, loud: 4.0}
Output: The volume of the speech is clear and adequately loud. However, there is moderate background noise and noticeable distortion. These degradations make the speech sound unnatural overall, so the MOS score is only 2.1.
"""


def build_expert_prompt_MOS(mos: float, noi: float, col: float, loud: float) -> str:
    # The trailing "\n" after "Output:" is the reference-stream analogue of
    # the PROMPT_TEMPLATE "\n" delimiter fix (commit a007248). Without it,
    # "Output:" + "The" merges to a single BPE token "Output:The", so the
    # rejected reference stream's first supervised token becomes
    # " synthesized" instead of "The". That misaligns the DPO reward at
    # position 0: policy sees "The", reference sees " synthesized". The "\n"
    # makes the prompt/response boundary a clean split, identical to the
    # policy stream. Verified by probe_collator_labels.py.
    current_input = f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}}}\nOutput:\n"
    return (
        DIMENSION_DEFINITIONS_MOS
        + EXPERT_TASK_MOS
        + EXPERT_FEW_SHOT_EXAMPLES_MOS
        + current_input
    )


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


class ALLDDPOCollator:
    """
    Dual-stream Collator for the ALLD method.
    Processes audio + text for the Policy Model.
    Processes text-only metadata for the Reference Model.
    """

    def __init__(self, audio_processor, text_tokenizer):
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

        # Force right-padding on both tokenizers. _build_labels assumes the real
        # prompt+response starts at index 0; under left-padding (the Qwen2-Audio
        # processor default) the prompt starts after a variable run of PAD
        # tokens, so labels[:prompt_len] masks PADs and leaves the real prompt
        # supervised as response. Confirmed root cause of the DPO collapse
        # (diagnostic 28376116). Right-padding makes the per-row prompt length
        # an exact prefix mask. Reverting this re-introduces the collapse.
        self.audio_processor.tokenizer.padding_side = "right"
        self.text_tokenizer.padding_side = "right"

        # Ensure text tokenizer has a pad token
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

    def _build_labels(self, batch, prompt_lengths):
        """Mask prompt and padding tokens with -100 in labels.

        prompt_lengths: per-row count of real (non-pad) prompt tokens, computed
        from the prompt-only attention_mask. With right-padding the prompt is an
        exact prefix, so labels[:prompt_len] cleanly masks only the prompt. The
        trailing PAD is masked separately via the full batch attention_mask,
        which is robust even when pad_token_id == eos_token_id (true for the
        Qwen2-7B reference model) since it keys on the mask, not the token id.
        """
        labels = batch["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # ==========================================
        # 1. STREAM A: Policy Model (Audio + Text)
        # ==========================================

        # GET THE EOS TOKEN FOR THE POLICY MODEL
        audio_eos = self.audio_processor.tokenizer.eos_token

        audio_prompts = [f["audio_prompt"] for f in features]

        # APPEND EOS TOKEN TO THE RESPONSES
        policy_chosen = [f["audio_prompt"] + f["chosen"] + audio_eos for f in features]
        policy_rejected = [
            f["audio_prompt"] + f["rejected"] + audio_eos for f in features
        ]
        audios = [f["audio"] for f in features]

        # 2N Batching for DeepSpeed
        policy_texts = policy_chosen + policy_rejected
        policy_prompts = audio_prompts + audio_prompts
        concat_audios = audios + audios

        policy_inputs = self.audio_processor(
            text=policy_texts,
            audios=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audios=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["policy_input_ids"] = policy_inputs["input_ids"]
        batch["policy_attention_mask"] = policy_inputs["attention_mask"]
        batch["policy_audio_values"] = policy_inputs.get(
            "audio_values", None
        )  # Handle processor variations
        batch["policy_audio_features"] = policy_inputs.get("audio_features", None)
        # Prompt length per row = count of real tokens in the prompt-only batch.
        # Derived from attention_mask, not token-id != pad, so it is correct
        # even when pad_token_id collides with a content token.
        policy_prompt_lens = policy_prompt_inputs["attention_mask"].sum(dim=1)
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_lens
        )

        # ==========================================
        # 2. STREAM B: Reference Model (Text Only)
        # ==========================================

        # GET THE EOS TOKEN FOR THE REFERENCE MODEL
        text_eos = self.text_tokenizer.eos_token

        meta_prompts = [f["meta_prompt"] for f in features]

        # APPEND EOS TOKEN TO THE RESPONSES
        ref_chosen = [f["meta_prompt"] + f["chosen"] + text_eos for f in features]
        ref_rejected = [f["meta_prompt"] + f["rejected"] + text_eos for f in features]

        # 2N Batching for DeepSpeed
        ref_texts = ref_chosen + ref_rejected
        concat_meta_prompts = meta_prompts + meta_prompts

        ref_inputs = self.text_tokenizer(
            ref_texts,
            return_tensors="pt",
            padding=True,
        )

        ref_prompt_inputs = self.text_tokenizer(
            concat_meta_prompts,
            return_tensors="pt",
            padding=True,
        )

        batch["ref_input_ids"] = ref_inputs["input_ids"]
        batch["ref_attention_mask"] = ref_inputs["attention_mask"]
        ref_prompt_lens = ref_prompt_inputs["attention_mask"].sum(dim=1)
        batch["ref_labels"] = self._build_labels(ref_inputs, ref_prompt_lens)

        return batch


