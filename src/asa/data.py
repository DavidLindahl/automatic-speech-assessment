"""
data.py — SFT data pipeline for Qwen2-Audio fine-tuning.

Contains:
  - SFTDataset: PyTorch Dataset that loads JSONL + WAV files on-the-fly
  - Qwen2AudioCollator: batches samples and calls the Qwen2-Audio processor
"""

import os
import json
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
    DPO_METADATA_FIELDS_AB,
    load_processed_records,
    resolve_audio_path,
)


TARGET_SR = 16_000  # Qwen2-Audio expects 16 kHz mono

PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>Please describe and evaluate the synthetic speech."
PROMPT_TEMPLATE_AB = (
    "Please perform A/B preference test between<audio>and<audio>, including a tie."
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
    ):
        self.data_root = Path(data_root)
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

    @staticmethod
    def _is_valid(item):
        if not item.get("audios") or not isinstance(item["audios"], list) or len(item["audios"]) == 0:
            return False
        # Check if the path exists
        raw_path = item["audios"][0]
        candidate = Path(raw_path)
        if candidate.exists():
            return True
        normalized = raw_path.replace("\\", "/")
        if "NISQA_Corpus/" in normalized:
            relative = normalized.split("NISQA_Corpus/", maxsplit=1)[1]
            root = Path("data")
            return (root / "raw" / "NISQA_Corpus" / relative).exists()
        return False

    def _resolve_audio_path(self, raw_path: str) -> Path:
        """Map stored audio paths to a local path."""
        return resolve_audio_path(raw_path, self.data_root)

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


# ---------------------------------------------------------------------------
# AB-Test Dataset
# ---------------------------------------------------------------------------

AUDIO_PLACEHOLDER = "<audio>"
AUDIO_SPECIAL = "<|audio_bos|><|AUDIO|><|audio_eos|>"


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


# ---------------------------------------------------------------------------
# AB-Test Collator
# ---------------------------------------------------------------------------


class Qwen2AudioCollatorAB(Qwen2AudioCollator):
    """
    A/B preference variant of Qwen2AudioCollator.

    Each sample contributes *two* audios.  The Qwen2-Audio processor expects a
    **flat** list of waveforms — it assigns them to ``<|AUDIO|>`` tokens
    sequentially across the batch.
    """

    def _prepare_inputs(self, features):
        """Return (prompts, full_texts, audios) — audios flattened."""
        prompts = [f["prompt"] for f in features]
        
        # ADD THE EOS TOKEN HERE FOR A/B SFT:
        eos_token = self.processor.tokenizer.eos_token
        full_texts = [f["prompt"] + f["response"] + eos_token for f in features]
        
        # Flat list: [sample0_a, sample0_b, sample1_a, sample1_b, ...]
        audios = [audio for f in features for audio in [f["audio_a"], f["audio_b"]]]
        
        return prompts, full_texts, audios


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

def build_expert_prompt_MOS(
    mos: float, noi: float, col: float, loud: float
) -> str:
    current_input = f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}}}\nOutput:"
    return DIMENSION_DEFINITIONS_MOS + EXPERT_TASK_MOS + EXPERT_FEW_SHOT_EXAMPLES_MOS + current_input


# ==========================================
# 2. A/B TEST PROMPTS (Includes 'dis' feature)
# ==========================================

DIMENSION_DEFINITIONS_AB = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
    (5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

EXPERT_TASK_AB = """I need you to perform A/B test according to their mos (mos higher means winner). You can flexibly
select 1 to 3 aspects from the sub-dimensions with an obvious gap (usually score difference more than 0.5), then
compare them according to these distinctions. Finally, please give your preference with a reasonable
analysis.
"""

# Restored the original examples that correctly use the 'dis' feature
EXPERT_FEW_SHOT_EXAMPLES_AB = """
--- Example 1 ---
Input: {A_mos: 1.8, A_noi: 3.2, A_col: 1.9, A_dis: 2.3, A_loud: 3.3, B_mos: 3.6, B_noi: 2.6, B_col: 2.8, B_dis: 4.0, B_loud: 3.7}
Output: SpeechA and SpeechB have similar levels of distortion and loudness. However, SpeechB has better continuity than SpeechA. Although SpeechA is slightly cleaner, I would select SpeechB as better synthesized speech due to its significantly higher overall synthetic quality and better continuity.

--- Example 2 ---
Input: {A_mos: 4.0, A_noi: 4.2, A_col: 3.7, A_dis: 3.9, A_loud: 4.1, B_mos: 1.6, B_noi: 2.4, B_col: 1.8, B_dis: 2.5, B_loud: 1.6}
Output: SpeechA and SpeechB have significant gaps in several aspects. SpeechA has much lower noise, less distortion, and better continuity than SpeechB. Additionally, SpeechA is also much louder than SpeechB. Considering these substantial differences, I would select SpeechA as the better synthesized speech.
"""

def build_expert_prompt_ab(
    A_mos: float, A_noi: float, A_col: float, A_dis: float, A_loud: float,
    B_mos: float, B_noi: float, B_col: float, B_dis: float, B_loud: float,
) -> str:
    current_input = f"\n--- Current Task ---\nInput: {{A_mos: {A_mos}, A_noi: {A_noi}, A_col: {A_col}, A_dis: {A_dis}, A_loud: {A_loud}, B_mos: {B_mos}, B_noi: {B_noi}, B_col: {B_col}, B_dis: {B_dis}, B_loud: {B_loud}}}\nOutput:"
    return DIMENSION_DEFINITIONS_AB + EXPERT_TASK_AB + EXPERT_FEW_SHOT_EXAMPLES_AB + current_input

    
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

        # Ensure text tokenizer has a pad token
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

    def _build_labels(self, batch, prompt_batch, tokenizer):
        """Mask prompt and padding tokens with -100 in labels."""
        labels = batch["input_ids"].clone()
        for i in range(len(prompt_batch["input_ids"])):
            prompt_len = prompt_batch["input_ids"][i].ne(tokenizer.pad_token_id).sum()
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
        policy_rejected = [f["audio_prompt"] + f["rejected"] + audio_eos for f in features]
        audios = [f["audio"] for f in features]

        # 2N Batching for DeepSpeed
        policy_texts = policy_chosen + policy_rejected
        policy_prompts = audio_prompts + audio_prompts
        concat_audios = audios + audios

        policy_inputs = self.audio_processor(
            text=policy_texts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audio=concat_audios,
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
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_inputs, self.audio_processor.tokenizer
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
        batch["ref_labels"] = self._build_labels(
            ref_inputs, ref_prompt_inputs, self.text_tokenizer
        )

        return batch


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
            "audio_prompt": prompt,        # For Policy Model
            "meta_prompt": meta_prompt,    # For Reference Model
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "audio_a": audio_a,
            "audio_b": audio_b,
            "sampling_rate": TARGET_SR,
        }


class ALLDDPOCollatorAB(ALLDDPOCollator):
    """
    A/B preference variant of ALLDDPOCollator.
    The primary difference is flattening the dual `audio_a` and `audio_b` into
    a single sequence so Qwen2-Audio's processor correctly maps them to the
    two <|AUDIO|> tags in the prompt.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # ==========================================
        # 1. STREAM A: Policy Model (Audio + Text)
        # ==========================================
        audio_eos = self.audio_processor.tokenizer.eos_token
        
        audio_prompts = [f["audio_prompt"] for f in features]
        
        # FIX: Append EOS to chosen/rejected
        policy_chosen = [f["audio_prompt"] + f["chosen"] + audio_eos for f in features]
        policy_rejected = [f["audio_prompt"] + f["rejected"] + audio_eos for f in features]
        
        # Flatten audios for AB tests: [sample0_a, sample0_b, sample1_a, sample1_b, ...]
        audios = [audio for f in features for audio in [f["audio_a"], f["audio_b"]]]

        # 2N Batching for DeepSpeed
        policy_texts = policy_chosen + policy_rejected
        policy_prompts = audio_prompts + audio_prompts
        concat_audios = audios + audios

        policy_inputs = self.audio_processor(
            text=policy_texts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["policy_input_ids"] = policy_inputs["input_ids"]
        batch["policy_attention_mask"] = policy_inputs["attention_mask"]
        batch["policy_audio_values"] = policy_inputs.get("audio_values", None)
        batch["policy_audio_features"] = policy_inputs.get("audio_features", None)
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_inputs, self.audio_processor.tokenizer
        )

        # ==========================================
        # 2. STREAM B: Reference Model (Text Only)
        # ==========================================
        text_eos = self.text_tokenizer.eos_token
        
        meta_prompts = [f["meta_prompt"] for f in features]
        
        # FIX: Append EOS to chosen/rejected
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
        batch["ref_labels"] = self._build_labels(
            ref_inputs, ref_prompt_inputs, self.text_tokenizer
        )

        return batch