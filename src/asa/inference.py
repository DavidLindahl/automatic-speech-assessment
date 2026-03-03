"""
inference.py — Load a trained Qwen2-Audio checkpoint and run inference.

Public API
----------
load_model   : load processor + model from a local directory
run_inference : generate text responses for a list of audio files
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from asa.data import PROMPT_TEMPLATE, load_audio


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str | Path,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoProcessor, Qwen2AudioForConditionalGeneration, str]:
    """Load processor + model from a local checkpoint directory.

    Parameters
    ----------
    model_dir : path to the saved model (e.g. ``results/sft``).
    device    : ``"cuda"`` / ``"cpu"``.  Defaults to CUDA when available.
    dtype     : override the stored weight dtype (e.g. ``torch.float32``).

    Returns
    -------
    processor, model, device
    """
    model_dir = Path(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        low_cpu_mem_usage=True,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    return processor, model, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    audio_paths: Iterable[str | Path],
    device: Optional[str] = None,
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> List[str]:
    """Generate text responses for a list of audio files.

    Each file is loaded with :func:`asa.data.load_audio`, paired with the
    standard quality-evaluation prompt, and passed through
    ``model.generate``.

    Parameters
    ----------
    audio_paths    : paths to ``.wav`` files.
    device         : inferred from model parameters if not given.
    max_new_tokens : generation budget per sample.
    batch_size     : number of files to process at once (avoids OOM on
                     large file lists).

    Returns
    -------
    List of decoded model responses (prompt tokens are stripped).
    """
    if device is None:
        device = next(model.parameters()).device.type

    audio_paths = list(audio_paths)
    sr = processor.feature_extractor.sampling_rate
    all_responses: List[str] = []

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start : start + batch_size]

        texts = [PROMPT_TEMPLATE] * len(batch_paths)
        audios = [load_audio(str(p), target_sr=sr) for p in batch_paths]

        batch = processor(
            text=texts,
            audio=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        input_len = batch["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(**batch, max_new_tokens=max_new_tokens)

        # Strip prompt tokens so we only return the model's response
        response_ids = out_ids[:, input_len:]
        decoded = processor.batch_decode(response_ids, skip_special_tokens=True)
        all_responses.extend(decoded)

    return all_responses
