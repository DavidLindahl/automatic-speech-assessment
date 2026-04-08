"""Single-step forward/backward smoke test for SFT."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import random_split
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from asa.data import Qwen2AudioCollator, SFTDataset

pytestmark = pytest.mark.slow

MODEL_ID = "Qwen/Qwen2-Audio-7B"
MAX_SAMPLES = 6


def test_sft_forward_backward_smoke() -> None:
    """Validate one forward/backward pass on GPU with mixed precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SFT forward/backward smoke test.")

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "train_nisqa_llama_10k.json"
    data_root = project_root / "data"

    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)

    full_dataset = SFTDataset(
        json_path=dataset_path,
        data_root=data_root,
        max_samples=MAX_SAMPLES,
    )

    val_size = max(1, int(len(full_dataset) * 0.33))
    train_size = len(full_dataset) - val_size
    train_dataset, _ = random_split(full_dataset, [train_size, val_size])

    collator = Qwen2AudioCollator(processor)
    batch = collator([train_dataset[0]])

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    model.to("cuda")
    model.train()
    model.gradient_checkpointing_enable()

    batch = {key: value.to("cuda") for key, value in batch.items()}

    with torch.amp.autocast("cuda", dtype=dtype):
        out = model(**batch)
        loss = out.loss

    loss.backward()

    assert torch.isfinite(loss).item(), "Loss is NaN or Inf"
