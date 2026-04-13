"""DeepSpeed distributed training smoke test.

This test is intentionally marked ``slow`` and is designed for 2-GPU
Linux environments launched under ``torchrun``.
"""

from pathlib import Path
import os

import pytest
import torch
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from asa.data import Qwen2AudioCollator, SFTDataset

pytestmark = pytest.mark.slow

MODEL_ID = "Qwen/Qwen2-Audio-7B"
MAX_SAMPLES = 6
BATCH_SIZE = 1


def test_deepspeed_zero2_training_smoke(tmp_path: Path) -> None:
    """Run a minimal ZeRO-2 training smoke test in distributed GPU mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DeepSpeed smoke test.")

    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs are required for this test.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("Run under torchrun with WORLD_SIZE>=2 to execute this test.")

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "train_nisqa_llama_10k.json"
    data_root = project_root / "data"
    ds_config = project_root / "configs" / "ds_zero2.json"

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
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    collator = Qwen2AudioCollator(processor)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    output_dir = tmp_path / "deepspeed_test"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        deepspeed=str(ds_config),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    assert output_dir.exists()
