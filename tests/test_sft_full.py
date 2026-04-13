"""End-to-end SFT training smoke test."""

from pathlib import Path
import shutil

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


def test_sft_full_training_smoke(tmp_path: Path) -> None:
    """Run a tiny SFT training loop to verify trainer wiring."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SFT full training smoke test.")

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
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    collator = Qwen2AudioCollator(processor)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    output_dir = tmp_path / "sft_test"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=1,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
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

    # Keep this test from leaving behind large artifacts when run manually.
    if output_dir.exists():
        shutil.rmtree(output_dir)
