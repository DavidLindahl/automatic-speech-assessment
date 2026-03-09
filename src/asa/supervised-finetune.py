"""
supervised-finetune.py — SFT training script for Qwen2-Audio.

Imports SFTDataset and Qwen2AudioCollator from data.py.
Designed for multi-GPU training with DeepSpeed on HPC clusters.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import typer
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from asa.data import Qwen2AudioCollator, SFTDataset

app = typer.Typer()


@app.command()
def train(
    model_id: str = typer.Option(
        "Qwen/Qwen2-Audio-7B",
        help="HuggingFace model ID.",
    ),
    json_path: Path = typer.Option(
        Path("data/processed/train_nisqa_llama_10k.json"),
        help="Path to the JSONL training data.",
    ),
    data_root: Path = typer.Option(
        Path("data"),
        help="Root directory containing raw audio files.",
    ),
    model_name: str = typer.Option(
        ...,
        help="Name of the model to save (saved under models/<model_name>).",
    ),
    batch_size: int = typer.Option(4, help="Per-device batch size."),
    epochs: int = typer.Option(2, help="Number of training epochs."),
    lr: float = typer.Option(1e-5, help="Learning rate."),
    gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps."),
    bf16: bool = typer.Option(False, help="Use bf16 (A100/H100)."),
    fp16: bool = typer.Option(False, help="Use fp16 (V100)."),
    max_samples: Optional[int] = typer.Option(None, help="Limit dataset size for debugging."),
    deepspeed: Optional[str] = typer.Option(None, help="Path to DeepSpeed config JSON."),
    val_split: float = typer.Option(0.05, help="Fraction of data to use for validation (0 to disable)."),
    eval_steps: int = typer.Option(500, help="Run evaluation every N steps."),
    wandb_entity: Optional[str] = typer.Option("speech-quality-DTU-bachelor", help="Weights & Biases team/entity name."),
    wandb_project: Optional[str] = typer.Option("qwen2-audio-sft-simple", help="Weights & Biases project name (None to disable)."),
    wandb_run_name: Optional[str] = typer.Option(None, help="Weights & Biases run name."),
):
    """Run supervised fine-tuning on Qwen2-Audio."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    # ── 0. W&B setup ─────────────────────────────────────────────────────
    if wandb_project and is_main:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                "model_id": model_id,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "val_split": val_split,
                "max_samples": max_samples,
                "dtype": "bf16" if bf16 else "fp16" if fp16 else "fp32",
                "deepspeed": deepspeed is not None,
            },
        )
    report_to = "wandb" if wandb_project else "none"
    
    # ── 1. Output Dir Setup ──────────────────────────────────────────────
    output_dir = Path("models") / model_name
    if is_main:
        print(f"Model will be saved to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 2. Processor ─────────────────────────────────────────────────────
    if is_main:
        print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)
    # ── 2. Dataset + Collator ────────────────────────────────────────────
    full_dataset = SFTDataset(
        json_path=json_path,
        data_root=data_root,
        max_samples=max_samples,
    )
    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        if is_main:
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        if is_main:
            print(f"Train: {len(train_dataset)}, Val: disabled")
    collator = Qwen2AudioCollator(processor)
    # ── 3. Model ─────────────────────────────────────────────────────────
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if is_main:
        print(f"Loading model: {model_id} (dtype={dtype})")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
    )
    # ── 4. Training args ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=bf16,
        fp16=fp16,
        logging_steps=10,

        # --- OPTIMIZED SAVING STRATEGY ---
        save_strategy="no",
        # save_steps=500,               # Increased to reduce I/O overhead
        # save_total_limit=1,           # Reduced to 1. Will keep 1 latest + 1 best (2 total)
        # save_only_model=True,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",

        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=eval_steps if val_dataset is not None else None,
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=wandb_run_name,
    )
    # ── 5. Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    if is_main:
        print("Starting training...")
    trainer.train()
    if is_main:
        print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    if wandb_project and is_main:
        wandb.finish()

if __name__ == "__main__":
    app()