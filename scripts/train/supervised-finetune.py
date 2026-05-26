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
        Path("data/processed/sft/train_nisqa_llama_10k.json"),
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
    warmup_ratio: float = typer.Option(
        0.0, help="Fraction of total steps used for LR warmup."
    ),
    lr_scheduler_type: str = typer.Option(
        "linear", help="LR scheduler (linear|cosine|constant|...)."
    ),
    gradient_accumulation_steps: int = typer.Option(
        4, help="Gradient accumulation steps."
    ),
    weight_decay: float = typer.Option(0.0, help="Weight decay."),
    label_smoothing_factor: float = typer.Option(0.0, help="Label smoothing factor."),
    bf16: bool = typer.Option(False, help="Use bf16 (A100/H100)."),
    fp16: bool = typer.Option(False, help="Use fp16 (V100)."),
    max_samples: Optional[int] = typer.Option(
        None, help="Limit dataset size for debugging."
    ),
    use_query_prompt: bool = typer.Option(
        False,
        help="Use per-record `query` as prompt when available.",
    ),
    deepspeed: Optional[str] = typer.Option(
        None, help="Path to DeepSpeed config JSON."
    ),
    val_split: float = typer.Option(
        0.05, help="Fraction of data to use for validation (0 to disable)."
    ),
    eval_steps: int = typer.Option(500, help="Run evaluation every N steps."),
    wandb_entity: Optional[str] = typer.Option(
        "speech-quality-DTU-bachelor", help="Weights & Biases team/entity name."
    ),
    wandb_project: Optional[str] = typer.Option(
        "qwen2-audio-sft-simple",
        help="Weights & Biases project name (None to disable).",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None, help="Weights & Biases run name."
    ),
    resume_from_checkpoint: Optional[str] = typer.Option(
        None,
        help="Path to a Trainer checkpoint dir to resume from (e.g. models/foo/checkpoint-565). Pass 'auto' to let Trainer pick the latest checkpoint in output_dir.",
    ),
    hub_model_id: Optional[str] = typer.Option(
        None,
        help="HF Hub repo id for streaming checkpoints (e.g. Leng2beat/foo). If set, enables push_to_hub.",
    ),
    save_steps: int = typer.Option(
        200, help="Steps between checkpoint saves. Used when hub_model_id is set."
    ),
    save_total_limit: int = typer.Option(
        1,
        help="Max local checkpoints to retain (rotation). Keeps /work3 quota bounded.",
    ),
    hub_private: bool = typer.Option(True, help="Make Hub repo private."),
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
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": lr_scheduler_type,
                "batch_size": batch_size,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "weight_decay": weight_decay,
                "label_smoothing_factor": label_smoothing_factor,
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
        use_query_prompt=use_query_prompt,
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
        torch_dtype=dtype,
    )

    push_to_hub = hub_model_id is not None
    if push_to_hub and is_main:
        print(
            f"Hub streaming ENABLED: pushing checkpoints to {hub_model_id} every {save_steps} steps "
            f"(local rotation: keep last {save_total_limit}; save_only_model=True so each local ckpt stays at model-weights size)."
        )
    elif is_main:
        print(
            "Hub streaming DISABLED (no --hub-model-id). Final save will land on local disk only "
            "— this is the fragile path. Pass --hub-model-id to make the run quota-safe."
        )

    # ── 4. Training args ─────────────────────────────────────────────────
    # Saving: stream to Hub every save_steps with save_only_model=True so each
    # rotated local ckpt is ~16 GB (model weights only) instead of ~63 GB
    # (DeepSpeed full ckpt). save_total_limit=1 keeps /work3 quota flat.
    # Final save is wrapped in try/except below; Hub push is the durable path.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing_factor,
        bf16=bf16,
        fp16=fp16,
        logging_steps=10,
        save_strategy="steps" if push_to_hub else "no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=True,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=eval_steps if val_dataset is not None else None,
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=wandb_run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="every_save" if push_to_hub else "end",
        hub_private_repo=hub_private,
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
    if resume_from_checkpoint is None:
        trainer.train()
    elif resume_from_checkpoint == "auto":
        if is_main:
            print(f"Resuming from latest checkpoint in {output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        if is_main:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if is_main:
        print(f"Saving model to {output_dir}")
    try:
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        local_save_ok = True
    except OSError as e:
        # Quota / disk-full: don't lose the run. Fall through to Hub push.
        local_save_ok = False
        if is_main:
            print(f"WARNING: local save failed ({e!r}). Attempting Hub-only rescue.")

    if push_to_hub and is_main:
        try:
            trainer.push_to_hub(commit_message="final model", blocking=True)
            processor.push_to_hub(hub_model_id, private=hub_private)
            print(f"Final checkpoint pushed to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"ERROR: final Hub push failed: {e!r}")
            if not local_save_ok:
                print("Both local save and Hub push failed — run output is LOST.")
            raise
    elif not local_save_ok:
        raise RuntimeError(
            "Local save failed and --hub-model-id was not set; run output is lost."
        )

    if wandb_project and is_main:
        wandb.finish()


if __name__ == "__main__":
    app()
