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
    set_seed,
)

from asa.data import Qwen2AudioCollator, SFTDataset
from asa.modeling_timeaudio import (
    Qwen2AudioTimeForConditionalGeneration,
    install_time_tokens,
)

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
    seed: int = typer.Option(
        42, help="Random seed for reproducibility (data split, init, shuffling)."
    ),
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
    use_abs_time_embedding: bool = typer.Option(
        False,
        "--use-abs-time-embedding/--no-abs-time-embedding",
        help=(
            "Add a learnable, zero-init absolute-time frame embedding to the "
            "audio features (TimeAudio mechanism 2). Off = bit-for-bit vanilla "
            "Qwen2-Audio, so the flag is a clean on/off ablation."
        ),
    ),
    install_time_tokens_flag: bool = typer.Option(
        False,
        "--install-time-tokens/--no-time-tokens",
        help=(
            "Register anchor/offset <a><f> time tokens and seed them from "
            "numeral embeddings (TimeAudio mechanism 1). Required when training "
            "on anchor-offset-localization targets."
        ),
    ),
):
    """Run supervised fine-tuning on Qwen2-Audio."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    # Seed everything (Python, NumPy, torch) before any model/data work.
    set_seed(seed)
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
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
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

    use_timeaudio = use_abs_time_embedding or install_time_tokens_flag
    if use_timeaudio:
        # Subclass carries the optional learnable absolute-time embedding. The
        # flag is written onto the config so the value round-trips through
        # save_pretrained/from_pretrained on the saved checkpoint.
        model = Qwen2AudioTimeForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        model.config.use_abs_time_embedding = use_abs_time_embedding
        model.use_abs_time_embedding = use_abs_time_embedding
        model.abs_time_embedding.weight.requires_grad_(use_abs_time_embedding)
        # Record token install on the config too. Both flags route eval loading
        # to the subclass: a tokens-only checkpoint has an extended vocab (and a
        # resized lm_head) that the stock class cannot absorb, so it must also
        # load via Qwen2AudioTimeForConditionalGeneration.
        model.config.use_time_tokens = install_time_tokens_flag
        if install_time_tokens_flag:
            num_added = install_time_tokens(model, processor, seed_from_numerals=True)
            if is_main:
                print(
                    f"Installed {num_added} anchor/offset time tokens "
                    f"(seeded from numeral embeddings)."
                )
        if is_main:
            print(
                f"TimeAudio: abs_time_embedding={'ON' if use_abs_time_embedding else 'OFF'}, "
                f"time_tokens={'ON' if install_time_tokens_flag else 'OFF'}."
            )
    else:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

    if is_main:
        print(
            "Save policy: FINAL ONLY (save_strategy='no'). No mid-training "
            "checkpoints are written, so /work3 never holds more than the one "
            "~16 GB final model per run. The final model is saved LOCALLY to "
            f"{output_dir}; push it to the Hub manually afterward if you want a "
            "durable off-scratch copy. A mid-run crash loses the run (no "
            "checkpoint to resume from), which is the accepted trade for not "
            "risking a quota overflow mid-save."
        )

    # ── 4. Training args ─────────────────────────────────────────────────
    # Final-save-only: save_strategy="no" means the Trainer writes nothing during
    # training. The single save_model() call after train() writes exactly one
    # ~16 GB copy to output_dir. This deliberately avoids the per-step
    # checkpoint + Hub-staging duplication that previously doubled on-disk
    # footprint to ~32 GB and overflowed the /work3 hard limit mid-save.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
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
        save_strategy="no",
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=eval_steps if val_dataset is not None else None,
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=wandb_run_name,
        push_to_hub=False,
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
    # Single final save to local disk. With save_strategy="no" this is the only
    # on-disk copy of the model, so /work3 holds at most ~16 GB for this run.
    # Push to the Hub manually afterward for a durable off-scratch copy.
    if is_main:
        print(f"Saving final model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    if is_main:
        print(f"Final model saved to {output_dir} (local only).")

    if wandb_project and is_main:
        wandb.finish()


if __name__ == "__main__":
    app()
