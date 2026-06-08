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
from huggingface_hub import HfApi
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from asa.data import Qwen2AudioCollator, SFTDataset
from asa.modeling_timeaudio import (
    Qwen2AudioTimeForConditionalGeneration,
    install_time_tokens,
)

app = typer.Typer()


class HubCheckpointCallback(TrainerCallback):
    """Upload each rotated checkpoint to the Hub directly from its folder.

    The stock ``hub_strategy="every_save"`` path copies all model shards from
    ``checkpoint-N/`` up into ``output_dir`` before uploading (transformers
    ``Trainer._push_from_checkpoint`` does ``shutil.copy`` per shard), so a
    16 GB checkpoint occupies ~32 GB on disk: once under ``checkpoint-N/`` and
    once at the top level. On a tight ``/work3`` quota that doubled footprint
    overflowed the hard limit mid-save and killed the job.

    This callback uploads straight from ``checkpoint-N/`` (which the Trainer
    already wrote and rotates via ``save_total_limit``), so there is no second
    on-disk copy. ``push_to_hub`` is left False on the Trainer so the stock path
    never runs. Uploads are best-effort: a failed push logs and continues rather
    than killing a healthy training run.
    """

    def __init__(self, repo_id: str, private: bool = False) -> None:
        self._repo_id = repo_id
        self._private = private
        self._api = HfApi()
        self._created = False

    def _ensure_repo(self) -> None:
        if not self._created:
            self._api.create_repo(
                self._repo_id, private=self._private, exist_ok=True
            )
            self._created = True

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        """Upload the just-written checkpoint folder from rank 0."""
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        if not os.path.isdir(ckpt_dir):
            print(f"WARNING: checkpoint dir not found for upload: {ckpt_dir}")
            return
        try:
            self._ensure_repo()
            self._api.upload_folder(
                repo_id=self._repo_id,
                folder_path=ckpt_dir,
                commit_message=f"Training checkpoint, step {state.global_step}",
            )
            print(f"Uploaded step {state.global_step} to {self._repo_id}")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: Hub upload of step {state.global_step} failed: {e!r}")


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
    # Saving: write a rotated local checkpoint every save_steps with
    # save_only_model=True so each ckpt is ~16 GB (model weights only) instead
    # of ~63 GB (DeepSpeed full ckpt). save_total_limit=1 keeps /work3 flat.
    #
    # Hub upload is done by HubCheckpointCallback (added below), NOT by the
    # built-in push_to_hub path. The built-in path (hub_strategy="every_save")
    # shutil.copies every shard from checkpoint-N/ into output_dir before
    # uploading, doubling on-disk footprint to ~32 GB per run, which overflowed
    # the /work3 quota mid-save and killed a run. So push_to_hub stays False and
    # the callback uploads straight from checkpoint-N/ (no second copy).
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
    if push_to_hub:
        trainer.add_callback(
            HubCheckpointCallback(repo_id=hub_model_id, private=hub_private)
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

    # Final Hub upload via HfApi (the built-in trainer.push_to_hub is disabled).
    # On a clean local save, upload the final model from output_dir (one copy on
    # disk). If the local save failed (quota), the per-step HubCheckpointCallback
    # has already streamed the latest rotated checkpoint to the Hub, so that is
    # the durable copy; uploading the partial output_dir would corrupt the repo,
    # so we skip it and rely on the callback's last good push.
    if push_to_hub and is_main:
        if local_save_ok:
            try:
                api = HfApi()
                api.create_repo(hub_model_id, private=hub_private, exist_ok=True)
                api.upload_folder(
                    repo_id=hub_model_id,
                    folder_path=str(output_dir),
                    commit_message="final model",
                    ignore_patterns=[f"{PREFIX_CHECKPOINT_DIR}-*"],
                )
                print(
                    f"Final checkpoint pushed to https://huggingface.co/{hub_model_id}"
                )
            except Exception as e:
                print(f"ERROR: final Hub push failed: {e!r}")
                raise
        else:
            print(
                "Local save failed; relying on the last per-step checkpoint already "
                f"streamed to https://huggingface.co/{hub_model_id} by the callback."
            )
    elif not local_save_ok:
        raise RuntimeError(
            "Local save failed and --hub-model-id was not set; run output is lost."
        )

    if wandb_project and is_main:
        wandb.finish()


if __name__ == "__main__":
    app()
