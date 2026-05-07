"""
dpo-finetune.py — Alignment with LLM Distillation (ALLD) script.

Implements the cross-modal ALLD method from the paper:
- Policy Model: Qwen2-Audio (trainable, processes audio + text)
- Reference Model: Qwen2-Text (frozen, processes metadata + text)
"""

import os
from pathlib import Path
from typing import Optional

import torch
import typer
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# Import the new dataset and dual-stream collator
from asa.data import DPODataset, ALLDDPOCollator

app = typer.Typer()


class ALLDDPOTrainer(Trainer):
    """Custom Trainer implementing the ALLD cross-modal DPO loss."""

    def __init__(self, ref_model, beta=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

        # --- NEW: Safely move the reference model to the correct GPU ---
        if ref_model is not None:
            # We use the trainer's built-in accelerator to handle DeepSpeed & Multi-GPU
            self.ref_model = self.accelerator.prepare_model(
                ref_model, evaluation_mode=True
            )
        else:
            self.ref_model = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Route the prefixed inputs to their respective dictionaries
        policy_inputs = {
            "input_ids": inputs["policy_input_ids"],
            "attention_mask": inputs["policy_attention_mask"],
            "labels": inputs["policy_labels"],
        }
        # Safely handle audio features depending on the exact Qwen2 version
        if (
            "policy_audio_values" in inputs
            and inputs["policy_audio_values"] is not None
        ):
            policy_inputs["audio_values"] = inputs["policy_audio_values"]
        if (
            "policy_audio_features" in inputs
            and inputs["policy_audio_features"] is not None
        ):
            policy_inputs["audio_features"] = inputs["policy_audio_features"]

        ref_inputs = {
            "input_ids": inputs["ref_input_ids"],
            "attention_mask": inputs["ref_attention_mask"],
            "labels": inputs["ref_labels"],
        }

        # Batches are 2N (chosen first half, rejected second half)
        batch_size = policy_inputs["input_ids"].shape[0] // 2

        # 2. Forward pass active Policy Model (Audio LLM)
        outputs = model(**policy_inputs)
        policy_logits = outputs.logits

        # 3. Forward pass frozen Reference Model (Text LLM)
        with torch.no_grad():
            ref_outputs = self.ref_model(**ref_inputs)
            ref_logits = ref_outputs.logits

        def get_logprobs(logits, labels):
            # Shift to token prediction
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]

            loss_mask = labels != -100
            per_token_logprobs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(labels.shape)

            return (per_token_logprobs * loss_mask).sum(dim=1) / loss_mask.sum(
                dim=1
            ).clamp(min=1)

        policy_logprobs = get_logprobs(policy_logits, policy_inputs["labels"])
        ref_logprobs = get_logprobs(ref_logits, ref_inputs["labels"])

        # Split logprobs into chosen and rejected halves
        policy_chosen = policy_logprobs[:batch_size]
        policy_rejected = policy_logprobs[batch_size:]
        ref_chosen = ref_logprobs[:batch_size]
        ref_rejected = ref_logprobs[batch_size:]

        # Calculate standard DPO loss using the cross-modal logprobs
        logits_diff = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        loss = -torch.nn.functional.logsigmoid(self.beta * logits_diff).mean()

        # Extract logging metrics
        if self.state.is_world_process_zero:
            chosen_rewards = self.beta * (policy_chosen - ref_chosen).detach()
            rejected_rewards = self.beta * (policy_rejected - ref_rejected).detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            self.log(
                {
                    "rewards/chosen": chosen_rewards.mean().item(),
                    "rewards/rejected": rejected_rewards.mean().item(),
                    "rewards/accuracies": reward_accuracies.mean().item(),
                    "rewards/margins": (chosen_rewards - rejected_rewards)
                    .mean()
                    .item(),
                }
            )

        if return_outputs:
            return loss, outputs
        return loss


@app.command()
def train(
    model_id: str = typer.Option(
        "Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup",
        help="Path to SFT Audio model (Policy).",
    ),
    ref_model_id: str = typer.Option(
        "Qwen/Qwen2-7B", help="Path to Expert Text model (Reference)."
    ),
    json_path: Path = typer.Option(
        Path("data/processed/train_dpo_10k.json"), help="DPO dataset."
    ),
    data_root: Path = typer.Option(Path("data"), help="Root directory for audios."),
    model_name: str = typer.Option(
        ..., help="Name of the model to save (saved under models/<model_name>)."
    ),
    batch_size: int = typer.Option(2, help="Per-device batch size."),
    epochs: int = typer.Option(2, help="Training epochs."),
    beta: float = typer.Option(0.4, help="DPO margin parameter beta."),
    lr: float = typer.Option(5e-6, help="Learning rate."),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation."),
    bf16: bool = typer.Option(False, help="Use bf16."),
    fp16: bool = typer.Option(False, help="Use fp16."),
    max_samples: Optional[int] = typer.Option(None, help="Limit dataset size."),
    deepspeed: Optional[str] = typer.Option(None, help="Path to DeepSpeed config."),
    val_split: float = typer.Option(
        0, help="Validation fraction."
    ),  # Set to 0 to disable validation because of costum Trainer
    eval_steps: int = typer.Option(
        0, help="Eval interval."
    ),  # Set to 0 to disable validation because of costum Trainer
    wandb_entity: Optional[str] = typer.Option(
        "speech-quality-DTU-bachelor", help="W&B entity."
    ),
    wandb_project: Optional[str] = typer.Option(
        "qwen2-audio-alld", help="W&B project."
    ),
    wandb_run_name: Optional[str] = typer.Option("alld-finetune", help="W&B run name."),
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
    """Run ALLD (Alignment with LLM Distillation) on Qwen2-Audio."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if wandb_project and is_main:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                "model_id": model_id,
                "ref_model_id": ref_model_id,
                "learning_rate": lr,
                "batch_size": batch_size,
                "beta": beta,
                "epochs": epochs,
            },
        )
    report_to = "wandb" if wandb_project else "none"

    output_dir = Path("models") / model_name
    if is_main:
        print(f"Model will be saved to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    if is_main:
        print(f"Loading Policy processor: {model_id}")
    audio_processor = AutoProcessor.from_pretrained(model_id)

    if is_main:
        print(f"Loading Reference tokenizer: {ref_model_id}")
    text_tokenizer = AutoTokenizer.from_pretrained(ref_model_id)

    # Prepare Dataset
    full_dataset = DPODataset(
        json_path=json_path,
        data_root=data_root,
        max_samples=max_samples,
    )

    if val_split > 0:
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = None

    # Initialize Dual-Stream Collator
    collator = ALLDDPOCollator(
        audio_processor=audio_processor, text_tokenizer=text_tokenizer
    )

    # 1. Load Policy Model (Audio)
    if is_main:
        print(f"Loading Policy Model (Audio): {model_id} (dtype={dtype})")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype
    )

    # 2. Load Reference Model (Text)
    if is_main:
        print(f"Loading Reference Model (Text): {ref_model_id} (dtype={dtype})")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, torch_dtype=dtype)

    push_to_hub = hub_model_id is not None
    if push_to_hub and is_main:
        print(
            f"Hub streaming ENABLED: pushing checkpoints to {hub_model_id} every {save_steps} steps "
            f"(local rotation: keep last {save_total_limit})."
        )
    elif is_main:
        print(
            "Hub streaming DISABLED (no --hub-model-id). Final save will land on local disk only "
            "— this is the fragile path. Pass --hub-model-id to make the run quota-safe."
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=bf16,
        fp16=fp16,
        logging_steps=10,
        save_strategy="steps" if push_to_hub else "no",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=True,  # Carl never resumes from checkpoint; skip optimizer/scheduler/RNG state to save ~47GB per checkpoint and avoid OOM at save time.
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=eval_steps if val_dataset is not None else 0,
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        remove_unused_columns=False,  # Essential: keep custom prefix columns in batch
        report_to=report_to,
        run_name=wandb_run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="every_save" if push_to_hub else "end",
        hub_private_repo=hub_private,
    )

    trainer = ALLDDPOTrainer(
        ref_model=ref_model,
        beta=beta,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    if is_main:
        print("Starting ALLD training...")
    trainer.train()

    if is_main:
        print(f"Saving ALLD model to {output_dir}")
    try:
        trainer.save_model(str(output_dir))
        audio_processor.save_pretrained(str(output_dir))
        local_save_ok = True
    except OSError as e:
        # Quota / disk-full: don't lose the run. Fall through to Hub push.
        local_save_ok = False
        if is_main:
            print(f"WARNING: local save failed ({e!r}). Attempting Hub-only rescue.")

    if push_to_hub and is_main:
        # Trainer.push_to_hub uploads model + tokenizer; processor pushed separately.
        try:
            trainer.push_to_hub(commit_message="final model", blocking=True)
            audio_processor.push_to_hub(hub_model_id, private=hub_private)
            print(f"Final checkpoint pushed to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"ERROR: final Hub push failed: {e!r}")
            if not local_save_ok:
                print("Both local save and Hub push failed — run output is LOST.")
            raise
    elif not local_save_ok:
        # No Hub fallback configured and local save died.
        raise RuntimeError(
            "Local save failed and --hub-model-id was not set; run output is lost."
        )

    if wandb_project and is_main:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    app()
