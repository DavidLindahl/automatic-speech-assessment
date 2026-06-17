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
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# Import the new dataset and dual-stream collator
from asa.data import DPODataset, ALLDDPOCollator
from asa.modeling_timeaudio import Qwen2AudioTimeForConditionalGeneration

app = typer.Typer()


class ALLDDPOTrainer(Trainer):
    """Custom Trainer implementing the ALLD cross-modal DPO loss."""

    def __init__(self, ref_model, beta=0.4, length_norm=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.length_norm = length_norm

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
        # Feed the audio tower. The collator emits `policy_input_features` +
        # `policy_feature_attention_mask` (the Qwen2-Audio 4.48.x processor
        # names). These MUST reach the model: without them the policy runs
        # text-only and no audio grounding is learned. The legacy
        # `policy_audio_values` / `policy_audio_features` keys are also honored
        # for backward compatibility with older collated batches.
        if inputs.get("policy_input_features") is not None:
            policy_inputs["input_features"] = inputs["policy_input_features"]
        elif inputs.get("policy_audio_values") is not None:
            policy_inputs["input_features"] = inputs["policy_audio_values"]
        if inputs.get("policy_feature_attention_mask") is not None:
            policy_inputs["feature_attention_mask"] = inputs[
                "policy_feature_attention_mask"
            ]
        elif inputs.get("policy_audio_features") is not None:
            policy_inputs["feature_attention_mask"] = inputs["policy_audio_features"]

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

            summed_logprobs = (per_token_logprobs * loss_mask).sum(dim=1)
            if not self.length_norm:
                # Ablation: raw summed completion log-prob (original DPO/ALLD
                # objective before the 2026-04-14 length-normalisation fix).
                return summed_logprobs
            return summed_logprobs / loss_mask.sum(dim=1).clamp(min=1)

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
        Path("data/processed/dpo/train_dpo_10k.json"), help="DPO dataset."
    ),
    data_root: Path = typer.Option(Path("data"), help="Root directory for audios."),
    model_name: str = typer.Option(
        ..., help="Name of the model to save (saved under models/<model_name>)."
    ),
    batch_size: int = typer.Option(2, help="Per-device batch size."),
    epochs: int = typer.Option(2, help="Training epochs."),
    beta: float = typer.Option(0.4, help="DPO margin parameter beta."),
    length_norm: bool = typer.Option(
        True,
        help="Divide each completion's summed log-prob by its token count "
        "(mean per-token log-prob). True is the current default introduced "
        "to fix DPO mode collapse; pass --no-length-norm to ablate it and "
        "recover the raw summed-log-prob ALLD objective.",
    ),
    lr: float = typer.Option(5e-6, help="Learning rate."),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation."),
    bf16: bool = typer.Option(False, help="Use bf16."),
    fp16: bool = typer.Option(False, help="Use fp16."),
    max_samples: Optional[int] = typer.Option(None, help="Limit dataset size."),
    dims_source_json: Optional[Path] = typer.Option(
        None,
        help=(
            "Caption JSONL (e.g. train_nisqa_llama_10k.json) to join noi/col/loud "
            "from, by degraded-filename basename. Required for TEMPORAL DPO "
            "(temporal records lack these dims); the temporal reference prompt "
            "needs the full mos/noi/col/loud palette plus the interval. Ignored "
            "for MOS records."
        ),
    ),
    use_discontinuity: bool = typer.Option(
        False,
        help=(
            "Discontinuity-deviation ablation (global ALLD only): build the MOS "
            "reference prompt from 5 scores incl. `dis` instead of 4. Requires "
            "each record to carry a `dis` field (see generate_dpo_data.py "
            "--add-discontinuity-from). Default False = paper-faithful 4-dim."
        ),
    ),
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
    save_intermediate: bool = typer.Option(
        False,
        help="Save intermediate checkpoints to LOCAL disk every save_steps, "
        "without Hub push. Use to capture the collapse-onset curve in a "
        "diagnostic run. save_total_limit still bounds /work3 usage.",
    ),
    final_save_only: bool = typer.Option(
        False,
        help="Skip ALL intermediate checkpoints (save_strategy='no') and write "
        "exactly one model at the end, then push it to the Hub. Use with "
        "--hub-model-id when you want a single final model on the Hub and no "
        "step-checkpoints. The end save is wrapped in OSError -> Hub-rescue, and "
        "save_only_model keeps it small, so it avoids the classic hang/overflow "
        "of the naive single-save path while honouring 'save just one'.",
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
                "length_norm": length_norm,
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
        dims_source_json=dims_source_json,
        use_discontinuity=use_discontinuity,
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
    # TimeAudio checkpoints carry an extra abs_time_embedding param and a
    # resized vocab/lm_head; the stock class SILENTLY drops abs_time_embedding
    # (loads with a warning, no crash), which would strip mechanism 2 from a
    # TimeAudio policy. Detect the subclass from the config the same way
    # asa.inference.load_model does. Stock SFT checkpoints have neither flag and
    # load through the stock class unchanged.
    try:
        policy_cfg = AutoConfig.from_pretrained(model_id)
        is_timeaudio = bool(
            getattr(policy_cfg, "use_abs_time_embedding", False)
            or getattr(policy_cfg, "use_time_tokens", False)
        )
    except Exception:
        is_timeaudio = False
    policy_cls = (
        Qwen2AudioTimeForConditionalGeneration
        if is_timeaudio
        else Qwen2AudioForConditionalGeneration
    )
    if is_main:
        print(
            f"Loading Policy Model (Audio): {model_id} (dtype={dtype}, "
            f"class={policy_cls.__name__})"
        )
    model = policy_cls.from_pretrained(model_id, torch_dtype=dtype)

    # 2. Load Reference Model (Text)
    if is_main:
        print(f"Loading Reference Model (Text): {ref_model_id} (dtype={dtype})")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, torch_dtype=dtype)

    push_to_hub = hub_model_id is not None
    # Save intermediate checkpoints whenever Hub streaming is on OR the
    # diagnostic --save-intermediate flag is set. The flag keeps the saves
    # local (no Hub), so a quota-safe collapse-onset run is possible without
    # an HF account that has private-repo storage.
    # --final-save-only overrides both: no step-checkpoints at all, just the
    # single explicit end save+push below (save_only_model keeps it ~16 GB and
    # the OSError->Hub-rescue guards the quota-overflow-at-save failure mode).
    save_steps_strategy = (push_to_hub or save_intermediate) and not final_save_only
    if final_save_only and is_main:
        dest = hub_model_id if push_to_hub else str(output_dir)
        print(
            f"Final-save-only ENABLED: no intermediate checkpoints; one model "
            f"saved at the end and "
            f"{'pushed to ' + hub_model_id if push_to_hub else 'kept local at ' + str(output_dir)}."
        )
    elif push_to_hub and is_main:
        print(
            f"Hub streaming ENABLED: pushing checkpoints to {hub_model_id} every {save_steps} steps "
            f"(local rotation: keep last {save_total_limit})."
        )
    elif save_intermediate and is_main:
        print(
            f"Local intermediate checkpoints ENABLED: saving to disk every {save_steps} steps "
            f"(rotation: keep last {save_total_limit}), no Hub push."
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
        save_strategy="steps" if save_steps_strategy else "no",
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
        # every_save streams each step-checkpoint; with final_save_only there are
        # no step-saves, so "end" is correct (the explicit end push does the work).
        hub_strategy="end" if (final_save_only or not push_to_hub) else "every_save",
        hub_private_repo=hub_private,
    )

    trainer = ALLDDPOTrainer(
        ref_model=ref_model,
        beta=beta,
        length_norm=length_norm,
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
