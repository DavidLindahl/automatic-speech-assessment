"""
train.py — Unified training script for Qwen2-Audio (SFT & DPO, MOS & A/B testing).

Designed for multi-GPU training with DeepSpeed on HPC clusters.
Consolidates previous separate scripts into a structured factory pattern.
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

from asa.datasets import SFTDataset, SFTDatasetAB, DPODataset, DPODatasetAB
from asa.collators import (
    Qwen2AudioCollator,
    Qwen2AudioCollatorAB,
    ALLDDPOCollator,
    ALLDDPOCollatorAB,
)

app = typer.Typer()


class ALLDDPOTrainer(Trainer):
    """Custom Trainer implementing the ALLD cross-modal DPO loss."""

    def __init__(self, ref_model, beta=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(
                ref_model, evaluation_mode=True
            )
        else:
            self.ref_model = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        policy_inputs = {
            "input_ids": inputs["policy_input_ids"],
            "attention_mask": inputs["policy_attention_mask"],
            "labels": inputs["policy_labels"],
        }
        if "policy_audio_values" in inputs and inputs["policy_audio_values"] is not None:
            policy_inputs["audio_values"] = inputs["policy_audio_values"]
        if "policy_audio_features" in inputs and inputs["policy_audio_features"] is not None:
            policy_inputs["audio_features"] = inputs["policy_audio_features"]

        ref_inputs = {
            "input_ids": inputs["ref_input_ids"],
            "attention_mask": inputs["ref_attention_mask"],
            "labels": inputs["ref_labels"],
        }

        batch_size = policy_inputs["input_ids"].shape[0] // 2
        outputs = model(**policy_inputs)
        policy_logits = outputs.logits

        with torch.no_grad():
            ref_outputs = self.ref_model(**ref_inputs)
            ref_logits = ref_outputs.logits

        def get_logprobs(logits, labels):
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            loss_mask = labels != -100
            per_token_logprobs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(labels.shape)
            return (per_token_logprobs * loss_mask).sum(dim=1)

        policy_logprobs = get_logprobs(policy_logits, policy_inputs["labels"])
        ref_logprobs = get_logprobs(ref_logits, ref_inputs["labels"])

        policy_chosen = policy_logprobs[:batch_size]
        policy_rejected = policy_logprobs[batch_size:]
        ref_chosen = ref_logprobs[:batch_size]
        ref_rejected = ref_logprobs[batch_size:]

        logits_diff = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        loss = -torch.nn.functional.logsigmoid(self.beta * logits_diff).mean()

        if self.state.is_world_process_zero:
            chosen_rewards = self.beta * (policy_chosen - ref_chosen).detach()
            rejected_rewards = self.beta * (policy_rejected - ref_rejected).detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            self.log(
                {
                    "rewards/chosen": chosen_rewards.mean().item(),
                    "rewards/rejected": rejected_rewards.mean().item(),
                    "rewards/accuracies": reward_accuracies.mean().item(),
                    "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
                }
            )

        if return_outputs:
            return loss, outputs
        return loss


@app.command()
def train(
    method: str = typer.Option(
        "sft", help="Training method to use: 'sft' or 'dpo'."
    ),
    mode: str = typer.Option(
        "mos", help="Data mode to use: 'mos' (standard) or 'ab' (preference)."
    ),
    model_id: str = typer.Option(
        "Qwen/Qwen2-Audio-7B", help="HuggingFace model ID."
    ),
    ref_model_id: str = typer.Option(
        "Qwen/Qwen2-7B-Instruct", help="Path to Expert Text model (Required for DPO)."
    ),
    json_path: Path = typer.Option(
        Path("data/processed/train_nisqa_llama_10k.json"), help="Path to JSONL data."
    ),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory for raw audios."
    ),
    model_name: str = typer.Option(
        ..., help="Name of the model to save."
    ),
    batch_size: int = typer.Option(4, help="Per-device batch size."),
    epochs: int = typer.Option(2, help="Number of epochs."),
    beta: float = typer.Option(0.4, help="DPO margin parameter (Required for DPO)."),
    lr: float = typer.Option(1e-5, help="Learning rate."),
    gradient_accumulation_steps: int = typer.Option(4, help="Grad accumulation steps."),
    bf16: bool = typer.Option(False, help="Use bf16."),
    fp16: bool = typer.Option(False, help="Use fp16."),
    max_samples: Optional[int] = typer.Option(None, help="Limit dataset size."),
    deepspeed: Optional[str] = typer.Option(None, help="DeepSpeed config JSON."),
    val_split: float = typer.Option(0.05, help="Validation fraction (0 to disable)."),
    eval_steps: int = typer.Option(500, help="Eval interval."),
    wandb_entity: Optional[str] = typer.Option("speech-quality-DTU-bachelor", help="W&B entity."),
    wandb_project: Optional[str] = typer.Option("qwen2-audio-finetune", help="W&B project."),
    wandb_run_name: Optional[str] = typer.Option(None, help="W&B run name."),
):
    """Unified entrypoint for tracking and running SFT/DPO pipelines over MOS/AB data."""
    method = method.lower()
    mode = mode.lower()
    if method not in ["sft", "dpo"]:
        raise ValueError(f"Unknown method {method}. Must be 'sft' or 'dpo'")
    if mode not in ["mos", "ab"]:
        raise ValueError(f"Unknown mode {mode}. Must be 'mos' or 'ab'")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if wandb_project and is_main:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                "method": method,
                "mode": mode,
                "model_id": model_id,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "val_split": val_split,
                "max_samples": max_samples,
                "deepspeed": deepspeed is not None,
            },
        )
    report_to = "wandb" if wandb_project else "none"

    output_dir = Path("models") / model_name
    if is_main:
        print(f"Model will be saved to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    if is_main:
        print(f"Loading processor: {model_id}")
    audio_processor = AutoProcessor.from_pretrained(model_id)

    text_tokenizer = None
    if method == "dpo":
        if is_main:
            print(f"Loading Reference tokenizer: {ref_model_id}")
        text_tokenizer = AutoTokenizer.from_pretrained(ref_model_id)

    # Factory Selection Maps
    dataset_cls = {
        "sft": {"mos": SFTDataset, "ab": SFTDatasetAB},
        "dpo": {"mos": DPODataset, "ab": DPODatasetAB},
    }[method][mode]

    full_dataset = dataset_cls(
        json_path=json_path,
        data_root=data_root,
        max_samples=max_samples,
    )

    if val_split > 0 and method == "sft":
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = full_dataset
        val_dataset = None

    if method == "sft":
        collator_cls = {"mos": Qwen2AudioCollator, "ab": Qwen2AudioCollatorAB}[mode]
        collator = collator_cls(processor=audio_processor)
    else:
        collator_cls = {"mos": ALLDDPOCollator, "ab": ALLDDPOCollatorAB}[mode]
        collator = collator_cls(audio_processor=audio_processor, text_tokenizer=text_tokenizer)

    if is_main:
        print(f"Loading Policy Model (Audio): {model_id} (dtype={dtype})")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=dtype)

    ref_model = None
    if method == "dpo":
        if is_main:
            print(f"Loading Reference Model (Text): {ref_model_id} (dtype={dtype})")
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_id, dtype=dtype)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
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
    )

    if method == "sft":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            processing_class=audio_processor,  # Required for proper collator internal handling
        )
    else:
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
        print(f"Starting {method.upper()} {mode.upper()} training...")
    trainer.train()

    if is_main:
        print(f"Saving {method.upper()} model to {output_dir}")
    trainer.save_model(str(output_dir))
    audio_processor.save_pretrained(str(output_dir))

    if wandb_project and is_main:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    app()
