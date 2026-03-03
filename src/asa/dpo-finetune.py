"""
dpo-finetune.py — Direct Preference Optimization script for Qwen2-Audio.

Uses a custom DPOLoss implementation inside Trainer to naturally handle
multi-modal Qwen2-Audio batches without treading onto complex edge cases in TRL default logic.
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

from asa.data import Qwen2AudioDPOCollator, DPODataset

app = typer.Typer()


class Qwen2AudioDPOTrainer(Trainer):
    """Custom Trainer implementing standard DPO loss for 2N pre-concatenated batches."""
    def __init__(self, ref_model, beta=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        if self.ref_model is not None:
            self.ref_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs are already concatenated (chosen first half, rejected second half)
        batch_size = inputs["input_ids"].shape[0] // 2
        
        # 1. Forward pass policy model
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 2. Forward pass reference model (no gradients)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits
            
        def get_logprobs(logits, labels):
            # Shift to token prediction
            logits = logits[:, :-1, :]
            labels = labels[:, 1:]
            
            loss_mask = labels != -100
            per_token_logprobs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none"
            ).view(labels.shape)
            
            return (per_token_logprobs * loss_mask).sum(dim=1)
            
        policy_logprobs = get_logprobs(logits, inputs["labels"])
        ref_logprobs = get_logprobs(ref_logits, inputs["labels"])
        
        policy_chosen = policy_logprobs[:batch_size]
        policy_rejected = policy_logprobs[batch_size:]
        ref_chosen = ref_logprobs[:batch_size]
        ref_rejected = ref_logprobs[batch_size:]
        
        logits_diff = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        loss = -torch.nn.functional.logsigmoid(self.beta * logits_diff).mean()
        
        # Extract logging metrics (only on master to save compute optionally, but this is fine)
        if self.state.is_world_process_zero:
            chosen_rewards = self.beta * (policy_chosen - ref_chosen).detach()
            rejected_rewards = self.beta * (policy_rejected - ref_rejected).detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            # To log we must record it locally or rely on normal wandb mechanisms
            # Trainer logs loss automatically. Adding custom logs:
            self.log({
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/accuracies": reward_accuracies.mean().item(),
                "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            })

        if return_outputs:
            return loss, outputs
        return loss


@app.command()
def train(
    model_id: str = typer.Option("models/sft_warmup", help="Path to SFT model (policy & reference base)."),
    json_path: Path = typer.Option(Path("data/processed/train_dpo_10k.json"), help="DPO dataset."),
    data_root: Path = typer.Option(Path("data"), help="Root directory for audios."),
    output_dir: Path = typer.Option(Path("models/dpo_final"), help="Save directory."),
    batch_size: int = typer.Option(2, help="Per-device batch size. Note effective memory is 2x due to Chosen/Rejected split."),
    epochs: int = typer.Option(2, help="Training epochs."),
    beta: float = typer.Option(0.4, help="DPO margin parameter beta."),
    lr: float = typer.Option(5e-6, help="Learning rate (usually lower for DPO)."),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation."),
    bf16: bool = typer.Option(False, help="Use bf16."),
    fp16: bool = typer.Option(False, help="Use fp16."),
    max_samples: Optional[int] = typer.Option(None, help="Limit dataset size."),
    deepspeed: Optional[str] = typer.Option(None, help="Path to DeepSpeed config."),
    val_split: float = typer.Option(0.05, help="Validation fraction."),
    eval_steps: int = typer.Option(100, help="Eval interval."),
    wandb_entity: Optional[str] = typer.Option("speech-quality-DTU-bachelor", help="W&B entity."),
    wandb_project: Optional[str] = typer.Option("qwen2-audio-dpo", help="W&B project."),
    wandb_run_name: Optional[str] = typer.Option(None, help="W&B run name."),
):
    """Run Direct Preference Optimization (DPO) on Qwen2-Audio."""
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
                "learning_rate": lr,
                "batch_size": batch_size,
                "beta": beta,
                "epochs": epochs,
                "deepspeed": deepspeed is not None,
            },
        )
    report_to = "wandb" if wandb_project else "none"

    if is_main:
        print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)

    full_dataset = DPODataset(
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

    collator = Qwen2AudioDPOCollator(processor)

    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)

    if is_main:
        print(f"Loading policy model: {model_id} (dtype={dtype})")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    
    if is_main:
        print(f"Loading reference model: {model_id} (dtype={dtype})")
    # For ZeRO-2, DeepSpeed places everything elegantly on GPUs across devices
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, dtype=dtype)

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

    trainer = Qwen2AudioDPOTrainer(
        ref_model=ref_model,
        beta=beta,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    if is_main:
        print("Starting DPO training...")
    trainer.train()

    if is_main:
        print(f"Saving DPO model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    if wandb_project and is_main:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    app()
