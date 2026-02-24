"""
Supervised fine-tuning script for Qwen2Audio using TRL's SFTTrainer.
Fine-tunes the model using Full Fine-Tuning on the Hub dataset.
Designed to be run on HPC clusters using multi-GPU (e.g. Accelerate/Deepspeed).
"""

from pathlib import Path
import torch
import transformers
import typer
from transformers import AutoProcessor
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Configure typer
app = typer.Typer()


def format_qwen_chat(example, processor):
    """
    Minimal multi-modal mapping function.
    Reads the 'messages' array and applies the Qwen2-Audio chat template.
    Returns the processed 'text' alongside the already-casted 'audios'.
    """
    text = processor.apply_chat_template(
        example["messages"], add_generation_prompt=False, tokenize=False
    )
    return {"text": text}


@app.command()
def main(
    model_id: str = typer.Option(
        "Qwen/Qwen2-Audio-7B",
        help="The Hugging Face model ID to fine-tune.",
    ),
    dataset_type: str = typer.Option(
        "mos",
        help="Which dataset to fine-tune on: 'mos' (train_nisqa_llama_10k.parquet) or 'abtest' (train_nisqa_abtest_llama_10k.parquet).",
    ),
    output_dir: Path = typer.Option(
        Path("results"), help="Directory to save the trained model."
    ),
    batch_size: int = typer.Option(4, help="Batch size per device during training."),
    epochs: int = typer.Option(2, help="Total number of training epochs to perform."),
    lr: float = typer.Option(1e-5, help="Learning rate for the optimizer."),
    gradient_accumulation_steps: int = typer.Option(
        4, help="Number of update steps to accumulate before a backward pass."
    ),
    bf16: bool = typer.Option(
        True,
        help="Use bf16 mixed precision (recommended for A100/H100 HPC environments).",
    ),
    deepspeed: str = typer.Option(
        "default-zero2",
        help="Path to deepspeed config JSON file or 'default-zero2' / 'default-zero3'.",
    ),
):
    """
    Runs Full SFT to fine-tune Qwen2Audio.
    """
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    if dataset_type == "mos":
        dataset_path = Path("data/processed/train_nisqa_llama_10k.parquet")
    elif dataset_type == "abtest":
        dataset_path = Path("data/processed/train_nisqa_abtest_llama_10k.parquet")
    else:
        print(f"Unknown dataset type: {dataset_type}. Choose 'mos' or 'abtest'.")
        raise typer.Exit(code=1)

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Please run preprocessing first.")
        raise typer.Exit(code=1)

    print(f"Loading dataset from: {dataset_path}")
    # Load dataset natively (this handles audio binary loading securely behind the scenes)
    dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")

    # Pre-format conversations
    dataset = dataset.map(
        format_qwen_chat,
        fn_kwargs={"processor": processor},
        desc="Applying chat template",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        # SFTTrainer will automatically route "text" and "audios" through the DataCollatorForLanguageModeling
        dataset_text_field="text",
        remove_unused_columns=False,  # Keep the 'audios' column natively supported by transformers
    )

    print(f"Loading model: {model_id} for Full Fine-Tuning")
    torch_dtype = torch.bfloat16 if bf16 else torch.float32

    # Full fine-tuning (no PEFT/LoRA)
    model = transformers.Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    from typing import Dict, List, Any

    class Qwen2AudioDataCollator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Expects "text" and "audios" columns to be kept in dataset
            texts = [feature["text"] for feature in features]

            audios = []
            for feature in features:
                if "audios" in feature and feature["audios"] is not None:
                    for audio in feature["audios"]:
                        # Extract the raw array from the Hugging Face Audio feature dict
                        audios.append(audio["array"])

            if len(audios) == 0:
                audios = None

            batch = self.processor(
                text=texts,
                audio=audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )

            # Create labels for causal language modeling
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels

            return batch

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=Qwen2AudioDataCollator(processor),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    app()
