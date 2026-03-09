import logging
from pathlib import Path
from typing import Optional

import typer

from asa.inference import load_model, run_inference
from asa.processed_data import (
    load_processed_records,
    resolve_audio_path,
    write_processed_records,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = typer.Typer(
    help="Generate DPO dataset by running inference on the SFT warmup model."
)


@app.command()
def generate(
    input_json: Path = typer.Option(
        Path("data/processed/train_nisqa_llama_10k.json"), help="Input dataset."
    ),
    output_json: Path = typer.Option(
        Path("data/processed/train_dpo_10k.json"), help="Output DPO dataset."
    ),
    model_path: Path = typer.Option(
        Path("models/sft_warmup"), help="Path to the trained model."
    ),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory that contains the raw audio tree."
    ),
    batch_size: int = typer.Option(8, help="Inference batch size."),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to process (for debugging)."
    ),
):
    """Run inference to generate 'rejected' responses and create a DPO dataset."""

    output_json.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading input data from {input_json}")
    data = load_processed_records(input_json)

    if max_samples:
        data = data[:max_samples]
        logging.info(f"Limited generation to {max_samples} samples.")

    logging.info(f"Loading model from {model_path}")
    processor, model, device = load_model(model_path)

    audio_paths = []
    for item in data:
        audio_paths.append(str(resolve_audio_path(item["audios"][0], data_root)))

    logging.info(
        f"Running inference on {len(audio_paths)} samples with batch size {batch_size}..."
    )
    predictions = run_inference(
        model=model,
        processor=processor,
        audio_paths=audio_paths,
        device=device,
        batch_size=batch_size,
    )

    logging.info("Formatting DPO dataset...")
    dpo_data = []
    for item, pred in zip(data, predictions):
        dpo_item = item.copy()

        # Standardize for DPO
        dpo_item["chosen"] = item["response"]
        dpo_item["rejected"] = pred.strip()

        dpo_data.append(dpo_item)

    logging.info(f"Saving DPO dataset to {output_json}")
    write_processed_records(output_json, dpo_data)

    logging.info("Done.")


if __name__ == "__main__":
    app()
