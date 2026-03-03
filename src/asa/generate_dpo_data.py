import json
import logging
from pathlib import Path
from typing import Optional

import typer

from asa.inference import load_model, run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="Generate DPO dataset by running inference on the SFT warmup model.")

@app.command()
def generate(
    input_json: Path = typer.Option(Path("data/processed/train_nisqa_llama_10k.json"), help="Input dataset."),
    output_json: Path = typer.Option(Path("data/processed/train_dpo_10k.json"), help="Output DPO dataset."),
    model_path: Path = typer.Option(Path("models/sft_warmup"), help="Path to the trained model."),
    batch_size: int = typer.Option(8, help="Inference batch size."),
    max_samples: Optional[int] = typer.Option(None, help="Max samples to process (for debugging).")
):
    """Run inference to generate 'rejected' responses and create a DPO dataset."""
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading input data from {input_json}")
    data = []
    with open(input_json, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            
    if max_samples:
        data = data[:max_samples]
        logging.info(f"Limited generation to {max_samples} samples.")
        
    logging.info(f"Loading model from {model_path}")
    processor, model, device = load_model(model_path)
    
    audio_paths = []
    for item in data:
        raw_path = item["audios"][0]
        # Same path resolution as evaluate.py
        if "/workspace/data/nisqa/" in raw_path:
            resolved_path = raw_path.replace("/workspace/data/nisqa/", "data/raw/")
        else:
            resolved_path = raw_path
        audio_paths.append(resolved_path)
        
    logging.info(f"Running inference on {len(audio_paths)} samples with batch size {batch_size}...")
    predictions = run_inference(
        model=model,
        processor=processor,
        audio_paths=audio_paths,
        device=device,
        batch_size=batch_size
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
    with open(output_json, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logging.info("Done.")

if __name__ == "__main__":
    app()
