import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import typer
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu

from asa.inference import load_model, run_inference
from asa.processed_data import load_processed_records, resolve_audio_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Evaluate fine-tuned Qwen2-Audio models on standard datasets.")

def extract_mos(text: str) -> float:
    """Extract numeric MOS score from generated text."""
    # Look for explicit MOS mentions e.g., "MOS of 4.3" or "MOS score is 4.3"
    match = re.search(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Fallback to the last float/number found in the text
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if matches:
        return float(matches[-1])
    return 0.0

@app.command()
def evaluate(
    dataset_paths: List[Path] = typer.Option(..., "--dataset-path", help="Paths to the test JSONL datasets."),
    model_path: Path = typer.Option(..., help="Path to the model checkpoint dir."),
    data_root: Path = typer.Option(Path("data"), help="Root directory that contains the raw audio tree."),
    max_samples: Optional[int] = typer.Option(None, help="Max samples to evaluate (for testing)."),
    output_dir: Path = typer.Option(Path("results/evaluation/sft_warm_eval"), help="Dir to save results."),
    batch_size: int = typer.Option(4, help="Inference batch size.")
):
    """Run model inference and evaluate quality based on MOS and BLEU."""
    
    # Ensure NLTK punkt is available for tokenization, fallback to simple split if not
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading model from {model_path}...")
    processor, model, device = load_model(model_path)
    
    for dataset_path in dataset_paths:
        logging.info(f"Loading dataset from {dataset_path}")
        data = load_processed_records(dataset_path)
                
        if max_samples:
            data = data[:max_samples]
            logging.info(f"Limited evaluation to {max_samples} samples.")
            
        # Process audio paths securely
        audio_paths = []
        for item in data:
            audio_paths.append(str(resolve_audio_path(item["audios"][0], data_root)))
            
        logging.info("Running inference...")
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            device=device,
            batch_size=batch_size
        )
        
        logging.info("Calculating metrics...")
        results = []
        mos_errors = []
        bleu_scores = []
        
        for i, (item, pred) in enumerate(zip(data, predictions)):
            pred = pred.strip()
            true_mos = float(item["mos"])
            pred_mos = extract_mos(pred)
            
            true_resp = item["response"]
            
            # Calculate Errors
            error = abs(true_mos - pred_mos)
            mos_errors.append(error)
            
            # Calculate BLEU (simple word-level)
            try:
                ref_tokens = nltk.word_tokenize(true_resp.lower())
                hyp_tokens = nltk.word_tokenize(pred.lower())
                bleu = sentence_bleu([ref_tokens], hyp_tokens)
            except Exception:
                ref_tokens = true_resp.lower().split()
                hyp_tokens = pred.lower().split()
                bleu = sentence_bleu([ref_tokens], hyp_tokens)
            bleu_scores.append(bleu)
            
            # Keep track of results saving
            res_item = item.copy()
            res_item["predicted_response"] = pred
            res_item["predicted_mos"] = pred_mos
            res_item["mos_error"] = error
            res_item["bleu"] = bleu
            results.append(res_item)
            
        # Aggregate Metrics
        mae = sum(mos_errors) / len(mos_errors)
        mse = sum(e**2 for e in mos_errors) / len(mos_errors)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        
        logging.info("="*40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated: {len(data)}")
        logging.info(f"MOS MAE (Mean Absolute Error): {mae:.4f}")
        logging.info(f"MOS MSE (Mean Squared Error):  {mse:.4f}")
        logging.info(f"Average BLEU Score:            {avg_bleu:.4f}")
        logging.info("="*40)
        
        dataset_name = dataset_path.stem
        out_file = output_dir / f"{dataset_name}_results.json"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": {
                    "samples": len(data),
                    "mae": mae,
                    "mse": mse,
                    "bleu": avg_bleu
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Saved detailed results to {out_file}\n")

if __name__ == "__main__":
    app()
