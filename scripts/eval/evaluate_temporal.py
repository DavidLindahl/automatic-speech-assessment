import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import typer

from asa.inference import ASAModel, load_model, run_inference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Evaluate fine-tuned Qwen2-Audio models on Temporal-ALLD.")


def extract_timestamps(text: str):
    """Extract start and end timestamps from text containing <|XX.XX|> tokens."""
    matches = re.findall(r"<\|(\d+(?:\.\d+)?)\|>", text)
    if len(matches) >= 2:
        start_time = float(matches[0])
        end_time = float(matches[-1])
        if start_time < end_time:
            return start_time, end_time
    # Fallback to general floats if tokens are malformed
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if len(matches) >= 2:
        return float(matches[0]), float(matches[-1])
    return 0.0, 0.0


def calculate_tiou(
    pred_start: float, pred_end: float, true_start: float, true_end: float
) -> float:
    """Calculate Temporal Intersection over Union (t-IoU)."""
    if pred_end <= pred_start or true_end <= true_start:
        return 0.0

    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)

    if intersection_start >= intersection_end:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (true_end - true_start) - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def extract_noise_type(text: str) -> str:
    """Extract noise type from the generated text."""
    # Simple extraction Strategy: look for 'interrupted by a (.*?) occurring'
    match = re.search(r"interrupted by an? (.*?) occurring", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "unknown"


@app.command()
def eval_temporal(
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Paths to the test JSONL datasets."
    ),
    model_path: str = typer.Option(
        ASAModel.SFT, help="Hub repo ID or local checkpoint path."
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to evaluate (for testing)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Dir to save results. Defaults to results/evaluation/<model_name>_temporal.",
    ),
    batch_size: int = typer.Option(4, help="Inference batch size."),
):
    """Run model inference and evaluate quality based on t-IoU and Noise Detection."""

    if output_dir is None:
        model_name = Path(model_path).name
        if not model_name:
            model_name = "model"
        output_dir = Path(f"results/evaluation/{model_name}_temporal")

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading model from {model_path}...")
    processor, model, device = load_model(model_path)

    for dataset_path in dataset_paths:
        logging.info(f"Loading dataset from {dataset_path}")
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data.append(json.loads(line))

        if max_samples:
            data = data[:max_samples]
            logging.info(f"Limited evaluation to {max_samples} samples.")

        audio_paths = []
        for item in data:
            raw_path = item["audios"][0]
            if "/workspace/data/" in raw_path:
                resolved_path = raw_path.replace("/workspace/data/", "data/raw/")
            else:
                resolved_path = raw_path
            audio_paths.append(resolved_path)

        logging.info("Running inference...")
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            device=device,
            batch_size=batch_size,
        )

        logging.info("Calculating metrics...")
        results = []
        tiou_scores = []
        correct_detection = 0

        for i, (item, pred) in enumerate(zip(data, predictions)):
            pred = pred.strip()
            true_resp = item["response"]

            # Ground truth extraction
            true_start, true_end = extract_timestamps(true_resp)
            true_noise = extract_noise_type(true_resp)

            # Prediction extraction
            pred_start, pred_end = extract_timestamps(pred)
            pred_noise = extract_noise_type(pred)

            # Metrics
            tiou_val = calculate_tiou(pred_start, pred_end, true_start, true_end)
            tiou_scores.append(tiou_val)

            # Exact noise match or substring
            is_correct = (true_noise.lower() in pred_noise.lower()) or (
                pred_noise.lower() in true_noise.lower() and len(pred_noise) > 3
            )
            if is_correct:
                correct_detection += 1

            res_item = item.copy()
            res_item["predicted_response"] = pred
            res_item["predicted_start"] = pred_start
            res_item["predicted_end"] = pred_end
            res_item["predicted_noise"] = pred_noise
            res_item["tiou"] = tiou_val
            res_item["correct_noise"] = is_correct
            results.append(res_item)

        avg_tiou = sum(tiou_scores) / len(tiou_scores) if tiou_scores else 0.0
        n = len(data)
        accuracy = correct_detection / n if n > 0 else 0.0

        logging.info("=" * 40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated: {n}")
        logging.info(f"Average t-IoU:     {avg_tiou:.4f}")
        logging.info(f"Noise Detection Acc: {accuracy:.4f} ({correct_detection}/{n})")
        logging.info("=" * 40)

        dataset_name = dataset_path.stem
        out_file = output_dir / f"{dataset_name}_results.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "samples": n,
                        "avg_tiou": avg_tiou,
                        "detection_accuracy": accuracy,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logging.info(f"Saved detailed results to {out_file}\n")


if __name__ == "__main__":
    app()
