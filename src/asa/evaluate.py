import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import typer
import nltk
from nltk.translate.bleu_score import sentence_bleu

from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Evaluate fine-tuned Qwen2-Audio models on standard datasets.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model_path: Optional[str] = typer.Option(
        None, help="Hub repo ID or local checkpoint path (global default)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, help="Global output directory for evaluation results."
    ),
    data_root: Optional[Path] = typer.Option(
        None, help="Global data root to resolve audio paths."
    ),
    dataset_paths: Optional[List[Path]] = typer.Option(
        None, "--dataset-path", help="Paths to the test JSONL datasets (global)."
    ),
    batch_size: int = typer.Option(
        4, "--batch-size", help="Global inference batch size."
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", help="Global max samples to evaluate (for testing)."
    ),
):
    """Top-level callback to accept global options that subcommands can inherit."""
    ctx.ensure_object(dict)
    if model_path is not None:
        ctx.obj["model_path"] = model_path
    if output_dir is not None:
        ctx.obj["output_dir"] = output_dir
    if data_root is not None:
        ctx.obj["data_root"] = data_root
    if dataset_paths is not None:
        ctx.obj["dataset_paths"] = dataset_paths
    ctx.obj["batch_size"] = batch_size
    ctx.obj["max_samples"] = max_samples

    # If user passed dataset paths at top-level and didn't invoke a subcommand,
    # run the default `eval-mos` command for convenience (keeps backwards-compatibility
    # with scripts that passed options at the top level).
    if ctx.invoked_subcommand is None:
        if ctx.obj.get("dataset_paths"):
            # Call eval_mos with the collected globals; subcommand flags override.
            eval_mos(
                ctx,
                dataset_paths=ctx.obj.get("dataset_paths"),
                model_path=ctx.obj.get("model_path", None),
                data_root=ctx.obj.get("data_root", Path("data")),
                max_samples=ctx.obj.get("max_samples", None),
                output_dir=ctx.obj.get("output_dir", None),
                batch_size=ctx.obj.get("batch_size", 4),
            )
        else:
            # No subcommand and no dataset paths: show help
            print("No subcommand provided. Use --help to list available commands.")


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
def eval_mos(
    ctx: typer.Context,
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Paths to the test JSONL datasets."
    ),
    model_path: Optional[str] = typer.Option(
        None, help="Hub repo ID or local checkpoint path (overrides global)."
    ),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory that contains the raw audio tree."
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to evaluate (for testing)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Dir to save results. Defaults to results/evaluation/<model_name>_mos.",
    ),
    batch_size: int = typer.Option(4, help="Inference batch size."),
):
    """Run model inference and evaluate quality based on MOS and BLEU."""

    # Ensure NLTK punkt is available for tokenization, fallback to simple split if not
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Resolve model_path: prefer the command option, then the global, then default
    if model_path is None:
        model_path = ctx.obj.get("model_path", None)
    if model_path is None:
        model_path = ASAModel.SFT

    # Resolve output_dir: prefer command option, then global, then default
    if output_dir is None:
        output_dir = ctx.obj.get("output_dir", None)
    if output_dir is None:
        model_name = Path(model_path).name
        output_dir = Path(f"results/evaluation/{model_name}_mos")

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
            # Assuming format like `/workspace/data/nisqa/NISQA_Corpus/...`
            raw_path = item["audios"][0]
            # Replace the absolute reference to the current repository structure
            if "/workspace/data/nisqa/" in raw_path:
                resolved_path = raw_path.replace("/workspace/data/nisqa/", "data/raw/")
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

        predicted_responses = [r["predicted_response"] for r in results]
        unique_predictions = len(set(predicted_responses))
        top_prediction_frequency = max(Counter(predicted_responses).values()) / max(
            len(predicted_responses), 1
        )

        logging.info("=" * 40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated: {len(data)}")
        logging.info(f"MOS MAE (Mean Absolute Error): {mae:.4f}")
        logging.info(f"MOS MSE (Mean Squared Error):  {mse:.4f}")
        logging.info(f"Average BLEU Score:            {avg_bleu:.4f}")
        logging.info(
            f"Unique predictions: {unique_predictions} | Top prediction frequency: {top_prediction_frequency:.4f}"
        )
        logging.info("=" * 40)

        dataset_name = dataset_path.stem
        out_file = output_dir / f"{dataset_name}_results.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "samples": len(data),
                        "mae": mae,
                        "mse": mse,
                        "bleu": avg_bleu,
                        "unique_predictions": unique_predictions,
                        "top_prediction_frequency": top_prediction_frequency,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logging.info(f"Saved detailed results to {out_file}\n")


def extract_winner(text: str) -> str:
    """Extract the predicted winner (A / B / Tie) from an A/B response."""
    text_lower = text.lower()

    pref = re.search(
        r"\b(?:select|prefer|choose|better|winner)\b.{0,30}speech\s*([ab])\b",
        text_lower,
    )
    if pref:
        return pref.group(1).upper()

    shortpref = re.search(
        r"\b(?:select|prefer|choose|better|winner)\b.{0,10}\b([ab])\b",
        text_lower,
    )
    if shortpref:
        return shortpref.group(1).upper()

    if re.search(r"\b(tie|draw|equal|same quality)\b", text_lower):
        return "Tie"

    all_labels = re.findall(r"\bspeech\s*([ab])\b", text_lower)
    if all_labels:
        return all_labels[-1].upper()

    last = re.findall(r"\b([ab])\b", text_lower)
    if last:
        return last[-1].upper()

    return "Tie"


@app.command()
def eval_ab(
    ctx: typer.Context,
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Paths to A/B test JSONL datasets."
    ),
    model_path: Optional[str] = typer.Option(
        None, help="Path or Hub ID to the model checkpoint (overrides global)."
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to evaluate (for testing)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Dir to save results. Defaults to results/evaluation/<model_name>_ab.",
    ),
    batch_size: int = typer.Option(4, help="Inference batch size."),
):
    """Run model inference and evaluate accuracy and BLEU for A/B testing."""

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Resolve model_path: prefer the command option, then the global, then default
    if model_path is None:
        model_path = ctx.obj.get("model_path", None)
    if model_path is None:
        model_path = ASAModel.SFT_AB

    # Resolve output_dir: prefer command option, then global, then default
    if output_dir is None:
        output_dir = ctx.obj.get("output_dir", None)
    if output_dir is None:
        model_name = Path(model_path).name
        output_dir = Path(f"results/evaluation/{model_name}_ab")

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
            raw_path_a = item["audios"][0]
            raw_path_b = item["audios"][1]
            if "/workspace/data/nisqa/" in raw_path_a:
                raw_path_a = raw_path_a.replace("/workspace/data/nisqa/", "data/raw/")
                raw_path_b = raw_path_b.replace("/workspace/data/nisqa/", "data/raw/")
            audio_paths.append((raw_path_a, raw_path_b))

        logging.info("Running A/B inference...")
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            ab_mode=True,
            device=device,
            batch_size=batch_size,
        )

        logging.info("Calculating metrics...")
        results = []
        bleu_scores = []
        correct = 0

        classes = {
            "A": {"tp": 0, "total": 0},
            "B": {"tp": 0, "total": 0},
            "Tie": {"tp": 0, "total": 0},
        }

        for item, pred in zip(data, predictions):
            pred = pred.strip()
            true_winner = item["winner"]
            pred_winner = extract_winner(pred)

            is_correct = pred_winner == true_winner
            if is_correct:
                correct += 1

            if true_winner in classes:
                classes[true_winner]["total"] += 1
                if is_correct:
                    classes[true_winner]["tp"] += 1

            true_resp = item["response"]

            try:
                ref_tokens = nltk.word_tokenize(true_resp.lower())
                hyp_tokens = nltk.word_tokenize(pred.lower())
                bleu = sentence_bleu([ref_tokens], hyp_tokens)
            except Exception:
                ref_tokens = true_resp.lower().split()
                hyp_tokens = pred.lower().split()
                bleu = sentence_bleu([ref_tokens], hyp_tokens)
            bleu_scores.append(bleu)

            res_item = item.copy()
            res_item["predicted_response"] = pred
            res_item["predicted_winner"] = pred_winner
            res_item["correct"] = is_correct
            res_item["bleu"] = bleu
            results.append(res_item)

        n = len(data)
        accuracy = correct / n if n > 0 else 0
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        logging.info("=" * 40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated: {n}")
        logging.info(f"Accuracy:          {accuracy:.4f} ({correct}/{n})")
        for label, counts in classes.items():
            if counts["total"]:
                acc = counts["tp"] / counts["total"]
                logging.info(
                    f"  Acc ({label}):        {acc:.4f} ({counts['tp']}/{counts['total']})"
                )
        logging.info(f"Average BLEU Score: {avg_bleu:.4f}")
        logging.info("=" * 40)

        dataset_name = dataset_path.stem
        out_file = output_dir / f"{dataset_name}_results.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {
                        "samples": n,
                        "accuracy": accuracy,
                        "per_class": classes,
                        "bleu": avg_bleu,
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
