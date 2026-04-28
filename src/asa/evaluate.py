import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import sacrebleu
import typer

from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records

EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.9
EVAL_MAX_NEW_TOKENS = 150

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
                do_sample=True,
                temperature=EVAL_TEMPERATURE,
                top_p=EVAL_TOP_P,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
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
    do_sample: bool = typer.Option(
        True,
        "--do-sample/--greedy",
        help="Sample with temperature/top_p (default) or greedy decoding.",
    ),
    temperature: float = typer.Option(
        EVAL_TEMPERATURE, help="Sampling temperature (only used with --do-sample)."
    ),
    top_p: float = typer.Option(
        EVAL_TOP_P, help="Nucleus top-p (only used with --do-sample)."
    ),
    max_new_tokens: int = typer.Option(
        EVAL_MAX_NEW_TOKENS, help="Max new tokens to generate per sample."
    ),
):
    """Run model inference and evaluate quality based on MOS and BLEU."""

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

        logging.info(
            "Running inference (do_sample=%s, temperature=%.2f, top_p=%.2f, "
            "max_new_tokens=%d)...",
            do_sample,
            temperature,
            top_p,
            max_new_tokens,
        )
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            device=device,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        logging.info("Calculating metrics...")
        results = []
        mos_errors = []
        hyps: List[str] = []
        refs: List[str] = []

        for item, pred in zip(data, predictions):
            pred = pred.strip()
            true_mos = float(item["mos"])
            pred_mos = extract_mos(pred)
            true_resp = item["response"]

            error = abs(true_mos - pred_mos)
            mos_errors.append(error)

            hyps.append(pred)
            refs.append(true_resp)

            res_item = item.copy()
            res_item["predicted_response"] = pred
            res_item["predicted_mos"] = pred_mos
            res_item["mos_error"] = error
            results.append(res_item)

        mae = sum(mos_errors) / len(mos_errors)
        mse = sum(e**2 for e in mos_errors) / len(mos_errors)

        bleu_corpus = sacrebleu.corpus_bleu(hyps, [refs]).score
        bleu_corpus_lc = sacrebleu.corpus_bleu(
            [h.lower() for h in hyps], [[r.lower() for r in refs]]
        ).score

        unique_predictions = len(set(hyps))
        top_prediction_frequency = max(Counter(hyps).values()) / max(len(hyps), 1)

        logging.info("=" * 40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated:                    {len(data)}")
        logging.info(f"MOS MAE (Mean Absolute Error):        {mae:.4f}")
        logging.info(f"MOS MSE (Mean Squared Error):         {mse:.4f}")
        logging.info(f"BLEU (sacrebleu corpus, cased):       {bleu_corpus:.2f}")
        logging.info(f"BLEU (sacrebleu corpus, lowercased):  {bleu_corpus_lc:.2f}")
        logging.info(
            f"Unique predictions: {unique_predictions} / {len(hyps)} "
            f"| Top prediction frequency: {top_prediction_frequency:.4f}"
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
                        "bleu": bleu_corpus,
                        "bleu_lowercased": bleu_corpus_lc,
                        "unique_predictions": unique_predictions,
                        "top_prediction_frequency": top_prediction_frequency,
                    },
                    "decoding": {
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
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
