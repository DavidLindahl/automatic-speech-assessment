"""Global MOS-captioning evaluation CLI for Qwen2-Audio checkpoints.

Thin entrypoint: the scoring math (MOS parsing, caption metrics, diagnostics)
lives in :mod:`asa.eval.metrics` and is shared with every other eval. This file
is just the ``eval-mos`` command plus the audio-path plumbing around it. Metric
helpers are re-exported at module level so ``from evaluate import extract_mos``
(and the sibling temporal/Gemini scripts) keep working unchanged.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

from asa.eval.metrics import (
    compute_caption_metrics,
    diversity_metrics,
    extract_mos,
    log_caption_metrics,
    mos_regression_metrics,
)
from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records
from asa.prompts import build_zeroshot_prompt_MOS

# Re-exported for callers/tests that import these names from this module.
__all__ = [
    "compute_caption_metrics",
    "extract_mos",
    "log_caption_metrics",
    "app",
    "eval_mos",
]

# Off-the-shelf (untrained) Qwen2-Audio chat model, used as the zero-shot
# baseline. The source paper reports this model cannot do speech quality
# assessment without fine-tuning; the --zero-shot row reproduces that.
ZEROSHOT_BASELINE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

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
    zero_shot: bool = typer.Option(
        False,
        "--zero-shot",
        help=(
            "Zero-shot baseline mode: evaluate an untrained off-the-shelf model "
            "with an instructed prompt (dimension definitions + 'end with an MOS "
            "score') instead of the bare prompt the fine-tuned models saw. When "
            "no --model-path is given, defaults to the off-the-shelf "
            f"{ZEROSHOT_BASELINE_MODEL}. Metrics use the identical code path as "
            "every fine-tuned row, so the baseline stays comparable."
        ),
    ),
    seed: int = typer.Option(
        42, help="Random seed; makes sampled decoding reproducible across runs."
    ),
):
    """Run model inference and evaluate quality based on MOS and BLEU."""
    import torch

    torch.manual_seed(seed)

    # Resolve model_path: prefer the command option, then the global, then the
    # default. In --zero-shot mode the default is the off-the-shelf baseline
    # model rather than the fine-tuned SFT checkpoint.
    if model_path is None:
        model_path = ctx.obj.get("model_path", None)
    if model_path is None:
        model_path = ZEROSHOT_BASELINE_MODEL if zero_shot else ASAModel.SFT

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

        # In zero-shot mode, override the bare PROMPT_TEMPLATE (which the
        # fine-tuned models were trained on) with the instructed, non-leaking
        # zero-shot prompt rendered through the Instruct model's chat template.
        # Identical for every sample; run_inference falls back to PROMPT_TEMPLATE
        # when prompt_texts is None.
        prompt_texts = (
            [build_zeroshot_prompt_MOS(processor)] * len(audio_paths)
            if zero_shot
            else None
        )

        logging.info(
            "Running inference (zero_shot=%s, do_sample=%s, temperature=%.2f, "
            "top_p=%.2f, max_new_tokens=%d)...",
            zero_shot,
            do_sample,
            temperature,
            top_p,
            max_new_tokens,
        )
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            prompt_texts=prompt_texts,
            device=device,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        logging.info("Calculating metrics...")
        results = []
        mos_errors = []  # only over samples whose MOS parsed
        hyps: List[str] = []
        refs: List[str] = []

        for item, pred in zip(data, predictions):
            pred = pred.strip()
            true_mos = float(item["mos"])
            pred_mos = extract_mos(pred)
            true_resp = item["response"]

            # pred_mos is None when no score could be confidently parsed (an
            # honest failure, common for the untrained zero-shot baseline). Such
            # samples are excluded from MAE/MSE and counted via the parse rate;
            # the old "last number" fallback would have invented a number here.
            if pred_mos is not None:
                error = abs(true_mos - pred_mos)
                mos_errors.append(error)
            else:
                error = None

            hyps.append(pred)
            refs.append(true_resp)

            res_item = item.copy()
            res_item["predicted_response"] = pred
            res_item["predicted_mos"] = pred_mos
            res_item["mos_error"] = error
            results.append(res_item)

        # MAE/MSE over parsed samples only. When every sample parses (the case
        # for all fine-tuned runs, whose captions always end in "MOS of X"),
        # parsed == total, so these numbers are identical to the previous
        # implementation and previously reported results do not move.
        mos_agg = mos_regression_metrics(mos_errors, len(data))
        n_parsed = mos_agg["parsed"]
        parse_rate = mos_agg["parse_rate"]
        mae = mos_agg["mae"]
        mse = mos_agg["mse"]

        caption_metrics = compute_caption_metrics(hyps, refs)

        diversity = diversity_metrics(hyps)
        unique_predictions = diversity["unique_predictions"]
        top_prediction_frequency = diversity["top_prediction_frequency"]

        logging.info("=" * 40)
        logging.info(f"EVALUATION RESULTS FOR {dataset_path.name}:")
        logging.info(f"Samples evaluated:                    {len(data)}")
        logging.info(
            f"MOS parse rate:                       {parse_rate:.4f} "
            f"({n_parsed}/{len(data)})"
        )
        logging.info(f"MOS MAE (over parsed):                {mae:.4f}")
        logging.info(f"MOS MSE (over parsed):                {mse:.4f}")
        log_caption_metrics(caption_metrics)
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
                        "mos_parsed": n_parsed,
                        "mos_parse_rate": parse_rate,
                        "mae": mae,
                        "mse": mse,
                        **caption_metrics,
                        "unique_predictions": unique_predictions,
                        "top_prediction_frequency": top_prediction_frequency,
                    },
                    "decoding": {
                        "do_sample": do_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                    },
                    "run": {
                        "model_path": str(model_path),
                        "zero_shot": zero_shot,
                        "prompt": (prompt_texts[0] if zero_shot else None),
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
