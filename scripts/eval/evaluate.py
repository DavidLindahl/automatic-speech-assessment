import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import sacrebleu
import typer

from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records
from asa.prompts import build_zeroshot_prompt_MOS

# Off-the-shelf (untrained) Qwen2-Audio chat model, used as the zero-shot
# baseline. The source paper reports this model cannot do speech quality
# assessment without fine-tuning; the --zero-shot row reproduces that.
ZEROSHOT_BASELINE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.9
EVAL_MAX_NEW_TOKENS = 150

# Default BERTScore backbone. roberta-large is the bert-score library default and
# the most commonly cited choice for English; record it in the output for
# reproducibility since BERTScore values depend on the backbone.
BERTSCORE_MODEL = "roberta-large"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Evaluate fine-tuned Qwen2-Audio models on standard datasets.")


def compute_caption_metrics(
    hyps: List[str],
    refs: List[str],
    bertscore_model: str = BERTSCORE_MODEL,
) -> Dict[str, Any]:
    """Compute lexical and semantic caption-quality metrics.

    Three complementary views of how close each predicted caption is to its
    reference:

    - BLEU (sacrebleu corpus): n-gram precision, the surface phrasing overlap.
      Reported cased and lowercased. Corpus-level aggregation.
    - ROUGE-1/2/L (rouge-score): n-gram recall plus longest-common-subsequence.
      The recall-side mirror of BLEU. Per-sample F1, then averaged.
    - BERTScore P/R/F1 (bert-score): token-embedding cosine similarity, the only
      semantic (synonym-aware) view. Per-sample, then averaged.

    Args:
        hyps: Predicted captions.
        refs: Reference captions, aligned 1:1 with ``hyps``.
        bertscore_model: HuggingFace backbone for BERTScore (recorded in output).

    Returns:
        Flat dict of metric name to score. BLEU on the 0-100 sacrebleu scale;
        ROUGE and BERTScore on 0-1.
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps/refs length mismatch: {len(hyps)} vs {len(refs)}")

    # Imported lazily so the module loads even when these extras are absent
    # (e.g. on a node where only MOS/BLEU is needed).
    from rouge_score import rouge_scorer
    from bert_score import score as bertscore_fn

    # BLEU: corpus-level n-gram precision, cased and lowercased.
    bleu_cased = sacrebleu.corpus_bleu(hyps, [refs]).score
    bleu_lc = sacrebleu.corpus_bleu(
        [h.lower() for h in hyps], [[r.lower() for r in refs]]
    ).score

    # ROUGE: per-sample F1 for unigram, bigram, and LCS overlap, then averaged.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for hyp, ref in zip(hyps, refs):
        scores = scorer.score(ref, hyp)  # (target, prediction) order
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)

    n = max(len(hyps), 1)
    rouge1_f = sum(rouge1) / n
    rouge2_f = sum(rouge2) / n
    rougel_f = sum(rougel) / n

    # BERTScore: semantic similarity via contextual embeddings, averaged.
    logging.info("Computing BERTScore with backbone %s...", bertscore_model)
    bs_p, bs_r, bs_f1 = bertscore_fn(
        hyps,
        refs,
        model_type=bertscore_model,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )

    return {
        "bleu": bleu_cased,
        "bleu_lowercased": bleu_lc,
        "rouge1_f": rouge1_f,
        "rouge2_f": rouge2_f,
        "rougeL_f": rougel_f,
        "bertscore_precision": bs_p.mean().item(),
        "bertscore_recall": bs_r.mean().item(),
        "bertscore_f1": bs_f1.mean().item(),
        "bertscore_model": bertscore_model,
        # BLEU is corpus-level (sacrebleu); ROUGE and BERTScore are per-sample
        # then averaged. Recorded so cited numbers stay comparable across runs.
        "caption_metric_aggregation": {
            "bleu": "corpus",
            "rouge": "sample_mean_f1",
            "bertscore": "sample_mean",
        },
    }


def log_caption_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print the caption metrics produced by ``compute_caption_metrics``."""
    logging.info("BLEU (corpus, cased):       %.2f", metrics["bleu"])
    logging.info("BLEU (corpus, lowercased):  %.2f", metrics["bleu_lowercased"])
    logging.info(
        "ROUGE-1 / -2 / -L F1 (mean): %.4f / %.4f / %.4f",
        metrics["rouge1_f"],
        metrics["rouge2_f"],
        metrics["rougeL_f"],
    )
    logging.info(
        "BERTScore P / R / F1 (mean, %s): %.4f / %.4f / %.4f",
        metrics["bertscore_model"],
        metrics["bertscore_precision"],
        metrics["bertscore_recall"],
        metrics["bertscore_f1"],
    )


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


def extract_mos(text: str) -> Optional[float]:
    """Extract the model's MOS score from generated text.

    Returns the parsed score, or ``None`` when no score can be confidently
    located. ``None`` is deliberate: the old behaviour fell back to "the last
    number in the text", which on a zero-shot baseline that rambles or rates
    "3 out of 5" grabs the wrong digit (the "5" denominator) and fabricates a
    plausible-but-wrong MOS. An honest parse failure is better than a fake
    number, especially for the untrained baseline whose whole point is that it
    cannot do the task. Callers treat ``None`` as unparsed and report a parse
    rate alongside the error over parsed samples.

    Patterns are tried most-specific first:

    1. Explicit "MOS" mention, e.g. "MOS of 4.3", "overall MOS is 4.3". This is
       the format the fine-tuned models were trained to emit, so this branch is
       unchanged from the original implementation and their parsed scores are
       byte-for-byte identical.
    2. Out-of-5 ratings, e.g. "3 out of 5", "rated as 4 out of 5", "3/5". Takes
       the numerator, not the "5" denominator.
    3. "score/rating of X" and "rate ... as X" phrasings.

    There is intentionally NO blind last-number fallback.
    """
    # 1. Explicit MOS mention (fine-tuned format) — unchanged.
    match = re.search(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # 2. "X out of 5" / "X/5" — take the numerator (the rating), not the 5.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:out\s+of|/)\s*5\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # 3. "score of X" / "rating of X" / "rating is X" / "rating: X" / "rate ...
    #    as X" / "rate it (a) X".
    match = re.search(
        r"(?:score|rating)\s*(?:of|is|:|=)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    match = re.search(
        r"rate\s+(?:it|the\s+\w+(?:\s+\w+)?)\s+(?:as\s+)?(?:a\s+)?(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))

    # No confident match — honest parse failure.
    return None


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
):
    """Run model inference and evaluate quality based on MOS and BLEU."""

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
        n_parsed = len(mos_errors)
        parse_rate = n_parsed / max(len(data), 1)
        mae = sum(mos_errors) / n_parsed if n_parsed else float("nan")
        mse = sum(e**2 for e in mos_errors) / n_parsed if n_parsed else float("nan")

        caption_metrics = compute_caption_metrics(hyps, refs)

        unique_predictions = len(set(hyps))
        top_prediction_frequency = max(Counter(hyps).values()) / max(len(hyps), 1)

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


@app.command()
def rescore(
    ctx: typer.Context,
    results_paths: List[Path] = typer.Option(
        ...,
        "--results-path",
        help="Existing *_results.json file(s) from a prior eval run to re-score.",
    ),
    bertscore_model: str = typer.Option(
        BERTSCORE_MODEL, help="HuggingFace backbone for BERTScore."
    ),
    in_place: bool = typer.Option(
        False,
        "--in-place/--no-in-place",
        help="Merge caption metrics back into the source JSON instead of a sidecar.",
    ),
):
    """Re-score saved predictions with BLEU + ROUGE + BERTScore, no inference.

    Reads each ``*_results.json`` (produced by ``eval-mos``), pulls the stored
    reference (``response``) and prediction (``predicted_response``) for every
    sample, and computes the caption metrics offline. Useful for adding the new
    metrics to past runs without re-running the model. Writes a
    ``*_caption_metrics.json`` sidecar by default, or merges into the source
    file with ``--in-place``.
    """
    for results_path in results_paths:
        logging.info("Re-scoring %s", results_path)
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload.get("results", [])
        hyps: List[str] = []
        refs: List[str] = []
        skipped = 0
        for record in records:
            hyp = record.get("predicted_response")
            ref = record.get("response")
            if not isinstance(hyp, str) or not isinstance(ref, str):
                skipped += 1
                continue
            hyps.append(hyp.strip())
            refs.append(ref.strip())

        if not hyps:
            logging.warning(
                "No scorable (response, predicted_response) pairs in %s; skipping.",
                results_path,
            )
            continue
        if skipped:
            logging.warning(
                "Skipped %d record(s) missing response/predicted_response.", skipped
            )

        caption_metrics = compute_caption_metrics(
            hyps, refs, bertscore_model=bertscore_model
        )

        logging.info("=" * 40)
        logging.info("RE-SCORE: %s", results_path.name)
        logging.info("Samples scored: %d", len(hyps))
        log_caption_metrics(caption_metrics)
        logging.info("=" * 40)

        scored = {"samples_scored": len(hyps), **caption_metrics}
        if in_place:
            payload.setdefault("metrics", {}).update(scored)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logging.info("Merged caption metrics into %s\n", results_path)
        else:
            sidecar = results_path.with_name(
                f"{results_path.stem}_caption_metrics.json"
            )
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(scored, f, indent=2, ensure_ascii=False)
            logging.info("Saved caption metrics to %s\n", sidecar)


if __name__ == "__main__":
    app()
