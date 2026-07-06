"""Temporal-localization evaluation CLI for Qwen2-Audio checkpoints.

Thin entrypoint. The temporal model emits a joint answer: a MOS-style quality
caption plus a localized degradation interval. Interval parsing, t-IoU, offsets,
ground-truth extraction, caption timestamp stripping and the audio-blind
baselines live in :mod:`asa.eval.intervals`; the caption/MOS scoring reuses
:mod:`asa.eval.metrics` verbatim, so those numbers are directly comparable to
the global-task tables. This file is the ``eval-temporal`` command around them.

Interval and metric helpers are re-exported at module level so
``from evaluate_temporal import Interval, extract_interval, ...`` (used by the
tests and by ``evaluate_gemini_temporal.py``) keeps working unchanged.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from statistics import median
from typing import Any, List, Optional

import typer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from asa.data import PROMPT_TEMPLATE
from asa.eval.intervals import (
    Interval,
    best_constant_baseline,
    extract_ground_truth_interval,
    extract_interval,
    interval_iou,
    interval_offset_error,
    query_to_prompt,
    strip_non_timestamp_special_tokens,
    strip_time_tokens_for_caption,
    whole_clip_baseline_mean_tiou,
    _safe_float,
)
from asa.eval.metrics import (
    BERTSCORE_MODEL,
    compute_caption_metrics,
    extract_mos,
    mean_or_zero,
    mos_regression_metrics,
)
from asa.inference import ASAModel, load_model, run_inference
from asa.processed_data import load_processed_records, resolve_audio_path
from asa.prompts import build_zeroshot_prompt_temporal

# Re-exported so `from evaluate_temporal import ...` keeps resolving for the test
# suite and the Gemini temporal script (the historical public surface of this
# module before the scoring logic moved into asa.eval).
__all__ = [
    "Interval",
    "best_constant_baseline",
    "extract_ground_truth_interval",
    "extract_interval",
    "interval_iou",
    "interval_offset_error",
    "query_to_prompt",
    "strip_non_timestamp_special_tokens",
    "strip_time_tokens_for_caption",
    "whole_clip_baseline_mean_tiou",
    "app",
    "eval_temporal",
]

EVAL_TEMPERATURE = 0.7
EVAL_TOP_P = 0.9
EVAL_MAX_NEW_TOKENS = 150

# Off-the-shelf (untrained) Qwen2-Audio chat model, used as the zero-shot
# temporal baseline. The source paper reports off-the-shelf audio LLMs cannot do
# speech quality assessment without fine-tuning; the --zero-shot row reproduces
# that finding for the temporal-localization task (t-IoU floor before training).
ZEROSHOT_BASELINE_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(
    help="Evaluate fine-tuned Qwen2-Audio models on temporal localization."
)


@app.command()
def eval_temporal(
    dataset_paths: List[Path] = typer.Option(
        ..., "--dataset-path", help="Paths to the test JSONL datasets."
    ),
    model_path: Optional[str] = typer.Option(
        None,
        help=(
            "Hub repo ID or local checkpoint path. Defaults to the fine-tuned "
            "SFT model, or to the off-the-shelf Instruct baseline under "
            "--zero-shot."
        ),
    ),
    zero_shot: bool = typer.Option(
        False,
        "--zero-shot",
        help=(
            "Evaluate the off-the-shelf (untrained) Qwen2-Audio-7B-Instruct "
            "baseline. Uses a ChatML chat-template prompt instead of each "
            "record's bare query prompt, and suppresses the plain-number "
            "interval fallback so a rambling answer cannot manufacture a bogus "
            "interval. This is the defensible 'before fine-tuning' t-IoU floor."
        ),
    ),
    data_root: Path = typer.Option(
        Path("data"),
        help="Root directory used to resolve audio paths.",
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to evaluate (for testing)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Dir to save results. Defaults to results/evaluation/<model_name>_temporal.",
    ),
    batch_size: int = typer.Option(4, help="Inference batch size."),
    use_query_prompt: bool = typer.Option(
        True,
        "--use-query-prompt/--use-default-prompt",
        help="Use each record's temporal query prompt instead of the default prompt.",
    ),
    do_sample: bool = typer.Option(
        False,
        "--do-sample/--greedy",
        help="Sample with temperature/top_p or use greedy decoding.",
    ),
    temperature: float = typer.Option(
        EVAL_TEMPERATURE,
        help="Sampling temperature (used when --do-sample).",
    ),
    top_p: float = typer.Option(
        EVAL_TOP_P,
        help="Nucleus top-p (used when --do-sample).",
    ),
    max_new_tokens: int = typer.Option(
        EVAL_MAX_NEW_TOKENS,
        help="Max new tokens to generate per sample.",
    ),
    bertscore_model: str = typer.Option(
        BERTSCORE_MODEL,
        help="HuggingFace backbone for caption BERTScore (recorded in output).",
    ),
    seed: int = typer.Option(
        42, help="Random seed; makes sampled decoding reproducible across runs."
    ),
) -> None:
    """Run temporal inference and report localization quality metrics."""
    import torch

    torch.manual_seed(seed)

    # In --zero-shot mode the model defaults to the off-the-shelf Instruct
    # baseline; otherwise it defaults to the fine-tuned SFT checkpoint.
    if model_path is None:
        model_path = ZEROSHOT_BASELINE_MODEL if zero_shot else ASAModel.SFT

    if output_dir is None:
        model_name = Path(model_path).name or "model"
        output_dir = Path(f"results/evaluation/{model_name}_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model from %s", model_path)
    processor, model, device = load_model(model_path)

    # The zero-shot baseline uses one ChatML-templated, non-leaking prompt for
    # every row (overriding the per-record query prompt), and parses predictions
    # without the plain-number fallback so a non-temporal ramble cannot fake an
    # interval. allow_plain stays True for fine-tuned models (unchanged path).
    zeroshot_prompt: Optional[str] = None
    if zero_shot:
        zeroshot_prompt = build_zeroshot_prompt_temporal(processor)
        audio_token_count = zeroshot_prompt.count("<|AUDIO|>")
        if audio_token_count != 1:
            raise ValueError(
                "Zero-shot temporal prompt must contain exactly one <|AUDIO|> "
                f"token for run_inference alignment, found {audio_token_count}."
            )
        logging.info(
            "Zero-shot temporal baseline: ChatML prompt, plain-number fallback "
            "suppressed (parse via explicit range only)."
        )

    for dataset_path in dataset_paths:
        logging.info("Loading dataset from %s", dataset_path)
        rows = load_processed_records(dataset_path)
        if max_samples is not None:
            rows = rows[:max_samples]
            logging.info("Limited evaluation to %d samples", len(rows))

        resolved_rows: list[dict[str, Any]] = []
        missing_audio_ref = 0
        missing_audio_file = 0

        for item in rows:
            audios = item.get("audios")
            if not isinstance(audios, list) or not audios:
                missing_audio_ref += 1
                continue

            raw_audio = str(audios[0])
            resolved_audio = resolve_audio_path(raw_audio, data_root)
            if not resolved_audio.exists():
                missing_audio_file += 1
                continue

            duration_seconds = _safe_float(item.get("duration_seconds"))
            truth_interval, truth_source = extract_ground_truth_interval(item)
            if zeroshot_prompt is not None:
                prompt = zeroshot_prompt
            elif use_query_prompt:
                prompt = query_to_prompt(item.get("query"))
            else:
                prompt = PROMPT_TEMPLATE
            resolved_rows.append(
                {
                    "record": item,
                    "audio_path": str(resolved_audio),
                    "duration_seconds": duration_seconds,
                    "truth_interval": truth_interval,
                    "truth_source": truth_source,
                    "prompt": prompt,
                }
            )

        if not resolved_rows:
            raise ValueError(
                "No evaluable rows found after audio path resolution. "
                f"dataset={dataset_path}, data_root={data_root}"
            )

        logging.info(
            "Running inference on %d rows (skipped: missing_audios=%d, missing_files=%d)",
            len(resolved_rows),
            missing_audio_ref,
            missing_audio_file,
        )
        predictions = run_inference(
            model=model,
            processor=processor,
            audio_paths=[row["audio_path"] for row in resolved_rows],
            prompt_texts=[row["prompt"] for row in resolved_rows],
            device=device,
            batch_size=batch_size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=False,
        )

        details: list[dict[str, Any]] = []
        ious: list[float] = []
        start_errors: list[float] = []
        end_errors: list[float] = []
        # Signed endpoint offsets (predicted - truth): positive = predicted late,
        # negative = predicted early. Reported as means to expose directional bias
        # that the absolute errors above hide.
        start_offsets: list[float] = []
        end_offsets: list[float] = []
        offset_errors: list[float] = []
        parsed_count = 0
        ground_truth_count = 0
        # Tally which parse strategy produced each prediction interval. For the
        # zero-shot baseline this is the audit surface: "plain" should be absent
        # (it is suppressed), and a healthy baseline parses via "range" or not
        # at all. A surprising distribution is the smoke-gate signal.
        pred_source_counts: dict[str, int] = {}
        # Set of distinct (start, end) predicted intervals (rounded to 0.01 s).
        # This is the honesty diagnostic, the temporal parallel to the MOS
        # "unique predictions N/N" line. A near-constant interval (count of 1-2
        # across many samples) means the model emits a canned answer and any
        # nonzero t-IoU is chance overlap, NOT localization. A genuinely
        # localizing model varies its interval with the audio.
        unique_pred_intervals: set[tuple[float, float]] = set()
        # Caption and MOS accumulators. The temporal answer carries the same
        # MOS-style caption as the global task, so we score it identically:
        # MOS MAE/MSE over samples whose MOS parsed, and corpus/sample caption
        # metrics over the prose with the timestamp clause stripped (the interval
        # is already scored by IoU). hyps/refs are gold-aligned 1:1.
        caption_hyps: list[str] = []
        caption_refs: list[str] = []
        mos_errors: list[float] = []

        for row, prediction in zip(resolved_rows, predictions):
            record = row["record"]
            duration_seconds = row["duration_seconds"]
            prediction_text = strip_non_timestamp_special_tokens(prediction)

            pred_interval, pred_source = extract_interval(
                prediction_text, duration_seconds, allow_plain=not zero_shot
            )
            truth_interval = row["truth_interval"]
            truth_source = row["truth_source"]

            # MOS: parse the predicted score from the same cleaned text, compare
            # to the construction-time gold MOS. None means an honest parse
            # failure (excluded from MAE/MSE, surfaced via the MOS parse rate).
            gold_mos = _safe_float(record.get("mos"))
            pred_mos = extract_mos(prediction_text)
            mos_abs_err: Optional[float] = None
            if gold_mos is not None and pred_mos is not None:
                mos_abs_err = abs(gold_mos - pred_mos)
                mos_errors.append(mos_abs_err)

            # Caption: strip the localization clause from both sides so BLEU/
            # ROUGE/BERTScore reflect the descriptive prose only.
            gold_caption = str(record.get("response", ""))
            caption_hyps.append(strip_time_tokens_for_caption(prediction_text))
            caption_refs.append(strip_time_tokens_for_caption(gold_caption))

            pred_source_counts[pred_source] = (
                pred_source_counts.get(pred_source, 0) + 1
            )
            if pred_interval is not None:
                unique_pred_intervals.add(
                    (round(pred_interval.start, 2), round(pred_interval.end, 2))
                )
                parsed_count += 1
            if truth_interval is not None:
                ground_truth_count += 1

            # Compute the endpoint errors once and reuse them for both the
            # accumulators and the per-record detail row.
            tiou = 0.0
            start_abs_err: Optional[float] = None
            end_abs_err: Optional[float] = None
            start_offset_err: Optional[float] = None
            end_offset_err: Optional[float] = None
            offset_err: Optional[float] = None
            if pred_interval is not None and truth_interval is not None:
                tiou = interval_iou(pred_interval, truth_interval)
                start_offset_err = pred_interval.start - truth_interval.start
                end_offset_err = pred_interval.end - truth_interval.end
                start_abs_err = abs(start_offset_err)
                end_abs_err = abs(end_offset_err)
                offset_err = interval_offset_error(pred_interval, truth_interval)
                start_errors.append(start_abs_err)
                end_errors.append(end_abs_err)
                start_offsets.append(start_offset_err)
                end_offsets.append(end_offset_err)
                offset_errors.append(offset_err)
            if truth_interval is not None:
                ious.append(tiou)

            detail = dict(record)
            detail["audio_path_resolved"] = row["audio_path"]
            detail["predicted_response"] = prediction_text
            detail["gt_interval_source"] = truth_source
            detail["pred_interval_source"] = pred_source
            detail["gt_start"] = (
                truth_interval.start if truth_interval is not None else None
            )
            detail["gt_end"] = (
                truth_interval.end if truth_interval is not None else None
            )
            detail["pred_start"] = (
                pred_interval.start if pred_interval is not None else None
            )
            detail["pred_end"] = (
                pred_interval.end if pred_interval is not None else None
            )
            detail["tiou"] = tiou
            detail["start_abs_err"] = start_abs_err
            detail["end_abs_err"] = end_abs_err
            detail["start_offset_err"] = start_offset_err
            detail["end_offset_err"] = end_offset_err
            detail["offset_err"] = offset_err
            # MOS fields mirror the global eval's per-record schema so the same
            # downstream analysis (e.g. caption-vs-MOS) works on these JSONs too.
            detail["gold_mos"] = gold_mos
            detail["predicted_mos"] = pred_mos
            detail["mos_error"] = mos_abs_err
            details.append(detail)

        # MOS MAE/MSE over samples whose MOS parsed (same convention as the
        # global eval: an unparsed prediction is honest-failure, not a zero).
        mos_agg = mos_regression_metrics(mos_errors, len(resolved_rows))
        n_mos_parsed = mos_agg["parsed"]
        mos_parse_rate = mos_agg["parse_rate"]
        mos_mae = mos_agg["mae"]
        mos_mse = mos_agg["mse"]

        # Caption metrics over the timestamp-stripped prose. compute_caption_metrics
        # is the global eval's helper, so BLEU (corpus, cased), ROUGE-1/2/L F1 and
        # BERTScore P/R/F1 are computed exactly as on the global MOS task.
        caption_metrics = compute_caption_metrics(
            caption_hyps, caption_refs, bertscore_model=bertscore_model
        )

        metrics = {
            "samples_total": len(resolved_rows),
            "samples_with_ground_truth_interval": ground_truth_count,
            "samples_with_parsed_prediction_interval": parsed_count,
            "skipped_missing_audios_field": missing_audio_ref,
            "skipped_missing_audio_file": missing_audio_file,
            "mean_tiou": mean_or_zero(ious),
            "median_tiou": median(ious) if ious else 0.0,
            "hit_iou_ge_0_1": mean_or_zero([1.0 if v >= 0.1 else 0.0 for v in ious]),
            "hit_iou_ge_0_3": mean_or_zero([1.0 if v >= 0.3 else 0.0 for v in ious]),
            "hit_iou_ge_0_5": mean_or_zero([1.0 if v >= 0.5 else 0.0 for v in ious]),
            "mean_start_abs_err": mean_or_zero(start_errors),
            "mean_end_abs_err": mean_or_zero(end_errors),
            "expected_offset_error": mean_or_zero(offset_errors),
            "mean_start_offset_err": mean_or_zero(start_offsets),
            "mean_end_offset_err": mean_or_zero(end_offsets),
            "parse_rate": parsed_count / len(resolved_rows),
            "pred_interval_source_counts": pred_source_counts,
            "unique_pred_intervals": len(unique_pred_intervals),
            # MOS regression on the joint answer. Same scoring as the global task.
            "mos_mae": mos_mae,
            "mos_mse": mos_mse,
            "mos_parse_rate": mos_parse_rate,
            "samples_with_parsed_mos": n_mos_parsed,
            # Caption quality on the timestamp-stripped prose. The thesis grid
            # reads caption_bleu, rouge_l and bertscore; the remaining BLEU/ROUGE/
            # BERTScore sub-scores are kept for completeness and parity with the
            # global eval JSONs.
            "caption_bleu": caption_metrics["bleu"],
            "rouge_l": caption_metrics["rougeL_f"],
            "bertscore": caption_metrics["bertscore_f1"],
            "caption_metrics": caption_metrics,
            "caption_scoring": "timestamp_tokens_stripped",
            "prompt_mode": (
                "zero_shot_chatml"
                if zero_shot
                else ("query" if use_query_prompt else "default")
            ),
            "zero_shot": zero_shot,
            "model_path": model_path,
            "prompt": zeroshot_prompt if zero_shot else None,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        logging.info("========================================")
        logging.info("TEMPORAL EVALUATION: %s", dataset_path.name)
        logging.info("Samples evaluated: %d", metrics["samples_total"])
        logging.info("Prediction parse rate: %.4f", parsed_count / len(resolved_rows))
        logging.info("Pred interval sources: %s", pred_source_counts)
        logging.info(
            "Unique predicted intervals: %d of %d parsed (honesty diagnostic; "
            "near-1 means canned answer, not localization)",
            len(unique_pred_intervals),
            parsed_count,
        )
        logging.info("Mean t-IoU: %.4f", metrics["mean_tiou"])
        logging.info("Median t-IoU: %.4f", metrics["median_tiou"])
        logging.info("Hit@0.1: %.4f", metrics["hit_iou_ge_0_1"])
        logging.info("Hit@0.3: %.4f", metrics["hit_iou_ge_0_3"])
        logging.info("Hit@0.5: %.4f", metrics["hit_iou_ge_0_5"])
        logging.info("Mean |start error|: %.4f", metrics["mean_start_abs_err"])
        logging.info("Mean |end error|: %.4f", metrics["mean_end_abs_err"])
        logging.info(
            "Mean signed start/end offset: %.4f / %.4f  (expected offset %.4f)",
            metrics["mean_start_offset_err"],
            metrics["mean_end_offset_err"],
            metrics["expected_offset_error"],
        )
        logging.info(
            "MOS MAE: %.4f  MOS MSE: %.4f  (parse rate %.4f, %d/%d)",
            mos_mae,
            mos_mse,
            mos_parse_rate,
            n_mos_parsed,
            len(resolved_rows),
        )
        logging.info(
            "Caption (timestamps stripped) -> BLEU: %.2f  ROUGE-L: %.4f  "
            "BERTScore-F1: %.4f",
            metrics["caption_bleu"],
            metrics["rouge_l"],
            metrics["bertscore"],
        )
        logging.info("========================================")

        dataset_name = dataset_path.stem
        out_json = output_dir / f"{dataset_name}_results.json"
        out_csv = output_dir / f"{dataset_name}_results.csv"

        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(
                {"metrics": metrics, "results": details},
                handle,
                indent=2,
                ensure_ascii=False,
            )

        csv_columns = [
            "id",
            "filename_deg",
            "mix_filename",
            "audio_path_resolved",
            "duration_seconds",
            "gt_start",
            "gt_end",
            "pred_start",
            "pred_end",
            "tiou",
            "start_abs_err",
            "end_abs_err",
            "start_offset_err",
            "end_offset_err",
            "offset_err",
            "gt_interval_source",
            "pred_interval_source",
            "mos",
            "gold_mos",
            "predicted_mos",
            "mos_error",
            "predicted_response",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=csv_columns, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(details)

        logging.info("Saved detailed results to %s", out_json)
        logging.info("Saved tabular results to %s", out_csv)


if __name__ == "__main__":
    app()
