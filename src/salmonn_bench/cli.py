"""CLI for zero-shot SALMONN benchmark execution."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from salmonn_bench.config import load_benchmark_config
from salmonn_bench.data import load_records, resolve_audio_path
from salmonn_bench.eval import evaluate_ab, evaluate_mos
from salmonn_bench.inference import SalmonnZeroShotRunner, dump_predictions
from salmonn_bench.prompts import load_task_prompt

app = typer.Typer(help="Zero-shot SALMONN benchmark runner.")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_run_dir(output_dir: Path, run_id: str | None) -> Path:
    resolved_run_id = run_id or _timestamp()
    run_dir = output_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@app.command("run-mos")
def run_mos(
    dataset_paths: list[Path] = typer.Option(..., "--dataset-path", help="One or more MOS dataset JSONL paths."),
    config_path: Path = typer.Option(
        Path("configs/salmonn_zeroshot.yaml"), "--config-path", help="Benchmark config YAML path."
    ),
    data_root: Path = typer.Option(
        Path("data"), "--data-root", help="Local data root used to resolve dataset audio paths."
    ),
    output_dir: Path = typer.Option(
        Path("results/salmonn_zeroshot"), "--output-dir", help="Directory for run outputs."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id label."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Optional cap per dataset for smoke runs."),
) -> None:
    """Run zero-shot SALMONN MOS benchmark."""
    config = load_benchmark_config(config_path)
    task_prompt = load_task_prompt(config.prompts_path, config.mos_task)

    run_dir = _prepare_run_dir(output_dir, run_id)
    _save_json(
        run_dir / "run_config.json",
        {
            "config_path": str(config_path),
            "datasets": [str(p) for p in dataset_paths],
            "task": config.mos_task,
            "max_samples": max_samples,
            "mode": "mos",
        },
    )

    typer.echo(f"Loading SALMONN model on {config.device}...")
    runner = SalmonnZeroShotRunner(config)

    for dataset_path in dataset_paths:
        typer.echo(f"Running MOS dataset: {dataset_path}")
        records = load_records(dataset_path)
        if max_samples is not None:
            records = records[:max_samples]

        prediction_rows: list[dict[str, Any]] = []
        for idx, record in enumerate(records, start=1):
            audio_raw = str(record["audios"][0])
            audio_path = resolve_audio_path(audio_raw, data_root)
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio: {audio_path}")

            pred = runner.infer_mos(audio_path, task_prompt)
            out_row = dict(record)
            out_row["resolved_audio"] = str(audio_path)
            out_row["predicted_response"] = pred
            prediction_rows.append(out_row)

            if idx % 20 == 0:
                typer.echo(f"  Processed {idx}/{len(records)}")

        dataset_name = dataset_path.stem
        predictions_path = run_dir / f"{dataset_name}_predictions.jsonl"
        dump_predictions(predictions_path, prediction_rows)

        metrics, rows = evaluate_mos(prediction_rows)
        _save_json(run_dir / f"{dataset_name}_metrics.json", metrics)
        dump_predictions(run_dir / f"{dataset_name}_results.jsonl", rows)

        typer.echo(
            f"  {dataset_name}: MSE={metrics['mse']:.4f}, LCC={metrics['lcc']:.4f}, "
            f"SRCC={metrics['srcc']:.4f}, BLEU={metrics['bleu']:.4f}"
        )


@app.command("run-ab")
def run_ab(
    dataset_paths: list[Path] = typer.Option(..., "--dataset-path", help="One or more A/B dataset JSONL paths."),
    config_path: Path = typer.Option(
        Path("configs/salmonn_zeroshot.yaml"), "--config-path", help="Benchmark config YAML path."
    ),
    data_root: Path = typer.Option(
        Path("data"), "--data-root", help="Local data root used to resolve dataset audio paths."
    ),
    output_dir: Path = typer.Option(
        Path("results/salmonn_zeroshot"), "--output-dir", help="Directory for run outputs."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id label."),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Optional cap per dataset for smoke runs."),
) -> None:
    """Run zero-shot SALMONN A/B benchmark."""
    config = load_benchmark_config(config_path)
    task_prompt = load_task_prompt(config.prompts_path, config.ab_task)

    run_dir = _prepare_run_dir(output_dir, run_id)
    _save_json(
        run_dir / "run_config.json",
        {
            "config_path": str(config_path),
            "datasets": [str(p) for p in dataset_paths],
            "task": config.ab_task,
            "max_samples": max_samples,
            "mode": "ab",
        },
    )

    typer.echo(f"Loading SALMONN model on {config.device}...")
    runner = SalmonnZeroShotRunner(config)

    for dataset_path in dataset_paths:
        typer.echo(f"Running A/B dataset: {dataset_path}")
        records = load_records(dataset_path)
        if max_samples is not None:
            records = records[:max_samples]

        prediction_rows: list[dict[str, Any]] = []
        for idx, record in enumerate(records, start=1):
            audio_a_raw = str(record["audios"][0])
            audio_b_raw = str(record["audios"][1])
            audio_a_path = resolve_audio_path(audio_a_raw, data_root)
            audio_b_path = resolve_audio_path(audio_b_raw, data_root)

            if not audio_a_path.exists():
                raise FileNotFoundError(f"Missing audio A: {audio_a_path}")
            if not audio_b_path.exists():
                raise FileNotFoundError(f"Missing audio B: {audio_b_path}")

            pred = runner.infer_ab(audio_a_path, audio_b_path, task_prompt)
            out_row = dict(record)
            out_row["resolved_audio_a"] = str(audio_a_path)
            out_row["resolved_audio_b"] = str(audio_b_path)
            out_row["predicted_response"] = pred
            prediction_rows.append(out_row)

            if idx % 20 == 0:
                typer.echo(f"  Processed {idx}/{len(records)}")

        dataset_name = dataset_path.stem
        predictions_path = run_dir / f"{dataset_name}_predictions.jsonl"
        dump_predictions(predictions_path, prediction_rows)

        metrics, rows = evaluate_ab(prediction_rows)
        _save_json(run_dir / f"{dataset_name}_metrics.json", metrics)
        dump_predictions(run_dir / f"{dataset_name}_results.jsonl", rows)

        typer.echo(f"  {dataset_name}: Acc={metrics['accuracy']:.4f}, BLEU={metrics['bleu']:.4f}")


if __name__ == "__main__":
    app()
