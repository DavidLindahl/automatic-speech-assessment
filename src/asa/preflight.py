"""Read-only preflight checks for HPC training and evaluation jobs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import typer

from asa.processed_data import (
    DPO_METADATA_FIELDS,
    load_processed_records,
    resolve_audio_path,
)

app = typer.Typer(
    help="Validate datasets, checkpoints, and job script flags before submitting HPC jobs."
)

SCRIPT_FLAGS = {
    "src/asa/supervised-finetune.py": {
        "--model-id",
        "--json-path",
        "--data-root",
        "--output-dir",
        "--batch-size",
        "--epochs",
        "--lr",
        "--gradient-accumulation-steps",
        "--bf16",
        "--fp16",
        "--max-samples",
        "--deepspeed",
        "--val-split",
        "--eval-steps",
        "--wandb-entity",
        "--wandb-project",
        "--wandb-run-name",
    },
    "src/asa/dpo-finetune.py": {
        "--model-id",
        "--ref-model-id",
        "--json-path",
        "--data-root",
        "--output-dir",
        "--batch-size",
        "--epochs",
        "--beta",
        "--lr",
        "--gradient-accumulation-steps",
        "--bf16",
        "--fp16",
        "--max-samples",
        "--deepspeed",
        "--val-split",
        "--eval-steps",
        "--wandb-entity",
        "--wandb-project",
        "--wandb-run-name",
    },
    "src/asa/generate_dpo_data.py": {
        "--input-json",
        "--output-json",
        "--model-path",
        "--data-root",
        "--batch-size",
        "--max-samples",
    },
    "src/asa/evaluate.py": {
        "--dataset-path",
        "--model-path",
        "--data-root",
        "--max-samples",
        "--output-dir",
        "--batch-size",
    },
}

DEFAULT_JOB_SCRIPTS = {
    "sft": [
        Path("jobs/sft/sft_debug.sh"),
        Path("jobs/sft/sft_warmup.sh"),
        Path("jobs/sft/sft_full.sh"),
    ],
    "generate-dpo": [Path("jobs/train/generate_dpo.sh")],
    "dpo": [Path("jobs/train/dpo.sh"), Path("jobs/train/dpo_test.sh")],
    "evaluate": [Path("jobs/evaluate/evaluate.sh")],
    "pipeline": [
        Path("jobs/sft/sft_debug.sh"),
        Path("jobs/sft/sft_warmup.sh"),
        Path("jobs/sft/sft_full.sh"),
        Path("jobs/train/generate_dpo.sh"),
        Path("jobs/train/dpo.sh"),
        Path("jobs/train/dpo_test.sh"),
        Path("jobs/evaluate/evaluate.sh"),
    ],
}
VALID_MODES = tuple(DEFAULT_JOB_SCRIPTS)


def dataset_requirements(mode: str) -> dict[Path, set[str]]:
    """Return required dataset files and keys for a given mode."""
    requirements: dict[Path, set[str]] = {}
    if mode in {"sft", "generate-dpo", "pipeline"}:
        requirements[Path("data/processed/train_nisqa_llama_10k.json")] = {
            "audios",
            "response",
            *DPO_METADATA_FIELDS,
        }
    if mode in {"dpo", "pipeline"}:
        requirements[Path("data/processed/train_dpo_10k.json")] = {
            "audios",
            "chosen",
            "rejected",
            *DPO_METADATA_FIELDS,
        }
    if mode in {"evaluate", "pipeline"}:
        for name in ("test_FOR.json", "test_LIVE.json", "test_P501.json"):
            requirements[Path("data/processed") / name] = {"audios", "response", "mos"}
    return requirements


def file_requirements(mode: str) -> list[Path]:
    """Return non-dataset files that must exist for a mode."""
    required: list[Path] = []
    if mode in {"sft", "dpo", "pipeline"}:
        required.append(Path("configs/ds_zero2.json"))
    if mode in {"generate-dpo", "dpo", "evaluate", "pipeline"}:
        required.append(Path("models/sft_warmup"))
    return required


def validate_records(
    dataset_path: Path,
    required_keys: set[str],
    data_root: Path,
    audio_check_limit: int,
) -> list[str]:
    """Validate processed records and sample audio path resolution."""
    findings: list[str] = []
    records = load_processed_records(dataset_path)
    if not records:
        return [f"{dataset_path}: dataset is empty"]

    for index, record in enumerate(records, start=1):
        missing = sorted(key for key in required_keys if key not in record)
        if missing:
            findings.append(
                f"{dataset_path}: record {index} missing keys {', '.join(missing)}"
            )
            break

    for index, record in enumerate(records[:audio_check_limit], start=1):
        audios = record.get("audios", [])
        if not audios:
            findings.append(f"{dataset_path}: record {index} has no audio paths")
            break
        resolved = resolve_audio_path(audios[0], data_root)
        if not resolved.exists():
            findings.append(
                f"{dataset_path}: record {index} audio path does not exist after resolution: {resolved}"
            )
            break

    return findings


def validate_job_script(job_script: Path) -> list[str]:
    """Validate job script CLI flags against the current Python entrypoint contract."""
    text = job_script.read_text(encoding="utf-8")
    findings: list[str] = []
    for entrypoint, allowed_flags in SCRIPT_FLAGS.items():
        if entrypoint not in text:
            continue
        tail = text[text.index(entrypoint) :]
        used_flags = set(re.findall(r"--[a-z0-9-]+", tail))
        invalid_flags = sorted(flag for flag in used_flags if flag not in allowed_flags)
        if invalid_flags:
            findings.append(
                f"{job_script}: invalid flags for {entrypoint}: {', '.join(invalid_flags)}"
            )
    return findings


def run_preflight_checks(
    mode: str, data_root: Path, audio_check_limit: int, job_scripts: Iterable[Path]
) -> list[str]:
    """Run all configured preflight checks and return failures."""
    findings: list[str] = []
    for required_file in file_requirements(mode):
        if not required_file.exists():
            findings.append(f"Missing required file or directory: {required_file}")

    for dataset_path, required_keys in dataset_requirements(mode).items():
        if not dataset_path.exists():
            findings.append(f"Missing dataset: {dataset_path}")
            continue
        findings.extend(
            validate_records(dataset_path, required_keys, data_root, audio_check_limit)
        )

    for job_script in job_scripts:
        if not job_script.exists():
            findings.append(f"Missing job script: {job_script}")
            continue
        findings.extend(validate_job_script(job_script))

    return findings


@app.command()
def check(
    mode: str = typer.Option("pipeline", help=f"One of: {', '.join(VALID_MODES)}."),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory that contains the raw audio tree."
    ),
    audio_check_limit: int = typer.Option(
        100, help="How many records per dataset to verify for audio path existence."
    ),
    job_script: Optional[list[Path]] = typer.Option(
        None,
        help="Optional job script paths to validate. Defaults to scripts for the selected mode.",
    ),
) -> None:
    """Validate pipeline prerequisites without mutating the workspace."""
    if mode not in VALID_MODES:
        raise typer.BadParameter(
            f"Unsupported mode '{mode}'. Choose from {', '.join(VALID_MODES)}."
        )

    scripts = job_script or DEFAULT_JOB_SCRIPTS[mode]
    findings = run_preflight_checks(mode, data_root, audio_check_limit, scripts)

    if findings:
        for finding in findings:
            typer.echo(f"FAIL: {finding}")
        raise typer.Exit(code=1)

    typer.echo(
        f"Preflight OK for mode '{mode}' across {len(list(scripts))} job script(s)."
    )


if __name__ == "__main__":
    app()
