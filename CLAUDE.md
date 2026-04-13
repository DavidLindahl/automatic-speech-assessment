# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`asa` (automatic-speech-assessment) — DTU bachelor project on using transformer audio LLMs (primarily Qwen2-Audio, with SALMONN as a reference) for automatic speech quality assessment. Authors: David Lindahl, Carl Svejstrup. Active research direction is shifting from MOS / A/B preference prediction toward timestamp-localized descriptions of distortions (see `temporal.localization` branch).

## Toolchain

- Python 3.12, managed entirely with `uv` (do not use bare `pip`/`python`)
- Torch is platform-pinned via `[tool.uv.sources]`: CPU wheels on darwin, CUDA 11.8 wheels on linux
- Linting/formatting: `ruff` (line length 88, enforced via pre-commit)
- Task runner: `invoke` (see `tasks.py`)
- Tests: `pytest` (slow tests are marked `@pytest.mark.slow` — they load full Qwen2-Audio checkpoints)

## Common commands

```bash
# Run any python entrypoint
uv run <path/to/script.py>

# Tests
uv run pytest tests/                           # all tests
uv run pytest tests/test_collator.py -k name   # single test
uv run pytest -m "not slow"                    # skip checkpoint-loading tests
uv run invoke test                             # tests + coverage report

# Lint / format
uv run ruff check . --fix
uv run ruff format .
uv run pre-commit run --all-files

# Invoke tasks
uv run invoke --list
uv run invoke preprocess-sft   # data.py preprocess-sft data/raw data/processed
uv run invoke download-data    # pulls from GCS bucket nisqa-dataset
uv run invoke train

# Datasets
uv run scripts/download_quali_speech.py --output-dir data/raw/QualiSpeech

# Docs
uv run mkdocs serve --config-file docs/mkdocs.yaml

# Docker (HPC / cloud training)
uv run invoke docker-build
```

Installed console scripts: `asa-infer` (`asa.inference:app`), `asa-eval` (`asa.evaluate:app`).

## Architecture

The real code lives in `src/asa/`. The `model.py` / `train.py` files at the package root are stubs from the cookiecutter template — they are not the active training path. Actual training and inference go through the dedicated entrypoint scripts below.

### Data pipeline (`src/asa/data.py`, `processed_data.py`)

- `SFTDataset` loads JSONL records + WAV files on the fly. Two prompt modes:
  - Single-clip SFT: `PROMPT_TEMPLATE` ("Please describe and evaluate the synthetic speech")
  - A/B preference: `PROMPT_TEMPLATE_AB` (two `<audio>` slots + tie option)
- `Qwen2AudioCollator` batches samples and calls the HF `Qwen2-Audio` processor. Audio is resampled to 16 kHz mono (`TARGET_SR`).
- `processed_data.load_processed_records` accepts JSON array, JSONL, or object-stream JSON. Metadata field tuples (`DPO_METADATA_FIELDS`, `..._AB`) define which MOS/NOI/COL/DIS/LOUD scores are propagated through the pipeline — keep these in sync if you add a metric.
- Raw data is downloaded from GCS bucket `nisqa-dataset`; `download_quali_speech.py` fetches the QualiSpeech HF snapshot.

### Training entrypoints (multiple, by objective)

The package contains *parallel* training scripts rather than a single `train.py` with flags. Pick the one matching the objective:

- `supervised-finetune.py` / `supervised-finetune-ab.py` — SFT on single clips vs. A/B pairs
- `dpo-finetune.py` / `dpo-finetune-ab.py` — DPO variants
- `generate_dpo_data.py`, `generate_temporal_data.py`, `distill_temporal_targets.py` — synthetic / temporal label generation
- `caption_generator.py` — generates natural-language captions used as SFT targets
- `train.py` (root) — stub, do not extend; add new objectives as their own entrypoint script following the existing pattern

DeepSpeed config for multi-GPU runs lives in `configs/ds_zero2.json`. HPC launch wrappers are in `jobs/sft/*.sh`, `jobs/train/*.sh`, `jobs/evaluate/`.

### Evaluation & inference

- `evaluate.py` — general eval (`asa-eval` Typer app)
- `evaluate_temporal.py` — temporal-localization specific metrics
- `inference.py` — `asa-infer` Typer app
- `api.py` — FastAPI server wrapper (see `dockerfiles/api.dockerfile`)

### Branch layout

The repo has many active branches reflecting parallel experiments: `main`, `temporal.localization` (timestamp-output direction — currently the active scope), `dpo-ab`, `dpo_implementation`, `feature/hf-data-pipeline`, `Tain_Warmpup`, `Clean-slate`, `refactor`, etc. When making changes, check which branch you are on and whether the work belongs there before committing.

## Conventions (from AGENTS.md)

- Line length 120 in human-written files, but ruff formats to 88 — let ruff win on autoformatted files
- f-strings only
- Type hints on all public functions
- Google-style docstrings; every function and class should have one
- Do not add inline comments unless absolutely necessary
- Update `docs/` (mkdocs) when adding user-facing functionality
- Update `AGENTS.md` if a new tool or command is added to the project
