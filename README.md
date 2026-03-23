# SALMONN Zero-Shot Benchmark

This branch is a clean rebuild focused on one goal: run **zero-shot SALMONN** for speech quality evaluation benchmarks.

## Scope

Implemented in this branch:
1. Zero-shot MOS evaluation.
2. Zero-shot A/B preference evaluation.
3. Metric reporting (MAE, MSE, LCC, SRCC, BLEU, accuracy).

Not implemented in this branch:
1. LoRA training.
2. Q-former tuning.
3. Distillation or preference optimization.

## Repository Layout

- `src/salmonn_bench/`: benchmark package
- `third_party/salmonn/`: vendored SALMONN runtime code
- `third_party/salmonn_sqa/`: SQA prompt file
- `configs/salmonn_zeroshot.yaml`: runtime config template
- `jobs/run_salmonn_zeroshot.sh`: convenience benchmark runner
- `plan.md`: live migration and execution plan

## Setup

```bash
uv sync
```

## Required Model Assets

Set these paths in `configs/salmonn_zeroshot.yaml`:
1. `llama_path`
2. `whisper_path`
3. `beats_path`
4. `ckpt`

## Run Benchmarks

MOS benchmark:

```bash
uv run salmonn-bench run-mos \
  --config-path configs/salmonn_zeroshot.yaml \
  --dataset-path data/processed/test_FOR.json \
  --dataset-path data/processed/test_LIVE.json \
  --dataset-path data/processed/test_P501.json
```

A/B benchmark:

```bash
uv run salmonn-bench run-ab \
  --config-path configs/salmonn_zeroshot.yaml \
  --dataset-path data/processed/train_nisqa_abtest_llama_10k.json
```

Or run the script:

```bash
bash jobs/run_salmonn_zeroshot.sh
```

The branch is code-only and expects NISQA audio to be mounted externally. By default:

```bash
DATA_ROOT=data
```

Override when needed:

```bash
DATA_ROOT=/path/to/data bash jobs/run_salmonn_zeroshot.sh
```

Restore local NISQA layout in repo (ignored by git):

```bash
mkdir -p data/raw
cp -r /path/to/NISQA_Corpus data/raw/NISQA_Corpus
```

Or symlink instead of copy:

```bash
mkdir -p data/raw
ln -s /path/to/NISQA_Corpus data/raw/NISQA_Corpus
```

For cluster submission:

```bash
bsub < jobs/run_salmonn_zeroshot.sh
```
