# CLAUDE.md — Automatic Speech Assessment (ASA)

In-repo playbook for Claude Code when operating inside this project. This file is **standalone** — it duplicates the relevant parts of `AGENTS.md` so a fresh Claude session doesn't need to read two files to get started.

## What this project does

ASA is a bachelor project fine-tuning **Qwen2-Audio-7B** for automatic assessment of synthetic/recorded speech quality. Three training paradigms:

1. **SFT** (`src/asa/supervised-finetune.py`) — single-audio MOS-style quality prediction.
2. **SFT-AB** (`src/asa/supervised-finetune-ab.py`) — comparative A/B preference on paired audio samples (including ties).
3. **DPO** (`src/asa/dpo-finetune.py`) — Direct Preference Optimization. Uses the ALLD dual-stream method: a trainable Qwen2-Audio policy and a frozen Qwen2-7B text reference model, batched by `ALLDDPOCollator`.

Target sample rate is fixed at 16 kHz (`TARGET_SR` in `src/asa/data.py`). Inference + FastAPI service live in `src/asa/inference.py` and `src/asa/api.py`. Evaluation CLI is `src/asa/evaluate.py` (typer-based).

## Directory map

```
automatic-speech-assessment/
├── src/asa/                         # All source code
│   ├── supervised-finetune.py       # SFT training entry point
│   ├── supervised-finetune-ab.py    # SFT A/B training entry point
│   ├── dpo-finetune.py              # DPO training entry point
│   ├── data.py                      # Datasets + collators (Qwen2AudioCollator, ALLDDPOCollator)
│   ├── processed_data.py            # JSONL loaders, audio path resolution
│   ├── inference.py                 # load_model / run_inference API
│   ├── evaluate.py                  # MOS/BLEU eval CLI
│   ├── api.py                       # FastAPI service
│   ├── caption_generator.py         # Caption/prompt generation for AB data
│   ├── generate_dpo_data.py         # Build DPO training pairs
│   └── visualize.py                 # Plotting utilities
│
├── jobs/                            # LSF bsub scripts — COPY, DON'T REWRITE
│   ├── sft/         sft_full.sh, sft_warmup.sh, sft_ab_full.sh, sft_ab_warmup.sh, sft_debug.sh
│   ├── train/       dpo.sh, dpo_ab.sh, dpo_test.sh, generate_dpo.sh
│   ├── evaluate/    evaluate-sft-mos.sh
│   └── upload_*.sh  # push checkpoints to HF Hub
│
├── configs/
│   └── ds_zero2.json                # DeepSpeed Zero-2 w/ CPU offload
│
├── data/
│   ├── raw/                         # Original datasets (FOR, LIVE, P501, NISQA, ...)
│   ├── processed/                   # JSONL used by training/eval
│   │   ├── train_nisqa_llama_10k.json
│   │   ├── train_nisqa_abtest_llama_10k.json
│   │   ├── train_dpo_10k.json
│   │   ├── test_FOR.json, test_LIVE.json, test_P501.json
│   │   └── ab-test-set-captions.json
│   └── inference/                   # Inference inputs
│
├── models/                          # Saved checkpoints (sft_full, sft_warmup, dpo_final, ...)
├── results/
│   ├── evaluation/                  # JSON metric summaries per model/test set
│   ├── inference/                   # Per-sample predictions
│   └── analysis/                    # Plots, summary tables
│
├── logs/                            # Job stdout/stderr (named <job>_<LSB_JOBID>.{out,err})
├── wandb/                           # Local W&B run cache — do not clean blindly
├── tests/                           # pytest suite
├── pyproject.toml                   # uv-managed; Python 3.12
├── uv.lock
├── tasks.py                         # invoke tasks (uv run invoke --list)
└── AGENTS.md                        # tooling basics (kept in sync with this file)
```

## Python environment

- **Package manager**: `uv`. Never use bare `pip`.
- **Python**: 3.12 (see `.python-version`).
- **Venv**: `.venv/` at the repo root.

```bash
source .venv/bin/activate
uv add <package>                      # add dep
uv sync                               # reinstall from uv.lock
uv run pytest tests/                  # run tests
uv run ruff format .                  # format
uv run ruff check . --fix             # lint
uv run invoke --list                  # list project tasks
uv run pre-commit run --all-files     # all pre-commit hooks
```

Code style: line length 120, f-strings, type hints, Google docstrings, no inline comments unless absolutely necessary. If you add new tooling, update both this file and `AGENTS.md`.

## HPC specifics

Do **not** run training on the login node (`gbarlogin1`) or `transfer.gbar.dtu.dk`. Always go through LSF.

Minimum environment for any GPU script:

```bash
module load cuda/11.8
source /work3/s234817/automatic-speech-assessment/.venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

Home dir quota is ~30 GB — never write large artifacts to `~`. Everything lives under `/work3/s234817/`. Check with `getquota_zhome.sh`.

## Submitting jobs

Claude is authorized to submit jobs freely when the user asks. Prefer copying an existing script in `jobs/` over writing a new one — they already encode the right queue, modules, env vars, and log paths.

```bash
bsub < jobs/sft/sft_full.sh              # SFT full (10k, 2 ep, 2x L40S)
bsub < jobs/sft/sft_warmup.sh            # SFT warmup
bsub < jobs/sft/sft_ab_full.sh           # A/B SFT
bsub < jobs/train/dpo.sh                 # DPO (2x A40)
bsub < jobs/train/dpo_ab.sh              # A/B DPO
bsub < jobs/evaluate/evaluate-sft-mos.sh # MOS evaluation
```

Queues in use:

| Queue | Hardware | Used for |
|-------|----------|----------|
| `gpul40s` | L40S 48 GB | SFT training |
| `gpua40`  | A40 40 GB  | DPO training |
| `gpua10`  | A10        | Lightweight eval |

Typical training resource ask: `#BSUB -n 8`, `#BSUB -gpu "num=2:mode=exclusive_process"`, `#BSUB -R "rusage[mem=64GB]"`, `#BSUB -M 64GB`, `#BSUB -W 24:00`.

Every job script writes:
```
logs/<name>_<LSB_JOBID>.out
logs/<name>_<LSB_JOBID>.err
```

## Example: reference SFT submit script

This is `jobs/sft/sft_full.sh`. Copy it when creating new SFT variants:

```sh
#!/bin/sh
#BSUB -q gpul40s
#BSUB -J sft-full
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o logs/sft_full_%J.out
#BSUB -e logs/sft_full_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

nvidia-smi

torchrun \
    --nproc_per_node=2 \
    src/asa/supervised-finetune.py \
    --model-name sft_full \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 100 \
    --wandb-run-name "full-10k-2ep"
```

## Monitoring and managing jobs

```bash
bstat                   # active jobs
bstat -M                # memory usage
bstat -V                # CPU efficiency
bjobs -a                # all jobs incl. finished
bkill <JOBID>           # cancel a job
bnvtop <JOBID>          # GPU usage for a job
showstart <JOBID>       # estimated start
nodestat                # cluster node status
tail -f logs/<name>_<jobid>.out
```

## W&B

- **Entity**: `speech-quality-DTU-bachelor`
- **Projects**: `qwen2-audio-sft-simple` (SFT), `qwen2-audio-dpo` (DPO).
- Pass `--wandb-run-name "<name>"` on the training CLI.
- Local cache in `wandb/`. Offline runs are synced when connectivity returns — don't delete.

## Outputs — where things land

| Location | Contents |
|----------|----------|
| `models/<name>/` | Checkpoints (`sft_full`, `sft_warmup`, `dpo_final`, `dpo_ab_final`, ...) |
| `results/evaluation/<run>/` | JSON metric summaries per test set (FOR / LIVE / P501) |
| `results/inference/<dataset>/` | Per-sample predictions |
| `results/analysis/` | Plots, summary tables |
| `logs/` | Job stdout/stderr |

## Published HF Hub models

From `src/asa/inference.py`:

- `Leng2beat/speech-quality-assessement-qwen2audio-full-sft`
- `Leng2beat/speech-quality-assessement-qwen2audio-sft-ab`
- Additional ALLD variants planned.

DPO reference model: `Qwen/Qwen2-7B` (text-only, frozen).

## Gotchas

- **`PYTHONUNBUFFERED=1`** — without this, jobs look hung because stdout buffers until completion.
- **`TRITON_CACHE_DIR=/tmp/triton_cache`** — avoids venv bloat and permission issues.
- **DeepSpeed Zero-2 + CPU offload** — reduces VRAM but increases CPU RAM. If you see OOM kills, bump `rusage[mem=...]`.
- **BF16 vs FP16** — A40/L40S support BF16; V100 does not. Don't use `--bf16` on V100 queues.
- **Batch size** — SFT typically 4/device, DPO typically 2/device. Halve if OOM.
- **Audio sample rate** — Qwen2-Audio requires 16 kHz. `data.py` resamples; don't bypass it.
- **Evaluation queues** — eval jobs submitted to non-GPU queues silently fail with "No Nvidia-GPUs found" (see old `NONAME_*.out`). Use `gpua10` at minimum.
- **Unstaged work** — the `supervised-finetune-ab.py` / other files may have in-flight changes. `git status` first; ask before `git restore`.
- **Branch hygiene** — current branch is `main`. Don't force-push. Don't `--no-verify` past a failing pre-commit — fix it.

## When in doubt

- Read `AGENTS.md` for the short tooling cheatsheet (it's the same content as above, in shorter form).
- Read the most recent `logs/*.err` file to understand the last job's failure mode.
- Check `git log --oneline -20` and `git status` before making changes — work is often in flight.
- Ask the user rather than guessing queue, walltime, or checkpoint paths.
