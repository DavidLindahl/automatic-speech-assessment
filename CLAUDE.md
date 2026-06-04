# CLAUDE.md — Automatic Speech Assessment (ASA)

In-repo playbook for Claude Code when operating inside this project. This file is **standalone** — it duplicates the relevant parts of `AGENTS.md` so a fresh Claude session doesn't need to read two files to get started.

## What this project does

ASA is a bachelor project fine-tuning **Qwen2-Audio-7B** for automatic assessment of synthetic/recorded speech quality. Active scope: **MOS-style descriptive quality assessment**, with descriptive natural-language outputs evaluated by BLEU + MOS regression metrics.

Training paradigms in active use:

1. **SFT** (`scripts/train/supervised-finetune.py`) — single-audio MOS-style quality prediction. Primary path.
2. **DPO** (`scripts/train/dpo-finetune.py`) — Direct Preference Optimization. Uses the ALLD dual-stream method: a trainable Qwen2-Audio policy and a frozen Qwen2-7B text reference model, batched by `ALLDDPOCollator`.

**A/B preference is dropped from the bachelor scope.** All AB-specific entrypoints, datasets, collators, and prompts were removed in Wave 1 of the refactor (2026-05-26). Don't propose new A/B work; if you find an AB reference, it's either historical (in `jobs/_archive/`, `scripts/_legacy/`, or `data/processed/intermediate/`) or a bug that needs flagging.

Target sample rate is fixed at 16 kHz (`TARGET_SR` in `src/asa/audio.py`, re-exported through `src/asa/data.py`). The public inference API lives in `src/asa/inference.py` (`load_model`, `run_inference`). The MOS-style eval CLI is `scripts/eval/evaluate.py`; the temporal-localization eval CLI is `scripts/eval/evaluate_temporal.py`. Both are typer-based.

## Workflow split: laptop vs HPC

Two distinct surfaces. Don't conflate them.

- **Laptop (this repo, `~/code/dtu/automatic-speech-assessment/`)** — all code changes happen here. Read files locally, edit locally, run tests/lint locally, commit, push to `main`.
- **HPC (DTU LSF cluster, via `ssh dtu`)** — all training, evaluation, and log inspection happens here. Submit jobs with `bsub`, monitor with `bjobs`/`bstat`, tail logs in `/work3/s234817/automatic-speech-assessment/logs/`. The HPC checkout pulls from the same remote, so `git pull` after a push gets the latest code there.

Operational rule: when a question is "what does the code do / let me change it" → laptop. When a question is "what's the job doing / did it crash / what does the log say" → ssh to HPC.

### Analysis workflow: commit on HPC, analyse on laptop

When an analysis needs more than a quick `tail`/`grep` over SSH (e.g. computing MAE, BLEU, plotting, comparing eval JSONs, diffing dataset variants), do **not** run the analysis on HPC. Instead:

1. **On HPC**: `cd /work3/s234817/automatic-speech-assessment`, `git add results/<run>/ data/processed/<file>.json` (only small artifacts — JSON metric summaries, prediction files, dataset JSONLs), `git commit -m "results: ..."`, `git push`.
2. **On laptop**: `git pull`, then write/run the analysis script locally under `scripts/` with `uv run`. All plotting, stats, comparisons happen here.

Why: local tools are instant, scripts get version-controlled next to the data they consume, and the HPC shell stays focused on `bsub`/`bjobs`/`tail`. Never commit checkpoints, raw audio, full prediction dumps with waveforms, or anything > a few MB — those stay on `/work3/` or go to HF Hub. For large prediction files, either `scp` them down or compute summary stats on HPC and commit only the summary.

## Directory map

```
automatic-speech-assessment/
├── src/asa/                         # IMPORTABLE LIBRARY ONLY (no entrypoints)
│   ├── audio.py                       # load_audio, TARGET_SR (16 kHz)
│   ├── prompts.py                     # PROMPT_TEMPLATE, MOS expert-prompt builder
│   ├── datasets.py                    # SFTDataset, DPODataset
│   ├── collators.py                   # Qwen2AudioCollator, ALLDDPOCollator
│   ├── inference.py                   # load_model + run_inference public API
│   ├── processed_data.py              # JSONL loaders, audio-path resolution
│   ├── generate_temporal_data.py      # library helpers (overlay_noise, ...)
│   ├── distill_temporal_targets.py    # library: smoke-set target distillation
│   ├── sampler.py                     # dataset-sampling utilities
│   ├── data.py                        # compatibility shim re-exporting the above
│   └── README.md                      # full module map
│
├── scripts/                         # RUNNABLE ENTRYPOINTS, grouped by purpose
│   ├── train/                         # SFT + DPO trainers, submission shell
│   ├── eval/                          # evaluate.py, evaluate_temporal.py
│   ├── data/                          # data builders + smoke prep
│   ├── diagnostics/                   # collapse probes, sanity checkers
│   ├── analysis/                      # post-eval aggregators
│   ├── _legacy/                       # pre-temporal caption generator (archival only)
│   └── README.md
│
├── jobs/                            # LSF bsub submission
│   ├── sft/                           # SFT jobs
│   ├── train/                         # DPO + data-generation jobs
│   ├── evaluate/                      # eval jobs
│   ├── upload_*.sh                    # HF Hub uploaders
│   ├── _lib/                          # shared preamble, lint-budget firebreak, eval template
│   └── _archive/                      # historical scripts; not on the live path
│
├── configs/
│   ├── ds_zero2.json                  # DeepSpeed Zero-2 with CPU offload
│   └── ds_zero2_no_offload.json
│
├── data/
│   ├── raw/                           # NISQA_Corpus, P501, LIVE, FOR (gitignored)
│   ├── processed/                     # grouped by use, see data/processed/README.md
│   │   ├── sft/                         # SFT training inputs
│   │   ├── dpo/                         # DPO chosen/rejected pairs
│   │   ├── eval/                        # test_FOR / test_LIVE / test_P501 / test_nisqa_indomain
│   │   ├── temporal/                    # temporal-localization mixes + metadata (current focus)
│   │   └── intermediate/                # build artifacts, AB legacy
│   └── inference/                     # ad-hoc inference inputs
│
├── models/                          # Saved checkpoints (gitignored; on /work3 on HPC)
├── results/
│   ├── evaluation/                    # JSON metric summaries, grouped by model type
│   │   ├── dpo/                         # DPO checkpoints
│   │   ├── sft/                         # SFT checkpoints
│   │   └── temporal/                    # SFT-on-temporal-mix checkpoints (current scope)
│   ├── inference/                     # Per-sample predictions
│   └── analysis/                      # Plots, summary tables
│
├── logs/                            # Job stdout/stderr (gitignored)
├── wandb/                           # Local W&B run cache (gitignored)
├── tests/                           # pytest; CPU-safe subset runs in CI
├── pyproject.toml                   # uv-managed; Python 3.12
├── uv.lock
└── tasks.py                         # invoke tasks (uv run invoke --list)
```

**Where to look first when a question lands on your desk:**

- "How is X trained?" → `scripts/train/`
- "How is X evaluated?" → `scripts/eval/`
- "Where does dataset X come from?" → `scripts/data/`
- "Why does this prompt look this way?" → `src/asa/prompts.py`
- "What does the SFT collator do?" → `src/asa/collators.py`
- "What's the LSF preamble doing?" → `jobs/_lib/preamble.sh` and `jobs/_lib/README.md`
- "What JSONs feed what jobs?" → `data/processed/README.md`

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

Code style: **line length 88** (enforced by ruff and pre-commit), f-strings, type hints, Google docstrings, no inline comments unless absolutely necessary. If you add new tooling, update both this file and `AGENTS.md`.

## Decoding and BLEU policy

Settled defaults — change only with explicit reason:

- **Evaluation (`scripts/eval/evaluate.py`)**: `do_sample=True, temperature=0.7, top_p=0.9`. Greedy and beam are not the default; greedy collapses to repeated templates, beam is parked for a later A/B comparison study.
- **DPO data generation** (sampling π_θ for rejected completions): `temperature=1.1, top_p=0.9`. Encourages enough diversity in the negatives that DPO has a real signal. The previous `temp=1.0, top_p=1.0` setting led to length-bias reward hacking — do not revert.
- **BLEU metric**: `sacrebleu.corpus_bleu` reported on a 0–100 scale. The earlier `nltk.sentence_bleu` per-sample average is wrong by a factor of ~100x and unsmoothed — do not use for headline numbers.

## HPC specifics

Do **not** run training on the login node (`gbarlogin1`) or `transfer.gbar.dtu.dk`. Always go through LSF.

Minimum environment for any GPU script:

```bash
module load cuda/11.8
source /work3/s234817/automatic-speech-assessment/.venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
```

Home dir quota is ~30 GB — never write large artifacts to `~`. Everything lives under `/work3/s234817/`. Check with `getquota_zhome.sh`. /work3 quota is 100 GB; check with `getquota_work3.sh`.

## Submitting jobs

Claude is authorized to submit jobs freely when the user asks. Prefer copying an existing script in `jobs/` over writing a new one — they already encode the right queue, modules, env vars, and log paths.

**Hard workflow rule — jobs reach the HPC via GitHub, never by editing on DTU.** Write the `.sh` job script (and any code change) in this repo on the laptop, commit, push/merge to GitHub, then on DTU `git pull` and `bsub`. Never `ssh dtu` and edit a script or source file in place — that creates untracked drift between the HPC checkout and GitHub. Never run ad-hoc `python -c` training/eval on the HPC; wrap everything that runs on the cluster in a committed job script. Local one-off Python for *inspection* (tokenizer/data sanity checks) is fine.

**Hard rule — `/asa-update-site` after every run-state change.** Whenever a run changes state (submitted, completed, failed, parked, killed, re-statused), invoke the `asa-update-site` skill right after updating the `runs/` entry and `runs/INDEX.md`. The public experiment-log site goes stale silently otherwise. A run-state change not reflected on the site is an incomplete task.

**HTML / frontend artifacts — use the `frontend-design` skill.** Any HTML report, dashboard, or frontend output goes through `/frontend-design` for production-grade design; do not hand-roll CSS/HTML.

```bash
bsub < jobs/global/sft/sft_full.sh         # SFT full
bsub < jobs/global/sft/sft_warmup.sh       # SFT warmup
bsub < jobs/global/alld/dpo.sh             # DPO
bsub < jobs/global/eval/evaluate-sft-mos.sh # MOS evaluation
```

Jobs are organized by task then role: `jobs/<global|temporal>/<alld|sft|eval|data>/`.
`alld/` and `sft/` hold training scripts (DPO and SFT respectively), `eval/` holds all
evaluation scripts for that task, `data/` holds the `generate_*`/`build_*` data-prep jobs.
Checkpoint uploads live in `jobs/upload/`.

Queues in use:

| Queue | Hardware | Used for |
|-------|----------|----------|
| `gpuh100` | H100 80 GB | **All training runs (SFT and DPO)** — default for any new training job |
| `gpua10`  | A10        | Lightweight eval |
| `gpul40s` | L40S 48 GB | Legacy SFT runs only — do not submit new training here |
| `gpua40`  | A40 40 GB  | Legacy DPO runs only — do not submit new training here |

**Training submission rule:** every new training run (SFT, DPO, continuation) goes to `gpuh100`. Older `jobs/global/sft/*.sh` and `jobs/global/alld/*.sh` (and the `temporal/` equivalents) that target `gpul40s` or `gpua40` predate this rule — when reusing them, swap the queue to `gpuh100` and commit the change before submitting. Don't carry stale queues forward.

Typical training resource ask on H100: `#BSUB -n 8`, `#BSUB -gpu "num=1:mode=exclusive_process"` (single H100 has enough VRAM for SFT+DPO at typical batch sizes), `#BSUB -R "rusage[mem=64GB]"`, `#BSUB -M 64GB`, `#BSUB -W 24:00`. Bump to `num=2` only if VRAM forces it.

Every job script writes:
```
logs/<name>_<LSB_JOBID>.out
logs/<name>_<LSB_JOBID>.err
```

## Example: reference SFT submit script

This is the recommended template for new SFT runs (H100, single GPU is usually enough). The on-disk `jobs/global/sft/sft_full.sh` may still target `gpul40s` from older work — swap the queue when copying.

```sh
#!/bin/sh
#BSUB -q gpuh100
#BSUB -J sft-full
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
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

python scripts/train/supervised-finetune.py \
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
- **Projects**: `qwen2-audio-sft-simple` (SFT), `qwen2-audio-alld` (DPO/ALLD).
- Pass `--wandb-run-name "<name>"` on the training CLI.
- Local cache in `wandb/`. Offline runs are synced when connectivity returns — don't delete.

## Outputs — where things land

| Location | Contents |
|----------|----------|
| `models/<name>/` | Checkpoints (`sft_full`, `sft_warmup`, `dpo_final`, ...) |
| `results/evaluation/<dpo\|sft\|temporal>/<run>/` | JSON metric summaries per test set (FOR / LIVE / P501 / nisqa_indomain) |
| `results/inference/<dataset>/` | Per-sample predictions |
| `results/analysis/` | Plots, summary tables |
| `logs/` | Job stdout/stderr |

## Published HF Hub models

From `src/asa/inference.py`:

- `Leng2beat/speech-quality-assessement-qwen2audio-full-sft`
- `Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup-baseline`
- `Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup-plus1epoch`
- Additional ALLD variants planned.

DPO reference model: `Qwen/Qwen2-7B` (text-only, frozen).

## Gotchas

- **`PYTHONUNBUFFERED=1`** — without this, jobs look hung because stdout buffers until completion.
- **`TRITON_CACHE_DIR=/tmp/triton_cache`** — avoids venv bloat and permission issues.
- **DeepSpeed Zero-2 + CPU offload** — reduces VRAM but increases CPU RAM. If you see OOM kills, bump `rusage[mem=...]`.
- **BF16 vs FP16** — A40/L40S/H100 support BF16; V100 does not. Don't use `--bf16` on V100 queues.
- **Batch size** — SFT typically 4/device, DPO typically 2/device. Halve if OOM.
- **Audio sample rate** — Qwen2-Audio requires 16 kHz. `data.py` resamples; don't bypass it.
- **Evaluation queues** — eval jobs submitted to non-GPU queues silently fail with "No Nvidia-GPUs found". Use `gpua10` at minimum.
- **Unstaged work** — files like `supervised-finetune.py` may have in-flight changes. `git status` first; ask before `git restore`.
- **Branch hygiene** — don't force-push. Don't `--no-verify` past a failing pre-commit — fix it.

## When in doubt

- Read `AGENTS.md` for the short tooling cheatsheet.
- Read the most recent `logs/*.err` file (via ssh) to understand the last job's failure mode.
- Check `git log --oneline -20` and `git status` before making changes — work is often in flight.
- Ask the user rather than guessing queue, walltime, or checkpoint paths.
