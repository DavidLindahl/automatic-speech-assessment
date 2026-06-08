#!/bin/bash
### ============================================================
### DTU HPC — ZERO-SHOT BASELINE evaluation (untrained Qwen2-Audio)
###
### Evaluates the off-the-shelf Qwen/Qwen2-Audio-7B-Instruct model on the
### MOS test sets with NO fine-tuning, using the instructed (non-leaking)
### zero-shot prompt. This is the "before" row for the thesis results tables:
### the source paper (Chen et al. 2501.17202) reports that off-the-shelf
### audio LLMs, Qwen2-Audio included, cannot do speech quality assessment
### zero-shot, so the expected outcome is poor MOS correlation / hallucinated
### captions. Metrics flow through the identical evaluate.py path as every
### fine-tuned row, keeping the baseline apples-to-apples.
###
### Decoding is GREEDY for reproducibility (no sampling variance in a baseline).
### The model is pulled fresh from the Hub on first load (not previously cached
### on the node), so HF cache + auth from the preamble must be working.
###
### Mem check (per-core rusage rule): rusage[mem=32GB] x -n 4 = 128 GB total,
### well within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_zeroshot_baseline.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-zeroshot-baseline
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/eval_zeroshot_baseline_%J.out
#BSUB -e logs/eval_zeroshot_baseline_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

# HF auth: needed because Qwen2-Audio-7B-Instruct is downloaded fresh here.
# Prefer the env var, fall back to the cached login token.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

# Standard MOS test sets: 3 out-of-domain (FOR/LIVE/P501) + 1 in-domain.
DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
    "data/processed/eval/test_nisqa_indomain.json"
)

# --zero-shot:
#   * defaults the model to the off-the-shelf Qwen/Qwen2-Audio-7B-Instruct
#   * swaps the bare PROMPT_TEMPLATE for the instructed, non-leaking prompt
#   * scores with the identical MAE/MSE/BLEU/ROUGE/BERTScore code path
uv run python scripts/eval/evaluate.py eval-mos \
    --zero-shot \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/zeroshot/qwen2audio_instruct_baseline" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --dataset-path "${DATASETS[3]}" \
    --batch-size 8 \
    --greedy \
    --max-new-tokens 300

echo "=========================================="
echo "Zero-shot baseline evaluation complete: $(date)"
echo "=========================================="
