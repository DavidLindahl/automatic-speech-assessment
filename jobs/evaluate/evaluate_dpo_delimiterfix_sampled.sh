#!/bin/bash
### ============================================================
### DTU HPC — Sampled eval of the delimiter-fix full DPO run, 1x H100
### Submit with: bsub < jobs/evaluate/evaluate_dpo_delimiterfix_sampled.sh
###
### The greedy eval (28484786) of dpo_paper_half_h100_delimiterfix showed
### a degenerate repetition collapse: every output is " speech" x60,
### P(' speech') = 0.99 at step 0. The chosen captions in the training
### data never start with " speech" (8374x "This", 1605x "The"), so the
### collapse is DPO over-optimization, not data-shaped.
###
### This sampled eval (T=0.7, top-p=0.9) is the discriminating test:
###   - if it produces real captions -> the model is overconfident, not
###     collapsed; greedy hits an absorbing fixed point sampling escapes.
###   - if it is also " speech speech..." -> the policy distribution
###     itself collapsed; DPO over-amplification is confirmed.
###
### Memory: -n 4 x rusage[mem=32GB] = 128 GB total. Fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-delimiterfix-sampled
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/eval_dpo_delimiterfix_sampled_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/eval_dpo_delimiterfix_sampled_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free.
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    echo "WARN: no node-local scratch; HF cache stays on /work3 (quota risk)"
    export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

MODEL="$EXPERIMENT_DIR/models/dpo_paper_half_h100_delimiterfix"

DATASETS=(
    "data/processed/test_FOR.json"
    "data/processed/test_LIVE.json"
    "data/processed/test_P501.json"
    "data/processed/test_nisqa_indomain.json"
)

uv run python src/asa/evaluate.py eval-mos \
    --model-path "$MODEL" \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_delimiterfix_eval_sampled" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --dataset-path "${DATASETS[3]}" \
    --batch-size 8 \
    --do-sample \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-new-tokens 150

uv run python scripts/dpo_sanity_check.py \
    "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_delimiterfix_eval_sampled"

echo "=========================================="
echo "Sampled delimiter-fix DPO evaluation complete: $(date)"
echo "=========================================="
