#!/bin/bash
### ============================================================
### DTU HPC — Evaluate paper-style half-data SFT warmup
### Submit with: bsub < jobs/evaluate/evaluate_sft_warmup_paper_half_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-sft-warmup-paper-half-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/eval_sft_warmup_paper_half_h100_%J.out
#BSUB -e logs/eval_sft_warmup_paper_half_h100_%J.err

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

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
    "data/processed/eval/test_nisqa_indomain.json"
)

uv run python scripts/eval/evaluate.py eval-mos \
    --model-path "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/sft/sft_warmup_paper_half_h100_eval_sampled" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --dataset-path "${DATASETS[3]}" \
    --batch-size 8 \
    --do-sample \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-new-tokens 150

echo "=========================================="
echo "Warmup evaluation complete: $(date)"
echo "=========================================="
