#!/bin/bash
### ============================================================
### DTU HPC — Generate DPO pairs from paper-style half-data warmup
### Submit with: bsub < jobs/global/data/generate_dpo_paper_half_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J generate-dpo-paper-half-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/generate_dpo_paper_half_h100_%J.out
#BSUB -e logs/generate_dpo_paper_half_h100_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/data/processed"
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

uv run python scripts/data/generate_dpo_data.py \
    --input-json data/processed/sft/train_nisqa_llama_10k.json \
    --output-json "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_paper_half_h100.json" \
    --model-path "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --data-root data \
    --batch-size 8 \
    --do-sample \
    --temperature 1.1 \
    --top-p 0.9

uv run python scripts/diagnostics/sanity_check_dpo.py \
    "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_paper_half_h100.json"

echo "=========================================="
echo "DPO data generation complete: $(date)"
echo "=========================================="
