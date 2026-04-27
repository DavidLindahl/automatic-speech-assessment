#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Generate DPO Dataset from +2 SFT model
### ============================================================
#BSUB -q gpuh100
#BSUB -J generate-dpo-plus2-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/generate_dpo_%J.out
#BSUB -e logs/generate_dpo_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs

module load cuda/11.8 || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

uv run python src/asa/generate_dpo_data.py \
    --input-json data/processed/train_nisqa_llama_10k.json \
    --output-json data/processed/train_dpo_10k_plus2epoch_l40s.json \
    --model-path models/sft_warmup_plus2epoch_l40s \
    --batch-size 8 \
    --do-sample \
    --temperature 1.1 \
    --top-p 0.9

echo ""
echo "=========================================="
echo "Dataset generation complete: $(date)"
echo "=========================================="
