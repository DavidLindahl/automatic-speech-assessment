#!/bin/bash
### ============================================================
### DTU HPC — Diagnose why DPO model emits empty output at eval
### Submit with: bsub < jobs/diagnose_dpo_empty.sh
### ============================================================

#BSUB -q gpul40s
#BSUB -J diagnose-dpo-empty
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 00:30
#BSUB -o logs/diagnose_dpo_empty_%J.out
#BSUB -e logs/diagnose_dpo_empty_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3.
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    export HF_HOME="$PROJECT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

echo ""
echo "########## DPO model ##########"
uv run python scripts/diagnose_dpo_empty_output.py \
    --model models/dpo_paper_half_h100 \
    --dataset data/processed/test_LIVE.json \
    --num 3 \
    --max-new-tokens 50

echo ""
echo "########## SFT-warmup model (control) ##########"
uv run python scripts/diagnose_dpo_empty_output.py \
    --model models/sft_warmup_paper_half_h100 \
    --dataset data/processed/test_LIVE.json \
    --num 3 \
    --max-new-tokens 50

echo ""
echo "Diagnose complete: $(date)"
