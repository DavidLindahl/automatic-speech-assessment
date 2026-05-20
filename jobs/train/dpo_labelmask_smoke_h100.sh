#!/bin/bash
### ============================================================
### DTU HPC — DPO/ALLD smoke test on H100 with the fixed label mask
### Submit with: bsub < jobs/train/dpo_labelmask_smoke_h100.sh
###
### Small diagnostic run on the fix/dpo-label-mask branch. Trains
### ALLD on a 256-sample subset for a handful of steps, on a single
### H100. Purpose: verify the corrected ALLDDPOCollator no longer
### drives the policy to mode-collapse.
### Streams checkpoints to the Hub so /work3 quota stays bounded.
###
### Memory audit: -n 4 x rusage[mem=16GB] = 64 GB total. H100 nodes
### have ~720 GB system memory; 64 GB is well within node limits.
### (-n 4 because the esub rule requires >=4 cores per GPU.)
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-labelmask-smoke
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 3:00
#BSUB -o logs/dpo_labelmask_smoke_h100_%J.out
#BSUB -e logs/dpo_labelmask_smoke_h100_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free for checkpoints.
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    echo "WARN: no node-local scratch writable; HF cache stays on /work3 (quota risk)"
    export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# HF auth.
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF auth: using HF_TOKEN env var"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "HF auth: loaded HF_TOKEN from ~/.cache/huggingface/token"
else
    echo "ERROR: no HF auth available. Either export HF_TOKEN or run 'huggingface-cli login' on the HPC."
    exit 1
fi
export HF_TOKEN

HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-dpo-labelmask-smoke}"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "=========================================="
nvidia-smi

# Smoke run: 256 samples, beta and LR per the paper (the label-mask fix is the
# real change; LR stays at the paper value 5e-6 now that the gradient is clean).
torchrun --nproc_per_node=1 src/asa/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_labelmask_smoke_h100" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --ref-model-id "Qwen/Qwen2-7B" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --max-samples 256 \
    --batch-size 1 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-labelmask-smoke-h100" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 8 \
    --save-total-limit 1

echo "=========================================="
echo "DPO label-mask smoke complete: $(date)"
echo "=========================================="
