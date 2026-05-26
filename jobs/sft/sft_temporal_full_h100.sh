#!/bin/bash
### ============================================================
### DTU HPC — Temporal SFT Full-ft on full NISQA-SIM mix JSONL, 1x H100
### Mirrors sft_full_paper_h100.sh; trains the thesis-deliverable
### temporal model (time-localized degradation captions on NISQA-SIM mixes).
### Local-only save (no HF Hub); final ~16 GB checkpoint to /work3.
### Submit with: bsub < jobs/sft/sft_temporal_full_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-full-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_full_h100_%J.out
#BSUB -e logs/sft_temporal_full_h100_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free for the final checkpoint.
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

# Temporal training data — full NISQA-SIM mix JSONL (built by build_nisqa_temporal_json.py).
TRAIN_JSON="data/processed/train_nisqa_temporal_mix_max_mos3.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing temporal training JSONL: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/build_nisqa_temporal_max_json.sh"
    exit 1
fi

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Dataset  : $TRAIN_JSON"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

torchrun --nproc_per_node=1 src/asa/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_full_h100" \
    --use-query-prompt \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-project "Temporal-ALLD" \
    --wandb-run-name "sft-temporal-full-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
