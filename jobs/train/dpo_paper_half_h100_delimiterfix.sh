#!/bin/bash
### ============================================================
### DTU HPC — Full DPO/ALLD run with the prompt-delimiter fix, 1x H100
### Submit with: bsub < jobs/train/dpo_paper_half_h100_delimiterfix.sh
###
### First full DPO run after the EOS-collapse fix. PROMPT_TEMPLATE now ends
### with "\n" (commit a007248) so the first response word is a standalone
### supervised token. The 1024-sample delimiter smoke (28483856) confirmed
### the fix: P(EOS) at step 0 dropped from 0.64-0.79 to 0.0000, the model
### generates full captions again (diag 28484057).
###
### Hub push DISABLED: the Leng2beat HF account is out of private-repo
### storage, so push_to_hub 403s and leaves the checkpoint without its
### processor files. With no --hub-model-id the run does a single clean
### trainer.save_model() at the end (save_strategy="no"). All-or-nothing,
### but the prior full run 28481644 completed training in 1h46m with no
### intermediate save needed. Same setup.
###
### Memory audit (LSF rusage[mem] is PER-CORE): -n 4 x rusage[mem=64GB]
### = 256 GB total. H100 nodes have ~720 GB system memory, so this fits.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-paper-half-delimiterfix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_delimiterfix_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_delimiterfix_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free for the checkpoint.
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

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "=========================================="
nvidia-smi

# Full DPO: paper settings (LR 5e-6, beta 0.4, 1 epoch) on the cleaned
# 9979-row dataset. The PROMPT_TEMPLATE "\n" delimiter fix is picked up
# automatically. No --hub-model-id -> push_to_hub disabled, single clean
# local save at the end.
torchrun --nproc_per_node=1 src/asa/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_paper_half_h100_delimiterfix" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-paper-half-h100-delimiterfix"

echo "=========================================="
echo "DPO training complete: $(date)"
echo "=========================================="
