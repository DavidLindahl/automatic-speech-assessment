#!/bin/bash
### ============================================================
### DTU HPC — DPO/ALLD smoke test for the prompt-delimiter fix, 1x H100
### Submit with: bsub < jobs/train/dpo_delimiter_smoke_h100.sh
###
### Tests the EOS-collapse fix: PROMPT_TEMPLATE now ends with "\n" so the
### first response word ("This"/"The") is a standalone supervised token
### instead of being BPE-merged into the prompt tail. Confirmed by the
### probe_collator_labels.py output (first supervised token = "This"/"The").
###
### 1024 samples, batch 1, grad-accum 16 = 64 optimizer steps. Deliberately
### larger than the old 256-sample / 16-step smoke (28481618), which was too
### short to show the EOS rise. The discriminator is P(EOS) at step 0 from
### diagnose_dpo_empty_output.py run on the resulting checkpoint.
###
### Memory audit (LSF rusage[mem] is PER-CORE): -n 4 x rusage[mem=64GB]
### = 256 GB total. H100 nodes have ~720 GB system memory, so this fits.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-delimiter-smoke
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 3:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_delimiter_smoke_h100_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_delimiter_smoke_h100_%J.err

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

HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-dpo-delimiter-smoke}"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "=========================================="
nvidia-smi

# 1024 samples, batch 1, grad-accum 16 = 64 optimizer steps. LR 5e-6 so this
# isolates the delimiter fix against the collapsed 5e-6 run 28481644 — one
# variable. The PROMPT_TEMPLATE "\n" delimiter is picked up automatically.
torchrun --nproc_per_node=1 src/asa/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_delimiter_smoke_h100" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --ref-model-id "Qwen/Qwen2-7B" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --max-samples 1024 \
    --batch-size 1 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-delimiter-smoke-h100" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 32 \
    --save-total-limit 1

echo "=========================================="
echo "DPO delimiter smoke complete: $(date)"
echo "=========================================="
