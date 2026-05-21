#!/bin/bash
### ============================================================
### DTU HPC — DPO LR-1e-6 ablation WITH the delimiter fix, 1x H100
### Submit with: bsub < jobs/train/dpo_paper_half_h100_lr1e6_delimiterfix.sh
###
### One-variable ablation of the collapsed delimiter-fix run 28484104:
### LR 5e-6 -> 1e-6, everything else held identical (same SFT base,
### dataset, beta 0.4, 1 epoch, batch 2, grad-accum 16, delimiter fix).
###
### WHY: 28484104 trained cleanly (loss 0.05, margins 8.6, accuracies
### 1.0) but BOTH greedy (28484786) and sampled (28484895) eval showed a
### degenerate repetition collapse: every output is " speech" x60,
### P(' speech') = 0.99 at step 0. The collapse is in the policy
### distribution, not the decoder (sampling does not escape it). The
### label-mask probe confirmed the supervised span is correct (first
### token "This"/"The", clean after the "\n"), so the mask is NOT the
### bug. With a clean position-0 signal the run still collapsed onto an
### off-distribution token -> DPO over-optimization. LR is the lever:
### the only DPO run that ever produced real captions, April 27
### dsat5nxi, used LR 1e-6. The earlier LR-1e-6 ablation 28483786 was
### killed because it predated the delimiter fix; this one has it.
###
### COLLAPSE-ONSET CURVE: --save-intermediate saves a LOCAL checkpoint
### every 100 steps (save_total_limit 4, no Hub push). Evaluating each
### checkpoint shows at what step LR 1e-6 collapses (or whether it does
### not) — the actual scientific finding.
###
### Memory: -n 4 x rusage[mem=64GB] = 256 GB total. H100 nodes have
### ~720 GB system memory, so this fits.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-lr1e6-delimiterfix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_lr1e6_delimiterfix_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_lr1e6_delimiterfix_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free for the checkpoints.
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

# DPO LR-1e-6 ablation with the delimiter fix. --save-intermediate writes
# a local checkpoint every 100 steps (keep last 4) so the collapse-onset
# curve can be measured. No --hub-model-id -> Hub push disabled.
torchrun --nproc_per_node=1 src/asa/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_paper_half_h100_lr1e6_delimiterfix" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --save-intermediate \
    --save-steps 100 \
    --save-total-limit 4 \
    --wandb-run-name "dpo-paper-half-h100-lr1e6-delimiterfix"

echo "=========================================="
echo "DPO LR-1e-6 delimiter-fix training complete: $(date)"
echo "=========================================="
