#!/bin/sh
### ============================================================
### DTU HPC LSF job script — DPO A/B Training Qwen2-Audio
### Submit with: bsub < jobs/train/dpo_ab.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpua40

### -- Job name --
#BSUB -J qwen2-audio-dpo-ab

### -- CPU cores (min 4 per GPU) --
#BSUB -n 8

### -- GPUs --
#BSUB -gpu "num=2:mode=exclusive_process"

### -- Single node --
#BSUB -R "span[hosts=1]"

### -- System memory --
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB

### -- Walltime --
#BSUB -W 24:00

### -- Output / error logs --
#BSUB -o logs/dpo_ab_%J.out
#BSUB -e logs/dpo_ab_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p models/dpo_ab_final

module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

export HF_HOME="$PROJECT_DIR/.cache/huggingface"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi



torchrun \
    --nproc_per_node=2 \
    src/asa/dpo-finetune-ab.py \
    --model-name dpo_ab_final \
    --model-id "models/sft_warmup_ab" \
    --json-path "data/processed/train_dpo_abtest_10k.json" \
    --data-root "data" \
    --batch-size 2 \
    --epochs 2 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 8 \
    --bf16 \
    --deepspeed "configs/ds_zero2.json" \
    --wandb-run-name "dpo-ab-10k-2ep"

echo "=========================================="
echo "DPO A/B Training complete: $(date)"
echo "=========================================="
