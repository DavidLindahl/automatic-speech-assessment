#!/bin/sh
### ============================================================
### DTU HPC LSF job script — ALLD (DPO) Training Qwen2-Audio
### Submit with: bsub < jobs/train/dpo_alld.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J qwen2-audio-alld

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
#BSUB -o logs/alld_%J.out
#BSUB -e logs/alld_%J.err

set -euo pipefail

# Make sure this path matches your actual project directory
PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p models/alld_final

module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

# Run the new ALLD Finetuning script
torchrun \
    --nproc_per_node=2 \
    src/asa/dpo-finetune.py \
    --model-id "models/sft_warmup" \
    --ref-model-id "Qwen/Qwen2-7B-Instruct" \
    --json-path "data/processed/train_dpo_10k.json" \
    --data-root "data" \
    --output-dir "models/alld_final" \
    --batch-size 1 \
    --epochs 2 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 8 \
    --bf16 \
    --eval-steps 100 \
    --deepspeed "configs/ds_zero2.json" \
    --wandb-run-name "alld-10k-2ep"

echo "=========================================="
echo "ALLD Training complete: $(date)"
echo "=========================================="