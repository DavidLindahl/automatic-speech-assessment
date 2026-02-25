#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Qwen2-Audio Supervised Fine-Tuning
### Submit with: bsub < jobs/sft_hpc.sh
### ============================================================

### -- Queue: L40S 48GB (bf16 + DeepSpeed ZeRO-2) --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J qwen2-sft

### -- CPU cores (min 4 per GPU required by DTU HPC policy) --
#BSUB -n 8

### -- Request 2 GPUs in exclusive mode --
#BSUB -gpu "num=2:mode=exclusive_process"

### -- All cores on one node (required for shared-memory DeepSpeed) --
#BSUB -R "span[hosts=1]"

### -- System memory (audio preprocessing + model loading) --
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB

### -- Walltime (max 24h on GPU queues) --
#BSUB -W 24:00

### -- Output / error files (%J = job id) --
#BSUB -o logs/sft_%J.out
#BSUB -e logs/sft_%J.err

### -- Email notifications (uncomment and fill in your address) --
##BSUB -u s234817@dtu.dk
##BSUB -B
##BSUB -N

# -- end of LSF options --

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

# Create log dir if needed
mkdir -p logs

# Load CUDA module
module load cuda/11.8

# Activate project virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

# Run fine-tuning with Accelerate + DeepSpeed ZeRO-2
accelerate launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    src/asa/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --dataset-type mos \
    --output-dir results/sft-qwen2-mos \
    --batch-size 4 \
    --epochs 2 \
    --lr 1e-5 \
    --gradient-accumulation-steps 4 \
    --bf16 \
    --deepspeed configs/ds_zero2.json

echo "Training complete: $(date)"
