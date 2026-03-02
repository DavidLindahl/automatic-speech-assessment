#!/bin/sh
### ============================================================
### DTU HPC — DeepSpeed ZeRO-2 Smoke Test (2 GPUs)
### Submit with: bsub < jobs/test_deepspeed.sh
### ============================================================

#BSUB -q gpul40s
#BSUB -J ds-test
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o logs/ds_test_%J.out
#BSUB -e logs/ds_test_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

export PYTHONUNBUFFERED=1

# DeepSpeed needs torchrun to launch across 2 GPUs
torchrun \
    --nproc_per_node=2 \
    tests/test_deepspeed.py

echo "Test complete: $(date)"