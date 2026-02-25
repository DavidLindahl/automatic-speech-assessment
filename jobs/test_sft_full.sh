#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Qwen2-Audio SFT Smoke Test
### Submit with: bsub < jobs/test_sft.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J qwen2-sft-test

### -- CPU cores (min 4 per GPU) --
#BSUB -n 4

### -- Request 1 GPU --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- Single node --
#BSUB -R "span[hosts=1]"

### -- System memory (model loading ~15GB + overhead) --
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB

### -- Walltime (short — just a smoke test) --
#BSUB -W 1:00

### -- Output / error files --
#BSUB -o logs/sft_test_%J.out
#BSUB -e logs/sft_test_%J.err

# -- end of LSF options --

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

python tests/test_sft_training.py

echo "Test complete: $(date)"