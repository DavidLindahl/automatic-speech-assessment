#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Inference Smoke Test
### Submit with: bsub < jobs/test_inference.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J inference-test

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
#BSUB -o logs/inference_test_%J.out
#BSUB -e logs/inference_test_%J.err

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

uv run pytest tests/test_inference.py -v -s

echo "Inference test complete: $(date)"
