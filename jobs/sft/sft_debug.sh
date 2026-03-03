#!/bin/sh
### ============================================================
### DTU HPC LSF debug job — Qwen2-Audio SFT (100 samples)
### Single L40S 48GB — fast availability, OOM-safe at batch=1
### Submit with: bsub < jobs/sft_debug.sh
### ============================================================

### -- Queue: L40S 48GB (starts quickly, no reservation issues) --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J qwen2-sft-debug

### -- CPU cores (min 4 per GPU, single GPU) --
#BSUB -n 4

### -- 1 GPU in exclusive mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- No GPU model constraint: let scheduler pick any available L40S --

### -- All cores on one node --
#BSUB -R "span[hosts=1]"

### -- System memory --
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB

### -- Walltime: 1 hour is plenty for a debug run --
#BSUB -W 1:00

### -- Output / error files --
#BSUB -o logs/debug_%J.out
#BSUB -e logs/debug_%J.err

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

# Single-GPU debug run — no DeepSpeed, 100 samples, 1 epoch
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    src/asa/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --dataset-type mos \
    --output-dir results/sft-debug \
    --batch-size 1 \
    --epochs 1 \
    --lr 1e-5 \
    --gradient-accumulation-steps 1 \
    --bf16 \
    --max-samples 100 \
    --deepspeed ""

echo "Debug run complete: $(date)"
