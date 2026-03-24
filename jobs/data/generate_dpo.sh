#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Generate DPO Dataset
### Submit with: bsub < jobs/train/generate_dpo.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpul40s

### -- Job name --
#BSUB -J generate-dpo-data

### -- CPU cores (min 4 per GPU) --
#BSUB -n 4

### -- Request 1 GPU --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- Single node --
#BSUB -R "span[hosts=1]"

### -- System memory (model loading ~15GB + overhead) --
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB

### -- Walltime --
#BSUB -W 6:00

### -- Output / error files --
#BSUB -o logs/generate_dpo_%J.out
#BSUB -e logs/generate_dpo_%J.err

# -- end of LSF options --

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs

module load cuda/11.8 || true

source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi



uv run python scripts/data/generate_dpo_data.py \
    --input-json data/processed/train_nisqa_llama_10k.json \
    --output-json data/processed/train_dpo_10k.json \
    --model-path models/sft_warmup \
    --batch-size 8

echo ""
echo "=========================================="
echo "Dataset generation complete: $(date)"
echo "=========================================="
