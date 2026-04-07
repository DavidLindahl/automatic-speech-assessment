#!/bin/bash
### ============================================================
### DTU HPC — Upload SFT Full Model to Hugging Face
### Submit with: bsub < jobs/upload_sft_full.sh
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-full
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_full_%J.out
#BSUB -e logs/upload_sft_full_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

huggingface-cli upload \
    Leng2beat/speech-quality-assessement-qwen2audio-sft-full \
    ./models/sft_full \
    . \
    --repo-type model \
    --commit-message "Upload SFT full model"

echo "Upload complete: $(date)"
