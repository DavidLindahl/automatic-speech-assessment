#!/bin/bash
### ============================================================
### DTU HPC — Upload SFT AB Model to Hugging Face
### Submit with: bsub < jobs/upload_sft_ab.sh
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-ab
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_ab_%J.out
#BSUB -e logs/upload_sft_ab_%J.err

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

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path='./models/sft_ab',
    repo_id='Leng2beat/speech-quality-assessement-qwen2audio-full-sft-ab',
    repo_type='model',
    num_workers=8
)
"

echo "Upload complete: $(date)"
