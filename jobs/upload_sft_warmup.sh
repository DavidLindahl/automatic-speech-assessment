#!/bin/bash
### ============================================================
### DTU HPC — Upload SFT Warmup Model to Hugging Face
### Submit with: bsub < jobs/upload_sft_warmup.sh
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-warmup
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_warmup_%J.out
#BSUB -e logs/upload_sft_warmup_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate

REPO_ID="${HF_REPO_ID:-Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup}"
MODEL_DIR="./models/sft_warmup"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "Repo ID  : $REPO_ID"
echo "Model dir : $MODEL_DIR"
echo "=========================================="

python - <<PY
from huggingface_hub import HfApi

repo_id = "${REPO_ID}"
folder_path = "${MODEL_DIR}"
api = HfApi()
api.upload_large_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    num_workers=8,
)
PY

echo "Upload complete: $(date)"
