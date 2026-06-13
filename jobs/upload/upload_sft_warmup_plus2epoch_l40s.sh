#!/bin/bash
### ============================================================
### DTU HPC — Upload SFT Warmup +2epoch L40S Model to Hugging Face
### Submit with: bsub < jobs/upload_sft_warmup_plus2epoch_l40s.sh
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-warmup-plus2-l40s
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_warmup_plus2epoch_l40s_%J.out
#BSUB -e logs/upload_sft_warmup_plus2epoch_l40s_%J.err

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

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path="./models/sft_warmup_plus2epoch_l40s",
    repo_id="Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup-plus2epoch-l40s",
    repo_type="model",
    num_workers=8,
)
PY

echo "Upload complete: $(date)"
