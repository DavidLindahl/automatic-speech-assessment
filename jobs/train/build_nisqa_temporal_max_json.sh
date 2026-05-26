#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Build max_mos3 temporal training JSON
### Submit with: bsub < jobs/train/build_nisqa_temporal_max_json.sh
### ============================================================

#BSUB -q hpc
#BSUB -J build-temporal-max-json
#BSUB -n 2
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/build_temporal_max_json_%J.out
#BSUB -e logs/build_temporal_max_json_%J.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work3/s234817/automatic-speech-assessment}"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

OUTPUT_DIR="data/processed/nisqa_sim_mix_lowmos_active_max_mos3"
MANIFEST_PATH="$OUTPUT_DIR/manifest.csv"
TRAIN_JSON="data/processed/temporal/train_nisqa_temporal_mix_max_mos3.json"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "Manifest : $MANIFEST_PATH"
echo "Output   : $TRAIN_JSON"
echo "=========================================="

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Missing manifest: $MANIFEST_PATH"
  echo "Run first: bsub < jobs/train/generate_nisqa_temporal_max.sh"
  exit 1
fi

uv run scripts/data/build_nisqa_temporal_json.py \
  --manifest-path "$MANIFEST_PATH" \
  --mixes-dir "$OUTPUT_DIR" \
  --caption-jsonl data/processed/sft/train_nisqa_llama_10k.json \
  --output-jsonl "$TRAIN_JSON"

echo "Rows in JSONL:"
wc -l "$TRAIN_JSON"

echo "Done: $(date)"
