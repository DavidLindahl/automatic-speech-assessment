#!/bin/bash
### ============================================================
### DTU HPC — Generate ALLD/DPO pairs from the gc-PLAIN SFT model
### chosen = the gold global-caption temporal target (correct <|s|>
### interval + MOS caption); rejected = the plain SFT model's own
### temperature-sampled output (which the eval showed collapses to a
### near-constant interval). Paper-faithful: rejected is the policy's own
### sampling distribution. Output is a small JSONL (text), quota-light.
### Submit with: bsub < jobs/train/generate_dpo_gc_plain.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J gen-dpo-gc-plain
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/gen_dpo_gc_plain_%J.out
#BSUB -e logs/gen_dpo_gc_plain_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/data/processed/dpo"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

INPUT_JSON="data/processed/temporal/train_nisqa_temporal_global_caption_aug.json"
OUTPUT_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_gc_plain.json"
MODEL_PATH="$EXPERIMENT_DIR/models/sft_temporal_gc_plain_h100"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Input    : $INPUT_JSON (13,495 gc-plain targets)"
echo "Model    : $MODEL_PATH (plain SFT policy)"
echo "Output   : $OUTPUT_JSON"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no plain SFT checkpoint at $MODEL_PATH"
  exit 1
fi

uv run python scripts/data/generate_dpo_data.py \
    --input-json "$INPUT_JSON" \
    --output-json "$OUTPUT_JSON" \
    --model-path "$MODEL_PATH" \
    --data-root data \
    --batch-size 8 \
    --do-sample \
    --temperature 1.1 \
    --top-p 0.9

echo "=========================================="
echo "DPO data generation (gc-plain) complete: $(date)"
echo "=========================================="
