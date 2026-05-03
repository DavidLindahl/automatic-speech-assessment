#!/bin/sh
### ============================================================
### DTU HPC — SFT Temporal Mix (max_mos3)
### Submit with: bsub < jobs/sft/sft_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-h100
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_h100_%J.out
#BSUB -e logs/sft_temporal_h100_%J.err
#BSUB -o logs/sft_temporal_max_mos3_%J.out
#BSUB -e logs/sft_temporal_max_mos3_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

OUTPUT_DIR="data/processed/nisqa_sim_mix_lowmos_active_max_mos3"
TRAIN_JSON="data/processed/train_nisqa_temporal_mix_max_mos3.json"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "Dataset  : $OUTPUT_DIR"
echo "JSONL    : $TRAIN_JSON"
echo "=========================================="

nvidia-smi

if [ ! -f "$OUTPUT_DIR/manifest.csv" ]; then
  echo "Missing manifest: $OUTPUT_DIR/manifest.csv"
  echo "Run: bsub < jobs/train/generate_nisqa_temporal_max.sh"
  exit 1
fi

if [ ! -f "$TRAIN_JSON" ]; then
  echo "Building training JSONL from manifest..."
  uv run python src/asa/build_nisqa_temporal_json.py \
    --manifest-path "$OUTPUT_DIR/manifest.csv" \
    --mixes-dir "$OUTPUT_DIR" \
    --caption-jsonl data/processed/train_nisqa_llama_10k.json \
    --output-jsonl "$TRAIN_JSON"
fi

torchrun \
    --nproc_per_node=1 \
    src/asa/supervised-finetune.py \
    --model-name sft_temporal_max_mos3 \
    --json-path "$TRAIN_JSON" \
    --use-query-prompt \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 100 \
    --wandb-project "Temporal-sft" \
    --wandb-run-name "temporal-sft-h100"

echo "Training complete: $(date)"
