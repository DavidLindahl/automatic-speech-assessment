#!/bin/sh
### ============================================================
### DTU HPC — SFT Temporal Mix (max_mos3) on A40 x2 with CPU offload
### Submit with: bsub < jobs/sft/sft_temporal_max_mos3_a40.sh
### ============================================================

#BSUB -q gpua40
#BSUB -J sft-temporal-max-mos3-a40
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_max_mos3_a40_%J.out
#BSUB -e logs/sft_temporal_max_mos3_a40_%J.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work3/s234817/automatic-speech-assessment}"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

OUTPUT_DIR="data/processed/nisqa_sim_mix_lowmos_active_max_mos3"
MANIFEST_PATH="$OUTPUT_DIR/manifest.csv"
TRAIN_JSON="data/processed/train_nisqa_temporal_mix_max_mos3.json"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started  : $(date)"
echo "Manifest : $MANIFEST_PATH"
echo "JSONL    : $TRAIN_JSON"
echo "=========================================="

nvidia-smi

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Missing manifest: $MANIFEST_PATH"
  echo "Run first: bsub < jobs/train/generate_nisqa_temporal_max.sh"
  exit 1
fi

if [ ! -f "$TRAIN_JSON" ]; then
  echo "JSONL missing; building from manifest."
  uv run scripts/data/build_nisqa_temporal_json.py \
    --manifest-path "$MANIFEST_PATH" \
    --mixes-dir "$OUTPUT_DIR" \
    --caption-jsonl data/processed/train_nisqa_llama_10k.json \
    --output-jsonl "$TRAIN_JSON"
fi

torchrun \
    --nproc_per_node=2 \
    scripts/train/supervised-finetune.py \
    --model-name sft_temporal_max_mos3_a40 \
    --json-path "$TRAIN_JSON" \
    --use-query-prompt \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 100 \
    --wandb-project "Temporal-ALLD" \
    --wandb-run-name "temporal-max-mos3-sft-a40"

echo "Training complete: $(date)"
