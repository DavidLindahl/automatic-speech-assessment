#!/bin/bash
### ============================================================
### DTU HPC — SFT Temporal on NISQA-SIM localized degradation mixes
### Submit with: bsub < jobs/sft/sft_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_%J.out
#BSUB -e logs/sft_temporal_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

OUTPUT_DIR="data/processed/temporal/nisqa_sim_mix_lowmos_active_max_mos3"
TRAIN_JSON="data/processed/temporal/train_nisqa_temporal_mix_max_mos3_localized.json"

echo "Dataset  : $OUTPUT_DIR"
echo "JSONL    : $TRAIN_JSON"

if [ ! -f "$OUTPUT_DIR/manifest.csv" ]; then
  echo "Missing manifest: $OUTPUT_DIR/manifest.csv"
  echo "Run: bsub < jobs/train/generate_nisqa_temporal_max.sh"
  exit 1
fi

if [ ! -f "$TRAIN_JSON" ]; then
  echo "Building training JSONL from manifest..."
  uv run python scripts/data/build_nisqa_temporal_json.py \
    --manifest-path "$OUTPUT_DIR/manifest.csv" \
    --mixes-dir "$OUTPUT_DIR" \
    --caption-jsonl data/processed/sft/train_nisqa_llama_10k.json \
    --output-jsonl "$TRAIN_JSON"
fi

torchrun \
    --nproc_per_node=1 \
    scripts/train/supervised-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --use-query-prompt \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-project "Temporal-ALLD" \
    --wandb-run-name "sft-temporal"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
