#!/bin/bash
### ============================================================
### DTU HPC — Temporal SFT Full-ft on full NISQA-SIM mix JSONL, 1x H100
### Mirrors sft_full_paper_h100.sh; trains the thesis-deliverable
### temporal model (time-localized degradation captions on NISQA-SIM mixes).
### Local-only save (no HF Hub); final ~16 GB checkpoint to /work3.
### Submit with: bsub < jobs/sft/sft_temporal_full_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-full-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_full_h100_%J.out
#BSUB -e logs/sft_temporal_full_h100_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

# Temporal training data — full NISQA-SIM mix JSONL (built by build_nisqa_temporal_json.py).
TRAIN_JSON="data/processed/temporal/train_nisqa_temporal_mix_max_mos3.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing temporal training JSONL: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/build_nisqa_temporal_max_json.sh"
    exit 1
fi

echo "Dataset  : $TRAIN_JSON"

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_full_h100" \
    --use-query-prompt \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-project "Temporal-ALLD" \
    --wandb-run-name "sft-temporal-full-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
