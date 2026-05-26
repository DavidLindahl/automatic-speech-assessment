#!/bin/bash
### ============================================================
### DTU HPC — Evaluate paper-style DPO partial (step 200/311) with greedy
### Submit with: bsub < jobs/evaluate/evaluate_dpo_paper_half_h100_step200_greedy.sh
###
### Targets the checkpoint-200/ subdirectory from run 28376220, which was
### killed at step 200 by TERM_MEMLIMIT but had successfully written the
### model weights. The optimizer states (47GB global_step200/) have been
### removed; tokenizer files copied in from the parent dir.
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-step200-greedy
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/eval_dpo_step200_greedy_%J.out
#BSUB -e logs/eval_dpo_step200_greedy_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"
mkdir -p "$EXPERIMENT_DIR/results/evaluation"

DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
    "data/processed/eval/test_nisqa_indomain.json"
)

uv run python scripts/eval/evaluate.py eval-mos \
    --model-path "$EXPERIMENT_DIR/models/dpo_paper_half_h100/checkpoint-200" \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_step200_eval_greedy" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --dataset-path "${DATASETS[3]}" \
    --batch-size 8 \
    --greedy \
    --max-new-tokens 150

uv run python scripts/diagnostics/dpo_sanity_check.py \
    "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_step200_eval_greedy"

echo "=========================================="
echo "Greedy DPO step-200 evaluation complete: $(date)"
echo "=========================================="
