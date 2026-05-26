#!/bin/bash
### ============================================================
### DTU HPC — Evaluate the delimiter-fix full DPO run, 1x H100
### Submit with: bsub < jobs/evaluate/evaluate_dpo_paper_half_h100_delimiterfix.sh
###
### Evaluates models/dpo_paper_half_h100_delimiterfix (full DPO run, job
### 28484104, 311 steps / 1 epoch / loss 0.05, rewards/margins 8.6,
### accuracies 1.0). This is the first full run with the PROMPT_TEMPLATE
### "\n" delimiter fix (commit a007248) that stopped the EOS collapse.
###
### Three checks:
###   1. eval-mos greedy on the 4 test sets -> MAE/MSE/BLEU + captions
###   2. dpo_sanity_check.py -> empty-output / diversity audit
###   3. diagnose_dpo_empty_output.py -> P(EOS) at step 0 + sample captions,
###      to confirm the smoke result (P(EOS) 0.0000) holds on the full run
###      and to see whether the Chinese-output residual persists.
###
### Memory: -n 4 x rusage[mem=32GB] = 128 GB total. Fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-delimiterfix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/eval_dpo_delimiterfix_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/eval_dpo_delimiterfix_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"
mkdir -p "$EXPERIMENT_DIR/results/evaluation"

MODEL="$EXPERIMENT_DIR/models/dpo_paper_half_h100_delimiterfix"

DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
    "data/processed/eval/test_nisqa_indomain.json"
)

echo ""
echo "########## 1. eval-mos greedy ##########"
uv run python scripts/eval/evaluate.py eval-mos \
    --model-path "$MODEL" \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_delimiterfix_eval_greedy" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --dataset-path "${DATASETS[3]}" \
    --batch-size 8 \
    --greedy \
    --max-new-tokens 150

echo ""
echo "########## 2. sanity check (empty/diversity) ##########"
uv run python scripts/diagnostics/dpo_sanity_check.py \
    "$EXPERIMENT_DIR/results/evaluation/dpo_paper_half_h100_delimiterfix_eval_greedy"

echo ""
echo "########## 3. EOS diagnostic (P(EOS) at step 0) ##########"
uv run python scripts/diagnostics/diagnose_dpo_empty_output.py \
    --model "$MODEL" \
    --dataset data/processed/eval/test_LIVE.json \
    --num 5 \
    --max-new-tokens 60

echo "=========================================="
echo "Delimiter-fix DPO evaluation complete: $(date)"
echo "=========================================="
