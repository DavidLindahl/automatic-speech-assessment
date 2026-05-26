#!/bin/sh
### ============================================================
### DTU HPC — Re-eval plus2 and plus3 SFT checkpoints with
### sacrebleu corpus BLEU + sampled decoding (temp=0.7, top_p=0.9).
### Adds in-domain NISQA holdout to FOR/LIVE/P501.
### Submit with: bsub < jobs/evaluate/eval_v2_plus2_plus3.sh
### ============================================================

#BSUB -q gpua10
#BSUB -J eval-v2-plus2-plus3
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/eval_v2_plus2_plus3_%J.out
#BSUB -e logs/eval_v2_plus2_plus3_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8 || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
    "data/processed/eval/test_nisqa_indomain.json"
)

run_eval() {
    local model_path="$1"
    local out_dir="$2"
    echo "=========================================="
    echo "Evaluating $model_path -> $out_dir"
    echo "=========================================="
    uv run python scripts/eval/evaluate.py eval-mos \
        --model-path "$model_path" \
        --output-dir "$out_dir" \
        --dataset-path "${DATASETS[0]}" \
        --dataset-path "${DATASETS[1]}" \
        --dataset-path "${DATASETS[2]}" \
        --dataset-path "${DATASETS[3]}" \
        --batch-size 8 \
        --do-sample \
        --temperature 0.7 \
        --top-p 0.9 \
        --max-new-tokens 150
}

run_eval "models/sft_warmup_plus2epoch_l40s" \
         "results/evaluation/sft_warmup_plus2epoch_l40s_eval_v2"

run_eval "models/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus4" \
         "results/evaluation/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus4_eval_v2"

echo "=========================================="
echo "Eval v2 complete: $(date)"
echo "=========================================="
