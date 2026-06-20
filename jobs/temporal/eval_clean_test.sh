#!/bin/bash
#BSUB -q gpuh100
#BSUB -J eval-clean-expc
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 2:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/eval_clean_expc_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/eval_clean_expc_%J.err
set -euo pipefail
cd /work3/s234817/automatic-speech-assessment
module load cuda/11.8 || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=src TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$PWD/.cache/huggingface"
MODEL="${MODEL:-models/sft_expc_detect}"
uv run python scripts/eval/evaluate_temporal.py \
  --model-path "$MODEL" \
  --dataset-path data/processed/temporal/test_CLEAN_valsim_16k.json \
  --data-root data \
  --output-dir results/evaluation/temporal/sft_expc_detect_CLEAN \
  --batch-size 4 --greedy --temperature 0.0 --max-new-tokens 600 --use-query-prompt
echo "clean eval done: $(date)"
