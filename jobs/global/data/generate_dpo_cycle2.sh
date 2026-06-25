#!/bin/bash
### ============================================================
### DTU HPC — Generate DPO pairs for ALLD CYCLE 2
### Mirrors generate_dpo_full_sft.sh; only --model-path and --output-json
### differ. This is the SECOND ALLD cycle: the policy base is no longer the
### SFT, but the cycle-1 paired-DPO model (Hub: Leng2beat/DPO_Global_Full,
### downloaded to models/dpo_full_sft_paired_lr1e6). Rejected captions are
### resampled from THAT DPO model, so cycle 2 is paper-faithful iterative
### ALLD: chosen = LLaMA-3.1 ground-truth (unchanged), rejected = the current
### best policy's own sampling distribution.
###
### Produces train_dpo_cycle2.json. The cycle-2 DPO train script
### (dpo_cycle2.sh) consumes it and trains the same DPO model further with
### identical hypers (LR 1e-6, beta 0.4, 1 epoch). The frozen ALLD reference
### stays Qwen2-7B (dpo-finetune.py default) — only the policy base changes.
### Submit with: bsub < jobs/global/data/generate_dpo_cycle2.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J generate-dpo-cycle2
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/generate_dpo_cycle2_%J.out
#BSUB -e logs/generate_dpo_cycle2_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/data/processed/dpo"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

DPO_BASE="$EXPERIMENT_DIR/models/dpo_full_sft_paired_lr1e6"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Policy   : $DPO_BASE (cycle-1 best, Hub: Leng2beat/DPO_Global_Full)"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -d "$DPO_BASE" ]; then
    echo "ERROR: cycle-1 DPO model not found at $DPO_BASE"
    echo "Download it first: hf download Leng2beat/DPO_Global_Full --local-dir $DPO_BASE"
    exit 1
fi

uv run python scripts/data/generate_dpo_data.py \
    --input-json data/processed/sft/train_nisqa_llama_10k.json \
    --output-json "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_cycle2.json" \
    --model-path "$DPO_BASE" \
    --data-root data \
    --batch-size 8 \
    --do-sample \
    --temperature 1.1 \
    --top-p 0.9

echo "=========================================="
echo "DPO cycle-2 data generation complete: $(date)"
echo "=========================================="
