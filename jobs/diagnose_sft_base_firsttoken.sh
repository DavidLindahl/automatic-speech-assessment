#!/bin/bash
### ============================================================
### DTU HPC — Probe the SFT base first-token distribution, 1x H100
### Submit with: bsub < jobs/diagnose_sft_base_firsttoken.sh
###
### THE DECISIVE PROBE. The delimiter-fix DPO run 28484104 collapsed
### onto token 8806 (" speech"): P(" speech") = 0.99 at step 0, every
### output is " speech" x60, on both greedy (28484786) and sampled
### (28484895) eval. Two probes already ran:
###   - data shortcut RULED OUT: " speech" appears at near-identical
###     rate in chosen (3.20%) vs rejected (3.33%) captions, ratio 0.96.
###   - reference stream CONFIRMED malformed: build_expert_prompt_MOS
###     ends "Output:", so "Output:The" merges and the rejected
###     reference stream's first supervised token is " synthesized",
###     not "The" — a misaligned position-0 DPO reward term.
###
### This probe answers the remaining question: is the SFT base
### models/sft_warmup_paper_half_h100 ALREADY biased toward " speech"
### before DPO ever runs? diagnose_dpo_empty_output.py prints the top-5
### step-0 token distribution.
###
###   - If 8806 (" speech") is top-1/top-3 with P > 0.1 -> SFT base is
###     contaminated; retrain SFT (with the delimiter fix in the SFT
###     collator) before any new DPO run.
###   - If 8806 is buried (P < 0.01) and top token is "This"/"The" ->
###     SFT is clean; the bug is purely in DPO (reference Output: fix +
###     LR ablation).
###
### Memory: -n 4 x rusage[mem=32GB] = 128 GB total. Fits gpuh100.
### ============================================================

#BSUB -q gpuh100
#BSUB -J diag-sft-firsttoken
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 00:30
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/diag_sft_firsttoken_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/diag_sft_firsttoken_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    export HF_HOME="$PROJECT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

echo ""
echo "########## SFT base: first-token distribution ##########"
echo "Looking for token 8806 (' speech') in the top-5 at step 0."
uv run python scripts/diagnose_dpo_empty_output.py \
    --model models/sft_warmup_paper_half_h100 \
    --dataset data/processed/test_LIVE.json \
    --num 6 \
    --max-new-tokens 60

echo ""
echo "Diagnose complete: $(date)"
