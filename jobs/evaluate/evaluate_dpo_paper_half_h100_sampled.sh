#!/bin/bash
### ============================================================
### DTU HPC — Evaluate paper-style DPO with sampled decoding
### Submit with: bsub < jobs/evaluate/evaluate_dpo_paper_half_h100_sampled.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-paper-half-sampled
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/eval_dpo_paper_half_sampled_%J.out
#BSUB -e logs/eval_dpo_paper_half_sampled_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

MODEL_NAME="dpo_paper_half_h100"
DECODE_MODE="sampled"
MODEL_CATEGORY="dpo"

source "$(dirname "$0")/../_lib/templates/evaluate_mos.sh"
