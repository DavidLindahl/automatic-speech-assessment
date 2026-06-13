#!/bin/bash
### ============================================================
### DTU HPC — Evaluate paper-style DPO with deterministic decoding
### Submit with: bsub < jobs/global/eval/evaluate_dpo_paper_half_h100_greedy.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-paper-half-greedy
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/eval_dpo_paper_half_greedy_%J.out
#BSUB -e logs/eval_dpo_paper_half_greedy_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

MODEL_NAME="dpo_paper_half_h100"
DECODE_MODE="greedy"
MODEL_CATEGORY="dpo"

source /work3/s234817/automatic-speech-assessment/jobs/_lib/templates/evaluate_mos.sh
