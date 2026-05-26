#!/bin/sh
### ============================================================
### DTU HPC — Resume SFT cosine-warmup run from checkpoint-565.
### Original job 28309958 was killed by LSF for memory overrun
### (requested 64 GB, peak 408 GB, avg 178 GB). LSF rusage[mem]
### is per-core, total = mem x n. gpuh100 nodes have 720 GB usable
### but only 360 GB per GPU is available to a single-GPU job, so
### the cap is 360 GB total. Request below: 88 GB x 4 cores = 352
### GB total, just under the per-GPU ceiling. NOTE: original run
### peaked at 408 GB which exceeds this cap, so this script will
### likely OOM unless the deepspeed config is switched to
### ds_zero2_no_offload.json. Walltime 24h. If a checkpoint exists
### Trainer resumes; otherwise it starts from scratch.
###
### Submit with: bsub < jobs/sft/sft_base_5ep_cosine_resume.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-base-5ep-cosine-resume
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=88GB]"
#BSUB -M 88GB
#BSUB -W 24:00
#BSUB -o logs/sft_base_5ep_cosine_resume_%J.out
#BSUB -e logs/sft_base_5ep_cosine_resume_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
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

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-name sft_base_5ep_cosine \
    --json-path data/processed/sft/train_nisqa_llama_indomain.json \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 5 \
    --lr 1e-5 \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --val-split 0.05 \
    --eval-steps 100 \
    --resume-from-checkpoint auto \
    --wandb-run-name "sft-base-5ep-cosine-warmup-resume"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
