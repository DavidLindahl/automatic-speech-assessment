# Shared LSF preamble for ASA jobs on the DTU HPC.
#
# Source this AFTER the #BSUB header block (LSF reads BSUB directives from
# the file header, so they cannot live in a sourced file).
#
# Usage:
#   #!/bin/bash
#   #BSUB -q gpuh100
#   #BSUB ...                                    # job-specific BSUB block
#   source "$(dirname "$0")/../_lib/preamble.sh"
#   torchrun ...                                 # job-specific command
#
# Sets up: strict mode, EXPERIMENT_DIR, cwd, logs/ + models/, CUDA module,
# venv activation, PYTHONUNBUFFERED, TRITON_CACHE_DIR, HF cache off /work3,
# HF auth preflight, and a startup banner.

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

# HF cache off /work3 to keep quota free for checkpoints.
# Prefer node-local /scratch, then /tmp, fall back to /work3 with a warning.
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    echo "WARN: no node-local scratch writable; HF cache stays on /work3 (quota risk)"
    export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# HF auth: prefer env var if set, otherwise rely on cached login from
# `huggingface-cli login`. Fail fast if neither is available so we don't
# train for hours and then silently fail to push.
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "HF auth: using HF_TOKEN env var"
elif [ -f "$HOME/.cache/huggingface/token" ] || [ -f "$HOME/.cache/huggingface/stored_tokens" ]; then
    echo "HF auth: using cached login from ~/.cache/huggingface/"
else
    echo "ERROR: no HF auth available. Either export HF_TOKEN or run 'huggingface-cli login' on the HPC."
    exit 1
fi

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-unknown}"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi || true
