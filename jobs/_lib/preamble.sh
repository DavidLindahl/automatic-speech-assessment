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
# HF auth detection (warn-only — jobs that don't push to Hub work fine),
# and a startup banner.

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

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

# HF auth: prefer env var if set, otherwise load the cached login token
# from disk and EXPORT it explicitly. This matters because we redirect
# HF_HOME to /scratch above, but the cached token lives at the canonical
# ~/.cache/huggingface/token. Without an explicit export, the Trainer
# subprocess sees no HF_HOME token file and 401s on push_to_hub.
#
# Warn (not fail) when no token is available so jobs that don't push to
# the Hub can still use this preamble.
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    echo "HF auth: using HF_TOKEN env var"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
    echo "HF auth: loaded HF_TOKEN from ~/.cache/huggingface/token"
else
    echo "WARN: no HF auth available (HF_TOKEN unset and no cached login). Hub pushes will fail. Set HF_TOKEN or run 'huggingface-cli login' on the HPC if this job needs Hub access."
fi

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-unknown}"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi || true
