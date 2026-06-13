#!/bin/bash
### ============================================================
### DTU HPC — Sample the Phase-1 WINNER (ARM A) once, for all DPO cycles.
### chosen = the gold caption-last anchoroffset target; rejected = ARM A's
### own temperature-sampled output. This single sampling pass is spliced
### three ways downstream (MOS cycle, caption+MOS cycle, and as one of the
### two timestamp-cycle sources) so we pay the GPU cost ONCE.
###
### Policy = models/sft_gc_timelast_timeaudio_h100 (ARM A, t-IoU 0.88). It
### carries the TimeAudio mechanisms, so load_model auto-detects the subclass
### (use_abs_time_embedding/use_time_tokens) -- no flags needed here. The
### <a><f> time tokens are REGULAR added tokens, so they survive
### skip_special_tokens=True decoding and stay in the sampled rejected.
###
### max-new-tokens 300 (caption-LAST puts the timestamp clause at the end of a
### full caption; the default 100 could truncate before the timestamps).
### --use-query-prompt to match ARM A's training prompt.
###
### Mem: rusage[mem=32GB] x -n 4 = 128 GB total, fits gpuh100 (~720 GB).
### Output is text JSONL (quota-light). Submit with:
###   bsub < jobs/train/generate_dpo_armA_sampled.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J gen-dpo-armA-sampled
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 6:00
#BSUB -o logs/gen_dpo_armA_sampled_%J.out
#BSUB -e logs/gen_dpo_armA_sampled_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/data/processed/dpo"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

INPUT_JSON="data/processed/temporal/train_nisqa_temporal_gc_timelast_aug_anchoroffset.json"
OUTPUT_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_sampled.json"
MODEL_PATH="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Input    : $INPUT_JSON (13,495 caption-last anchoroffset targets)"
echo "Model    : $MODEL_PATH (ARM A, t-IoU 0.88, TimeAudio subclass)"
echo "Output   : $OUTPUT_JSON"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no ARM A checkpoint at $MODEL_PATH"
  exit 1
fi
if [ ! -f "$INPUT_JSON" ]; then
  echo "ERROR: missing sample-source dataset $INPUT_JSON"
  exit 1
fi

uv run python scripts/data/generate_dpo_data.py \
    --input-json "$INPUT_JSON" \
    --output-json "$OUTPUT_JSON" \
    --model-path "$MODEL_PATH" \
    --data-root data \
    --batch-size 8 \
    --do-sample \
    --temperature 1.1 \
    --top-p 0.9 \
    --max-new-tokens 300 \
    --use-query-prompt

uv run python scripts/diagnostics/sanity_check_dpo.py "$OUTPUT_JSON"

echo "=========================================="
echo "DPO data generation (ARM A sampled) complete: $(date)"
echo "=========================================="
