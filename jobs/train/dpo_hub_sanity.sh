#!/bin/bash
### ============================================================
### DTU HPC — Sanity check: DPO Hub-streaming save path
### Submit with: bsub < jobs/train/dpo_hub_sanity.sh
###
### Purpose: verify the new save_strategy=steps + push_to_hub plumbing
### works end-to-end before trusting it for a real run. This is NOT a
### training run — it does 4 samples, 1 step, and pushes to a TEST repo.
### ============================================================

#BSUB -q gpul40s
#BSUB -J dpo-hub-sanity
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 01:00
#BSUB -o logs/dpo_hub_sanity_%J.out
#BSUB -e logs/dpo_hub_sanity_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

# Test-only repo. Override with HUB_MODEL_ID=... if you want a different one.
HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/asa-dpo-hub-sanity}"
SANITY_MODEL_DIR="$EXPERIMENT_DIR/models/dpo_hub_sanity"
echo "Hub repo     : $HUB_MODEL_ID"
echo "Local dir    : $SANITY_MODEL_DIR"

# Wipe any prior sanity-run state so rotation/du checks are meaningful.
rm -rf "$SANITY_MODEL_DIR"
mkdir -p "$SANITY_MODEL_DIR"

# Tiny config: 4 samples, save every step, rotate to 1.
# Single GPU, no deepspeed — keep variables to a minimum so a failure
# points at the save plumbing, not at parallelism.
torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$SANITY_MODEL_DIR" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --max-samples 4 \
    --batch-size 1 \
    --epochs 1 \
    --gradient-accumulation-steps 1 \
    --bf16 \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 1 \
    --save-total-limit 1 \
    --wandb-project "qwen2-audio-alld-sanity" \
    --wandb-run-name "dpo-hub-sanity-$LSB_JOBID"

echo "=========================================="
echo "Training step done: $(date)"
echo "=========================================="

# Post-checks: did the save plumbing actually do its job?
echo ""
echo "--- POST-CHECK 1: local model dir size (rotation should keep this bounded) ---"
du -sh "$SANITY_MODEL_DIR" || true
ls -la "$SANITY_MODEL_DIR" || true

echo ""
echo "--- POST-CHECK 2: Hub repo contents (should contain config.json + weights) ---"
python - <<PY
from huggingface_hub import HfApi, list_repo_files
import sys

repo_id = "${HUB_MODEL_ID}"
try:
    files = list_repo_files(repo_id=repo_id)
except Exception as e:
    print(f"FAIL: could not list Hub repo {repo_id}: {e!r}")
    sys.exit(1)

print(f"Hub repo {repo_id} contains {len(files)} files:")
for f in sorted(files):
    print(f"  {f}")

required = {"config.json"}
weights_present = any(f.endswith((".safetensors", ".bin")) for f in files)
processor_present = any("preprocessor" in f or "tokenizer" in f for f in files)
missing = required - set(files)

if missing or not weights_present:
    print(f"FAIL: missing required files. Missing config: {missing}, weights present: {weights_present}")
    sys.exit(1)

print(f"OK: config.json present, weights present, processor files present: {processor_present}")
PY

echo ""
echo "=========================================="
echo "Sanity check complete: $(date)"
echo "If POST-CHECK 1 shows a small dir AND POST-CHECK 2 prints OK,"
echo "the Hub-streaming save path is working. Submit the real job."
echo "=========================================="
