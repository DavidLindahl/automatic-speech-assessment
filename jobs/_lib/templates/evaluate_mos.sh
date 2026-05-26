# Shared evaluation template for SFT or DPO checkpoints on MOS test sets.
#
# This is *sourced* by a thin driver script after the preamble has run.
# Drivers set env vars; this template reads them and runs the eval.
#
# Required env vars:
#   MODEL_NAME      — checkpoint folder name under $EXPERIMENT_DIR/models/
#                     (also names the output dir under results/evaluation/$MODEL_CATEGORY/)
#   DECODE_MODE     — "greedy" | "sampled"
#   MODEL_CATEGORY  — "dpo" | "sft" | "temporal" (where the output dir lands)
#
# Optional env vars:
#   DATASETS      — bash array (set by the driver). Defaults to the
#                   4-test-set NISQA bundle below.
#   BATCH_SIZE    — generation batch size (default: 8)
#   MAX_NEW_TOKENS — generation budget (default: 150)
#   TEMPERATURE   — sampling temperature when DECODE_MODE=sampled (default: 0.7)
#   TOP_P         — nucleus top-p when DECODE_MODE=sampled (default: 0.9)
#   RUN_SANITY    — "1" to run scripts/diagnostics/dpo_sanity_check.py after eval (default: 1)
#
# Drivers should `set -euo pipefail` themselves before sourcing the preamble.

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${DECODE_MODE:?DECODE_MODE is required (greedy|sampled)}"
: "${MODEL_CATEGORY:?MODEL_CATEGORY is required (dpo|sft|temporal)}"

case "$MODEL_CATEGORY" in
    dpo|sft|temporal) ;;
    *)
        echo "ERROR: MODEL_CATEGORY must be one of dpo|sft|temporal, got '$MODEL_CATEGORY'" >&2
        exit 1
        ;;
esac

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
RUN_SANITY="${RUN_SANITY:-1}"

if [ -z "${DATASETS+x}" ]; then
    DATASETS=(
        "data/processed/eval/test_FOR.json"
        "data/processed/eval/test_LIVE.json"
        "data/processed/eval/test_P501.json"
        "data/processed/eval/test_nisqa_indomain.json"
    )
fi

OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/${MODEL_CATEGORY}/${MODEL_NAME}_eval_${DECODE_MODE}"

case "$DECODE_MODE" in
    greedy)
        DECODE_FLAGS=(--greedy)
        ;;
    sampled)
        DECODE_FLAGS=(--do-sample --temperature "$TEMPERATURE" --top-p "$TOP_P")
        ;;
    *)
        echo "ERROR: DECODE_MODE must be 'greedy' or 'sampled', got '$DECODE_MODE'" >&2
        exit 1
        ;;
esac

dataset_args=()
for ds in "${DATASETS[@]}"; do
    dataset_args+=(--dataset-path "$ds")
done

uv run python scripts/eval/evaluate.py eval-mos \
    --model-path "$EXPERIMENT_DIR/models/$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    "${dataset_args[@]}" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${DECODE_FLAGS[@]}"

if [ "$RUN_SANITY" = "1" ]; then
    uv run python scripts/diagnostics/dpo_sanity_check.py "$OUTPUT_DIR"
fi

echo "=========================================="
echo "Evaluation complete ($DECODE_MODE): $(date)"
echo "  model:  $EXPERIMENT_DIR/models/$MODEL_NAME"
echo "  output: $OUTPUT_DIR"
echo "=========================================="
