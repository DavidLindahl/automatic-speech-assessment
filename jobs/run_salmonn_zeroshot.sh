#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/salmonn_zeroshot.yaml}
RUN_ID=${2:-salmonn_zeroshot_$(date +%Y%m%d_%H%M%S)}
DATA_ROOT=${DATA_ROOT:-data}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_ROOT}/raw/NISQA_Corpus" ]]; then
  echo "ERROR: Missing NISQA corpus at ${DATA_ROOT}/raw/NISQA_Corpus" >&2
  echo "Set DATA_ROOT to a location that contains raw/NISQA_Corpus." >&2
  exit 1
fi

uv run salmonn-bench run-mos \
  --config-path "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --dataset-path data/processed/test_FOR.json \
  --dataset-path data/processed/test_LIVE.json \
  --dataset-path data/processed/test_P501.json \
  --run-id "${RUN_ID}" \
  --output-dir results/salmonn_zeroshot

if [[ -f data/processed/train_nisqa_abtest_llama_10k.json ]]; then
  uv run salmonn-bench run-ab \
    --config-path "${CONFIG_PATH}" \
    --data-root "${DATA_ROOT}" \
    --dataset-path data/processed/train_nisqa_abtest_llama_10k.json \
    --run-id "${RUN_ID}" \
    --output-dir results/salmonn_zeroshot
fi

echo "Completed run: ${RUN_ID}"
