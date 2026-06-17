#!/bin/bash
### ============================================================
### DTU HPC — Global ALLD with a 5-DIMENSION (discontinuity) text reference.
### Submit with: bsub < jobs/global/alld/dpo_full_sft_paired_lr5e6.sh
###
### Ablation (\autoref{sec:results-ablation-dis}): does giving the frozen
### text-reference LLM the NISQA discontinuity (`dis`) score, on top of the
### {mos, noi, col, loud} it already sees, change the ALLD outcome? This is a
### ONE-VARIABLE ablation of the paper-faithful 4-dim run
### dpo_full_sft_paired_lr1e6 (28557823):
###   - SAME policy (sft_full_paper_h100), SAME chosen/rejected pairs,
###     SAME hypers (LR 1e-6, beta 0.4, 1 epoch, batch 2, grad-accum 16).
###   - The ONLY differences: data is train_dpo_full_sft.json (the 4-dim
###     file with `dis` joined onto every record, identical pairs, built by
###     DPODataset feed the 5-dim reference prompt {mos, noi, col, dis, loud}
###     via build_expert_prompt_MOS_DIS. The audio policy stream is unchanged.
### So any delta vs the 4-dim baseline is attributable to the reference seeing
### `dis`, nothing else.
###
### Save: --final-save-only to the public Hub (one ~16 GB model), push then delete
### locally after eval. Memory audit (LSF rusage[mem] is PER-CORE):
### 64GB x 4 cores = 256 GB total, fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-full-sft-paired-lr5e6
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_lr5e6_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_lr5e6_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_full_sft.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_full_paper_h100"
HUB_REPO="Leng2beat/dpo-full-sft-paired-lr5e6"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing 5-dim DPO dataset: $TRAIN_JSON"
    echo "Build it: python scripts/data/join_dis_into_dpo.py --in-json data/processed/dpo/train_dpo_full_sft.json --csv data/raw/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv --out-json $TRAIN_JSON"
    exit 1
fi
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: Full-SFT policy not found at $POLICY_BASE"
    echo "Restore from Hub: hf download Leng2beat/SFT_Global_Full --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_full_sft_paired_lr5e6" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --hub-model-id "$HUB_REPO" \
    --no-hub-private \
    --final-save-only \
    --wandb-run-name "dpo-full-sft-paired-lr5e6"

echo "=========================================="
echo "DPO Full-SFT 5-dim (discontinuity reference) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
