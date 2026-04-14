# Session handoff — DPO collapse fix

## Current branch

`fix/dpo-collapse-warmup` (commit `988a640`, not pushed)

## Context

The `dpo_hf_warmup_fix` run collapsed: 5 unique predictions and MOS=3.2
across 680 eval samples on FOR/LIVE/P501. Root cause is two compounding bugs,
both fixed on this branch at the warmup layer:

1. `ALLDDPOTrainer.get_logprobs` summed log-probs instead of averaging. In the
   cross-modal ALLD setup the reference sees a different prompt than the policy,
   so the standard DPO length cancellation fails and the ~10-word mean length
   gap between `rejected` and `chosen` leaks a free gradient signal. Fixed by
   dividing by the per-sequence unmasked token count.
2. `run_inference` had no sampling knobs, so `generate_dpo_data.py` built the
   DPO rejected set using greedy decoding from an already-collapsed `sft_warmup`
   (only 248 unique rejecteds across 10k pairs, mean rejected is 10 words
   longer than chosen). Fixed by exposing `do_sample` / `temperature` / `top_p`
   and defaulting `generate_dpo_data.py` to the paper Appendix B values
   (`True`, `1.1`, `0.9`).

A separate third issue is that `sft_warmup` was trained on only 5k samples with
no val split and already collapsed to 8 unique outputs before DPO ever ran.
`jobs/sft/sft_warmup.sh` has been retargeted at the full 10k with
`--val-split 0.05`, job renamed `sft-warmup-full`, walltime bumped to 8h.

## Changes already shipped on this branch

- `src/asa/dpo-finetune.py` — length-normalised log-probs in `get_logprobs`
- `src/asa/inference.py` — new `do_sample` / `temperature` / `top_p` params on
  `run_inference` and the `asa-infer` CLI (defaults preserve greedy)
- `src/asa/generate_dpo_data.py` — sampling flags with paper defaults, threaded
  into `run_inference`
- `jobs/sft/sft_warmup.sh` — full 10k corpus, `--val-split 0.05`, 8h walltime,
  renamed job and W&B run

Verification: `uv run ruff check` clean on all four files; all 65 non-slow
tests pass.

## What to do next — ordered

### 1. Submit the warmup job (manual, HPC-only)

On `/work3/s234817/automatic-speech-assessment`:

```bash
git fetch && git checkout fix/dpo-collapse-warmup
bsub < jobs/sft/sft_warmup.sh
bstat                    # watch status
tail -f logs/sft_warmup_<jobid>.out
```

### 2. Evaluate the new warmup checkpoint

Before touching DPO, run the MOS evaluation against FOR, LIVE, and P501 and
confirm the new warmup is healthy:

- strictly more than 100 unique predicted strings across 680 eval samples
  (old warmup: 8)
- predicted MOS spans at least the `[1.4, 4.6]` range seen in `sft_full_eval`
- MAE comparable to or better than `sft_full_eval` (0.67). If warmup MAE is
  worse than 1.0, something is still wrong — do not proceed to DPO.

### 3. Regenerate DPO pairs (post-warmup)

Run `uv run src/asa/generate_dpo_data.py` against the new `models/sft_warmup`.
The defaults now match the paper (`do_sample=True`, `temperature=1.1`,
`top_p=0.9`). Sanity-check the output file:

- unique rejected strings should be in the thousands, not 248
- mean length gap `|rejected - chosen|` in words should shrink substantially

### 4. Tighten DPO hyperparams

Edit `jobs/train/dpo.sh`:

- `--lr 5e-6` → `--lr 1e-6`
- `--epochs 2` → `--epochs 1`
- Pass through `--save-strategy steps --save-steps 100 --save-total-limit 3`
  (requires exposing these CLI flags in `src/asa/dpo-finetune.py` — currently
  `save_strategy="no"` is hardcoded at around line 250)

### 5. Re-enable DPO validation

In `src/asa/dpo-finetune.py`:

- flip `val_split=0` default back to `val_split=0.05` (currently disabled with
  a comment claiming the custom Trainer can't eval — verify; `compute_loss`
  overrides are compatible with standard `Trainer.evaluate`)
- add a lightweight callback that logs on each eval step: mean generated length,
  number of unique generations on a tiny held-out subset. Collapse shows up as
  unique-count → 1 before loss ever goes weird.

### 6. Submit DPO training

```bash
bsub < jobs/train/dpo.sh
```

Watch `rewards/accuracies`, `rewards/margins`, and eval output diversity in W&B
(`speech-quality-DTU-bachelor/qwen2-audio-alld`). Kill early if diversity drops.

### 7. Optional — paper fidelity

Add the `dis` (discontinuity) dimension to the single-MOS expert prompt in
`src/asa/data.py` (`DIMENSION_DEFINITIONS_MOS`, `EXPERT_FEW_SHOT_EXAMPLES_MOS`,
`build_expert_prompt_MOS`, and `DPO_METADATA_FIELDS` in
`src/asa/processed_data.py`). The paper's Appendix B uses all 5 dimensions for
MOS prediction as well. This does not fix collapse; it just aligns the
reimplementation with the paper.

## Known gotchas discovered during this session

- `supervised-finetune.py` hardcodes `save_strategy="no"`. The warmup run only
  persists final weights via the explicit `trainer.save_model(...)` call after
  training ends. There is no "best checkpoint" available for DPO init — you
  always get end-of-run weights. If warmup eval loss starts climbing, you have
  no good checkpoint to roll back to.
- `run_inference` still defaults to greedy. `generate_dpo_data.py` now defaults
  to sampling, but the `asa-infer` CLI and `evaluate.py` still call
  `run_inference` without sampling — that is intentional for evaluation
  (deterministic, reproducible) but worth remembering.
- `results/evaluation/sft_warm_eval/` already shows the warmup was collapsed
  (8 unique preds). Use those numbers as the baseline the retrained warmup must
  beat.
- `wandb/` folder is missing locally — W&B runs live online under
  `speech-quality-DTU-bachelor/qwen2-audio-dpo` and `qwen2-audio-sft-simple`.
