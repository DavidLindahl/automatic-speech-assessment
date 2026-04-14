# Session handoff — DPO collapse fix

Written for the agent running on the DTU HPC node
(`/work3/s234817/automatic-speech-assessment`). Everything below is what the
previous session did off-cluster and what still has to happen on-cluster.

## Current branch

`fix/dpo-collapse-warmup` — pushed to origin, two commits on top of `main`:

- `988a640` fix: address DPO mode collapse at length-bias and data-gen layers
- `b89a2ce` docs: add session handoff with post-warmup checklist

Start by syncing:

```bash
cd /work3/s234817/automatic-speech-assessment
git fetch
git checkout fix/dpo-collapse-warmup
git pull
```

## Context — why the previous DPO run was broken

The `dpo_hf_warmup_fix` run in `results/evaluation/dpo_hf_warmup_fix_eval/`
collapsed catastrophically: only **5 unique predicted strings and a single
predicted MOS (3.2) across 680 eval samples** on FOR / LIVE / P501. Diagnosis
(full analysis in the prior chat) identified three compounding problems:

1. **Length bias in the DPO loss.** `ALLDDPOTrainer.get_logprobs` in
   `src/asa/dpo-finetune.py` summed per-token log-probs instead of averaging.
   In standard DPO this partly cancels because policy and reference share the
   prompt, but in ALLD the reference sees the meta-info text prompt while the
   policy sees the audio prompt, so cancellation breaks. Combined with a
   training set where `rejected` is on average **10 words longer** than
   `chosen` (43.6 vs 33.6), this leaked a free gradient signal from length
   alone. Easiest way to minimise the loss was to collapse to a short generic
   string — exactly what happened.

2. **Greedy DPO-data generation.** `src/asa/generate_dpo_data.py` builds
   `rejected` by calling `run_inference` on the `sft_warmup` checkpoint. But
   `run_inference` had **no sampling knobs**, so it was greedy. Sampling
   greedily from an already-collapsed warmup produced only **248 unique
   rejected strings across 10000 pairs** (top rejected string appears 1744×).
   The paper (Appendix B) explicitly uses `temperature=1.1, top_p=0.9` for
   DPO data generation.

3. **SFT warmup was already collapsed.** The old `sft_warmup.sh` trained on
   `--max-samples 5000` for 2 epochs with no val split. The resulting checkpoint
   produces only **8 unique outputs across 680 eval samples** (see
   `results/evaluation/sft_warm_eval/`). Every subsequent DPO pass was
   amplifying a collapsed base.

## Fixes already shipped on this branch

Four files modified, 65 non-slow tests pass, ruff clean:

- **`src/asa/dpo-finetune.py`** — length-normalised log-probs. `get_logprobs`
  now divides by `loss_mask.sum(dim=1).clamp(min=1)`.
- **`src/asa/inference.py`** — `run_inference` now accepts `do_sample`,
  `temperature`, `top_p`. Defaults preserve greedy so `asa-infer` and
  `evaluate.py` behaviour is unchanged. Threaded through the Typer `infer`
  command.
- **`src/asa/generate_dpo_data.py`** — same three flags added, defaults set to
  the paper's Appendix B values (`True`, `1.1`, `0.9`). Defaults here are
  diverse sampling, not greedy.
- **`jobs/sft/sft_warmup.sh`** — `--max-samples 5000` removed, `--val-split 0.05`
  added, walltime 3h → 8h, job renamed `sft-warmup-full`, W&B run
  `sft-warmup-full-10k`.

## What you (the HPC agent) need to do — in order

### Step 1. Submit the warmup retrain

```bash
bsub < jobs/sft/sft_warmup.sh
bstat
```

Watch it with `tail -f logs/sft_warmup_<jobid>.out`. Job trains Qwen2-Audio on
the full 10k NISQA set for 2 epochs on 2× A40 with DeepSpeed Zero-2. Expect
several hours. When it finishes, the checkpoint lands in `models/sft_warmup/`
(same name as before — the old one is overwritten). W&B run name is
`sft-warmup-full-10k` under project `qwen2-audio-sft-simple`.

### Step 2. Evaluate the new warmup before doing anything else

Submit `jobs/evaluate/evaluate-sft-mos.sh` against the new
`models/sft_warmup/` for all three test sets (FOR, LIVE, P501). Results land
in `results/evaluation/sft_warm_eval/` (overwriting old).

**Hard gates before moving on to DPO** (compare against the previous baseline
numbers in the git history of `results/evaluation/sft_warm_eval/`):

- **>100 unique predicted strings** across 680 samples (previous: 8)
- Predicted MOS spans at least `[1.4, 4.6]` (previous: same range but only 8
  distinct values)
- **MAE ≤ 0.75** across the three test sets (previous: ~1.0, full SFT: 0.67)

If any of those fails, stop and report — something else is wrong. Do not
regenerate DPO data from a still-collapsed warmup.

### Step 3. Regenerate the DPO training pairs

From the repo root:

```bash
uv run src/asa/generate_dpo_data.py \
    --input-json data/processed/train_nisqa_llama_10k.json \
    --output-json data/processed/train_dpo_10k.json \
    --model-path models/sft_warmup \
    --data-root data \
    --batch-size 8
```

The defaults now include `--do-sample`, `--temperature 1.1`, `--top-p 0.9`, so
you don't need to pass those explicitly. Note this **overwrites**
`data/processed/train_dpo_10k.json` — if you want to keep the old file for
comparison, rename it first.

**Sanity-check the new file.** Expected improvements over the old pairs:

- Unique `rejected` strings should be in the low thousands, not 248
- Mean word-length gap `len(rejected) - len(chosen)` should drop from ~10
  toward 0
- `chosen == response` should still be true for all 10k records (that part is
  the paper's `y_t`, unchanged)

Quick check:

```python
import json
from collections import Counter
decoder = json.JSONDecoder()
text = open('data/processed/train_dpo_10k.json').read()
records, idx = [], 0
while idx < len(text):
    while idx < len(text) and text[idx] in ' \n\r\t,': idx += 1
    if idx >= len(text): break
    obj, idx = decoder.raw_decode(text, idx)
    records.append(obj)
print(f'records: {len(records)}')
print(f'unique rejected: {len(Counter(r["rejected"] for r in records))}')
import statistics as s
cw = [len(r['chosen'].split()) for r in records]
rw = [len(r['rejected'].split()) for r in records]
print(f'mean chosen words: {s.mean(cw):.1f}, mean rejected words: {s.mean(rw):.1f}')
print(f'mean(rejected - chosen) words: {s.mean([b - a for a, b in zip(cw, rw)]):.1f}')
```

### Step 4. Tighten the DPO hyperparameters and re-enable validation

Before re-submitting DPO, make the following code/config changes. These are
**not yet on the branch** — you have to edit them:

**`src/asa/dpo-finetune.py`**

- Change the `val_split` default from `0` back to `0.05`. The existing comment
  claims eval is disabled because of the custom Trainer, but `compute_loss`
  overrides are compatible with `Trainer.evaluate` — verify once, then re-enable.
- Currently `save_strategy="no"` is hardcoded inside `TrainingArguments(...)`.
  Either expose it via a CLI flag or just change it to `save_strategy="steps"`,
  `save_steps=100`, `save_total_limit=3` so you can rewind if collapse starts.
- Optionally add a simple eval callback that logs, on every eval step, the
  number of unique generations and the mean generated length on a tiny held-out
  subset. Mode collapse shows up as unique-count → 1 before the loss does
  anything visible.

**`jobs/train/dpo.sh`**

- `--lr 5e-6` → `--lr 1e-6`. Cross-modal log-prob gradients are noisier than
  plain DPO.
- `--epochs 2` → `--epochs 1`.
- Bump the job name / W&B run name so logs don't clobber the old run
  (`dpo_hf_warmup_fix` → e.g. `dpo_warmup_v2`).
- If you expose the save flags in the script, pass them through.

### Step 5. Submit DPO

```bash
bsub < jobs/train/dpo.sh
```

Watch W&B project `qwen2-audio-alld` (entity `speech-quality-DTU-bachelor`).
Kill early if `rewards/accuracies` saturates at 1.0 in the first 100 steps
while eval output diversity collapses — that's the length-exploit signature
returning, and means the normalisation didn't fully fix it.

### Step 6. Evaluate the new DPO model

Run the same eval as step 2 but against the new DPO checkpoint. The
comparison the bachelor project cares about is `sft_full_eval` (MAE 0.67) vs.
the new DPO run. A successful DPO should match or beat SFT on MAE **and**
produce BLEU > 10 on the descriptive responses (old run: ~0.005).

### Step 7 (optional, paper fidelity)

Add the `dis` (discontinuity) dimension to the single-MOS expert prompt in
`src/asa/data.py` — paper's Appendix B uses all 5 dimensions even for the MOS
task. Files/symbols to update:

- `DIMENSION_DEFINITIONS_MOS`, `EXPERT_TASK_MOS`, `EXPERT_FEW_SHOT_EXAMPLES_MOS`,
  `build_expert_prompt_MOS` in `src/asa/data.py`
- `DPO_METADATA_FIELDS` in `src/asa/processed_data.py`

This does not fix anything; it just matches the paper exactly.

## Gotchas to remember

- `src/asa/supervised-finetune.py` hardcodes `save_strategy="no"`. The warmup
  job only persists end-of-run weights via `trainer.save_model(...)` — there
  is no "best eval loss" checkpoint to recover. If the retrained warmup still
  looks bad, the only option is to retrain again with a smaller LR, not roll
  back a step.
- `run_inference` still defaults to greedy. That is intentional: `evaluate.py`
  and `asa-infer` need deterministic output. Only `generate_dpo_data.py`
  defaults to sampling.
- The old `results/evaluation/sft_warm_eval/` files show the baseline to beat
  (8 unique preds, MAE ~1.0). The new evaluation will overwrite them — stash
  them somewhere first if you want side-by-side comparison.
- `wandb/` is not checked into the repo; all run history lives at
  `speech-quality-DTU-bachelor/qwen2-audio-sft-simple` (SFT) and
  `speech-quality-DTU-bachelor/qwen2-audio-alld` (DPO).
- DPO and warmup use different queues: warmup is `gpul40s` / L40S 48GB
  (retargeted to `gpua40` in the current script — verify that's still what
  you want before submitting), DPO is `gpua40` / A40 40GB.

## Summary task list

| # | Status | What |
|---|--------|------|
| 1 | todo | `bsub < jobs/sft/sft_warmup.sh` |
| 2 | todo | Evaluate new warmup on FOR/LIVE/P501; gate on MAE ≤ 0.75 and >100 unique preds |
| 3 | todo | Regenerate `train_dpo_10k.json` with sampling |
| 4 | todo | Edit `dpo-finetune.py` (val_split=0.05, save_strategy=steps) + `jobs/train/dpo.sh` (lr 1e-6, epochs 1) |
| 5 | todo | `bsub < jobs/train/dpo.sh` |
| 6 | todo | Evaluate new DPO model; compare against sft_full (MAE 0.67, BLEU 0.10) |
| 7 | optional | Add `dis` dimension to MOS expert prompt for paper fidelity |

Report back with the step 2 and step 6 numbers — those are the gates.
