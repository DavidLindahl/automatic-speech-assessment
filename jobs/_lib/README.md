# `jobs/_lib/` — shared LSF infrastructure

## `preamble.sh`

Common setup for every LSF job: strict mode, `EXPERIMENT_DIR`, `cwd`, CUDA
module, venv activation, `PYTHONUNBUFFERED`, `TRITON_CACHE_DIR`, the HF cache
quota-safe fallback (`/scratch` → `/tmp` → `/work3`), HF auth preflight, and
a startup banner.

### How to use

```bash
#!/bin/bash
#BSUB -q gpuh100
#BSUB -J my-job
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/my_job_%J.out
#BSUB -e logs/my_job_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py ...
```

Source the preamble **after** the `#BSUB` header — LSF reads BSUB directives
from the literal file header, so they cannot live in a sourced file.

`jobs/sft/sft_warmup_paper_half_h100.sh` is the reference migration; copy it
when wiring a new script.

## `templates/`

Reusable job bodies for things that come in variants (decode mode, target
checkpoint, eval-time hyperparameters). Drivers stay tiny — set env vars,
source the template.

### `templates/evaluate_mos.sh`

MOS evaluation template for SFT or DPO checkpoints. Drivers set
`MODEL_NAME` and `DECODE_MODE` (`greedy` | `sampled`), the template handles
the rest. Reference drivers:
`jobs/evaluate/evaluate_dpo_paper_half_h100_{greedy,sampled}.sh`.

Knobs (all have defaults): `BATCH_SIZE`, `MAX_NEW_TOKENS`, `TEMPERATURE`,
`TOP_P`, `RUN_SANITY`, plus a `DATASETS` bash array if you need a non-default
test set bundle.

This collapses the greedy/sampled variant pair (and any future temperature
sweep) from ~75 lines apiece down to ~22-line drivers. Existing variant
scripts have not been retrofitted yet — see the existing list under
`jobs/evaluate/`.

## `lint-budget.sh`

Hard rule: LSF interprets `#BSUB -R "rusage[mem=X]"` together with
`#BSUB -n N` as **X × N total memory per job**. Getting this wrong has
twice (2026-04-30 and 2026-05-04) triggered angry emails from Sebastian at
DTU IT and killed the entire pending queue.

The linter parses every `jobs/**/*.sh`, computes the total, and exits
non-zero if any script exceeds its target queue's per-node memory cap.

```sh
bash jobs/_lib/lint-budget.sh
```

Queue caps are conservative; verify against `nodestat -F <queue>` if you
need to push higher. Caps live in the `queue_cap()` function at the top of
the script.

Excluded by design: `jobs/_archive/` (frozen history) and `jobs/_lib/`
(this directory itself).

### Known violators (as of Phase 4)

The first run flagged 13 live scripts that already violate the rule. They
have not been fixed in this PR — landing the linter without churning the
queue scripts keeps the blast radius small. Fix-up is a follow-up: pick
the script, decide the intended total memory, divide by `-n` cores, set
both `rusage[mem]` and `-M`. Then re-run the linter to confirm.
