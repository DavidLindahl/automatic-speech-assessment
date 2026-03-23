# SALMONN Zero-Shot Rebuild Plan

## Live Status

Last updated: 2026-03-23

Current phase: Phase 7 (smoke benchmark pending model assets)

Completed work in branch:
1. Legacy `src/asa` code removed.
2. New `src/salmonn_bench` package added.
3. Vendored SALMONN runtime and SQA prompt assets added.
4. New SALMONN-focused `pyproject.toml` and `.python-version` set.
5. Zero-shot MOS/A-B CLI and metrics pipeline implemented.
6. CLI and module compile sanity checks pass.
7. Repository minimized to execution-only files for branch switch + `bsub` runs.
8. Runner moved to `jobs/run_salmonn_zeroshot.sh`.
9. SQA prompt JSON normalized to valid JSON (removed trailing comma).
10. `jobs/run_salmonn_zeroshot.sh` now enforces external `DATA_ROOT` for NISQA audio.
11. Added `data/raw` ignore policy with tracked `.gitkeep` so NISQA can be restored locally without being committed.

## 1. Goal and Scope

This plan replaces the current branch implementation with a clean SALMONN-first codebase and benchmarks **zero-shot SALMONN** for speech quality evaluation.

Primary objective:
1. Rebuild the repository so the first working system is a **zero-shot SALMONN evaluator**.
2. Evaluate on our benchmark datasets with reproducible outputs.
3. Keep tuning (LoRA, Q-former training) out of the initial implementation and explicitly defer it to future phases.

Out of scope for initial implementation:
1. Any parameter updates to SALMONN.
2. Distillation, DPO, SFT, or joint training.
3. Frontend/UI work.

---

## 2. Product Definition (What "Done" Means)

The initial rebuild is complete when all items below are true:
1. The branch contains a new minimal project focused on SALMONN inference and evaluation.
2. Running one command performs zero-shot inference + metrics on selected benchmark splits.
3. We produce machine-readable result artifacts and a comparison summary.
4. The run is reproducible from a fresh clone with documented setup steps.

Required output artifacts:
1. `results/salmonn_zeroshot/<run_id>/<dataset>_predictions.jsonl`
2. `results/salmonn_zeroshot/<run_id>/<dataset>_metrics.json`
3. `results/salmonn_zeroshot/<run_id>/summary.md`
4. `results/salmonn_zeroshot/<run_id>/run_config.json`

---

## 3. High-Level Architecture

### 3.1 Core components
1. `src/salmonn_bench/config.py`
2. `src/salmonn_bench/data.py`
3. `src/salmonn_bench/inference.py`
4. `src/salmonn_bench/eval.py`
5. `src/salmonn_bench/cli.py`

### 3.2 Dependency strategy
1. Use a **new `pyproject.toml`** tuned for SALMONN runtime compatibility.
2. Create a clean `.venv` for this branch only.
3. Keep SALMONN runtime deps pinned to known-compatible versions.

### 3.3 Model strategy (zero-shot only)
1. Use SALMONN inference code path.
2. Use base/official checkpoint inference behavior.
3. No training loops, no optimizer, no adapter updates.

---

## 4. Phase Plan

## Phase 0: Safety, Baseline Freeze, and Branch Hygiene

Objective: preserve recoverability before deleting/replacing most code.

Steps:
1. Create a safety tag for current state: `pre-salmonn-rebuild`.
2. Export a quick inventory of current files and important artifacts.
3. Record current benchmark outputs that we may want for comparison.

Deliverables:
1. Git tag for rollback.
2. `reports/migration/pre_rebuild_inventory.md`.

Acceptance criteria:
1. We can recover the old branch state in one command.

---

## Phase 1: Hard Re-scope of Repository Layout

Objective: remove legacy training/eval stack and keep only what supports SALMONN zero-shot benchmarking.

Steps:
1. Remove old `src/asa/*` modules that are irrelevant to SALMONN zero-shot.
2. Remove unused jobs/tests/configs tied to Qwen/ALLD pipelines.
3. Keep only datasets/results/docs required for benchmark continuity.
4. Create new package namespace (`salmonn_bench`).

Target minimal tree:
1. `pyproject.toml`
2. `.python-version`
3. `src/salmonn_bench/*`
4. `jobs/run_salmonn_zeroshot.sh`
5. `configs/salmonn_zeroshot.yaml`
6. `data/` (existing dataset files retained)
7. `README.md`

Deliverables:
1. New minimal skeleton committed.

Acceptance criteria:
1. Repo no longer references legacy Qwen training/inference paths.

---

## Phase 2: New Environment and Dependency Lock

Objective: create a reliable runtime for SALMONN inference.

Steps:
1. Replace `pyproject.toml` dependencies with SALMONN-oriented set.
2. Pin core versions needed by SALMONN runtime path.
3. Recreate `.venv` from scratch.
4. Generate/update lockfile.

Dependency notes:
1. SALMONN upstream references older Transformers/PyTorch combinations.
2. CUDA/GPU environment should be treated as required for full benchmark runs.
3. If needed, split CPU smoke-test profile and GPU full-run profile.

Deliverables:
1. New `pyproject.toml`.
2. Fresh lockfile.
3. Setup instructions in `README.md`.

Acceptance criteria:
1. `uv run python -m salmonn_bench.cli --help` works in clean env.

---

## Phase 3: Integrate SALMONN Runtime Assets

Objective: integrate SALMONN inference logic and model asset wiring.

Steps:
1. Vendor/reference SALMONN inference components required for decode only.
2. Add config fields for:
   - `llama_path`
   - `whisper_path`
   - `beats_path`
   - `ckpt`
3. Add prompt templates for quality tasks (MOS description and A/B preference).
4. Add robust path validation with explicit startup errors.

Deliverables:
1. `configs/salmonn_zeroshot.yaml` template with placeholders.
2. Runtime loader in `src/salmonn_bench/inference.py`.

Acceptance criteria:
1. A single sample inference works end-to-end for MOS prompt.
2. A single sample pair inference works end-to-end for A/B prompt.

---

## Phase 4: Data Contracts and Dataset Adapters

Objective: normalize existing benchmark datasets into SALMONN input format.

Steps:
1. Support current JSONL format used in this project.
2. Implement adapters for:
   - single-audio MOS samples
   - A/B paired-audio samples
3. Normalize audio paths and verify existence before run.
4. Enforce runtime assumptions:
   - 16kHz processing path
   - max 30s truncation behavior

Deliverables:
1. `src/salmonn_bench/data.py` with deterministic parsing.
2. Validation report for skipped/missing files.

Acceptance criteria:
1. Dataset loader produces zero malformed items on target benchmark splits.

---

## Phase 5: Zero-Shot Inference Pipeline

Objective: implement batch inference for MOS and A/B tasks using frozen SALMONN.

Steps:
1. Build `infer-mos` command:
   - input dataset path(s)
   - output predictions JSONL
2. Build `infer-ab` command:
   - paired audio handling
   - output predictions JSONL
3. Add generation controls in config:
   - `max_new_tokens`, `num_beams`, `temperature`, `top_p`
4. Add deterministic run metadata capture:
   - seed
   - checkpoint id/path
   - prompt key

Deliverables:
1. CLI commands in `src/salmonn_bench/cli.py`.
2. Prediction files with stable schema.

Acceptance criteria:
1. Inference job runs on all target datasets without manual edits.

---

## Phase 6: Evaluation Pipeline and Metrics

Objective: compute benchmark metrics from prediction outputs.

Metrics (MOS task):
1. MSE
2. MAE
3. LCC (Pearson)
4. SRCC (Spearman)
5. BLEU (reference response vs generated response)

Metrics (A/B task):
1. Accuracy
2. Per-class accuracy (`A`, `B`, `Tie` if present)
3. BLEU

Steps:
1. Implement robust MOS value extraction from generated text.
2. Implement winner extraction for A/B responses.
3. Save both per-sample results and aggregate metrics.
4. Build one `evaluate` command for MOS and A/B outputs.

Deliverables:
1. `src/salmonn_bench/eval.py`.
2. `<dataset>_metrics.json` and `<dataset>_results.json`.

Acceptance criteria:
1. Metrics are reproducible across reruns (except expected decoding variation when sampling enabled).

---

## Phase 7: Benchmark Execution Matrix

Objective: run the complete zero-shot benchmark suite.

Primary benchmark splits:
1. MOS: `NISQA_VAL_LIVE`, `NISQA_TEST_FOR`, `NISQA_TEST_P501`
2. A/B: project A/B test datasets aligned to the same corpora

Run order:
1. Smoke run: 50 samples per dataset.
2. Full run: complete datasets.
3. Repeat with any alternate checkpoint only if explicitly requested.

Deliverables:
1. `jobs/run_salmonn_zeroshot.sh`.
2. Result bundle under one run id.

Acceptance criteria:
1. Full matrix completed and summarized.

---

## Phase 8: Reporting and Reproducibility

Objective: make results auditable and easy to compare.

Steps:
1. Auto-generate `summary.md` table across datasets.
2. Include environment and config snapshot (`run_config.json`).
3. Record runtime and hardware notes.
4. Add `README` section with exact run commands.

Deliverables:
1. `results/salmonn_zeroshot/<run_id>/summary.md`.
2. Updated root `README.md`.

Acceptance criteria:
1. Another user can reproduce from instructions without hidden steps.

---

## 5. CLI Contract (Planned)

1. `uv run salmonn-bench run-mos --config-path configs/salmonn_zeroshot.yaml --dataset-path <path>`
2. `uv run salmonn-bench run-ab --config-path configs/salmonn_zeroshot.yaml --dataset-path <path>`
3. `uv run bash jobs/run_salmonn_zeroshot.sh`

---

## 6. Risks and Mitigations

1. Dependency incompatibility between SALMONN upstream and latest ecosystem.
   - Mitigation: strict pinning + lockfile + GPU-specific profile.
2. Missing model assets or path mismatch.
   - Mitigation: startup validation and clear error output.
3. Throughput/runtime cost on large datasets.
   - Mitigation: smoke-first workflow and resumable outputs.
4. Parsing instability for MOS/winner extraction.
   - Mitigation: explicit extraction tests and fallback patterns.

---

## 7. Test Strategy

1. Unit tests:
   - dataset parsing
   - MOS extraction
   - winner extraction
2. Integration tests:
   - one-file MOS inference
   - one-pair A/B inference
3. End-to-end smoke:
   - 50 samples per split with full metric output

---

## 8. Execution Checklist

1. [ ] Tag baseline state.
2. [x] Replace repo structure with minimal SALMONN-first layout.
3. [x] Replace `pyproject.toml` and lock dependencies.
4. [x] Implement runtime config and model loader.
5. [x] Implement dataset adapters.
6. [x] Implement zero-shot inference commands.
7. [x] Implement evaluation metrics.
8. [ ] Run smoke benchmark.
9. [ ] Run full benchmark.
10. [ ] Publish summary and reproducibility notes.

Notes:
1. Smoke/full runs are blocked until SALMONN model assets are available at paths in `configs/salmonn_zeroshot.yaml`.

---

## 9. Future Phases (After Zero-Shot Baseline)

These are intentionally deferred until zero-shot benchmark is stable.

### Future Phase A: LoRA Adaptation
1. Add LoRA-based parameter-efficient finetuning for quality tasks.
2. Train on existing generated MOS + A/B instruction datasets.
3. Re-benchmark against zero-shot baseline.

### Future Phase B: Q-former/Adapter Refinement
1. Introduce Q-former-focused adaptation strategy for stronger acoustic alignment.
2. Evaluate whether adapter tuning improves MSE/Acc/BLEU without full model updates.
3. Compare against LoRA-only and zero-shot tracks.

### Future Phase C: Joint Training and Distillation
1. Add MOS + A/B joint training experiments.
2. Add distillation-based alignment as separate experimental track.
3. Keep all tuned tracks benchmarked against the locked zero-shot baseline.

---

## 10. Final Note

This plan deliberately prioritizes a **clean, runnable, zero-shot SALMONN benchmark** over model tuning complexity.

The branch should only move to LoRA/Q-former work after the zero-shot pipeline is fully operational and reproducible.
