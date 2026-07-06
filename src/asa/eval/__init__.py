"""Shared evaluation library for the ASA project.

The CLI entrypoints under ``scripts/eval/`` (``evaluate.py``,
``evaluate_temporal.py``, ``evaluate_gemini_mos.py``,
``evaluate_gemini_temporal.py``) are thin wrappers over this package. All the
scoring math lives here exactly once so the four evals stay comparable and the
numbers cannot drift between copies:

- :mod:`asa.eval.metrics`   — MOS parsing, caption metrics, MOS MAE/MSE and the
  output-diversity ("is it guessing?") diagnostics. Shared by every task.
- :mod:`asa.eval.intervals` — the temporal ``Interval`` type, the timestamp
  parser cascade, t-IoU / offset metrics, ground-truth extraction, caption
  timestamp stripping, and the audio-blind baselines.
- :mod:`asa.eval.gemini_api` — the resumable Gemini API driver: quota handling,
  cost accounting, JSONL resume, run-config integrity, and Batch helpers.
"""
