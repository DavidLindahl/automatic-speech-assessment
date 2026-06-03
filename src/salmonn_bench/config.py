"""Configuration loading for zero-shot SALMONN benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    """Runtime configuration for benchmark execution."""

    model: dict[str, Any]
    generate: dict[str, Any]
    prompts_path: Path
    mos_task: str
    ab_task: str
    device: str
    use_amp: bool
    silence_samples_ab: int


REQUIRED_MODEL_KEYS = {
    "llama_path",
    "whisper_path",
    "beats_path",
    "ckpt",
    "prompt_template",
}


def _ensure_required_model_keys(model_cfg: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MODEL_KEYS - set(model_cfg.keys()))
    if missing:
        msg = ", ".join(missing)
        raise ValueError(f"Missing model config keys: {msg}")


def _resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load and validate benchmark configuration.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed and validated benchmark configuration.
    """
    project_root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    model_cfg = dict(payload.get("model", {}))
    _ensure_required_model_keys(model_cfg)

    for key in ("llama_path", "whisper_path", "beats_path", "ckpt"):
        model_cfg[key] = str(_resolve_path(model_cfg[key], project_root))

    prompts_path = _resolve_path(payload.get("prompts_path"), project_root)

    return BenchmarkConfig(
        model=model_cfg,
        generate=dict(payload.get("generate", {})),
        prompts_path=prompts_path,
        mos_task=str(payload.get("mos_task", "mos_evaluation_description")),
        ab_task=str(payload.get("ab_task", "mos_ABtest")),
        device=str(payload.get("device", "cuda:0")),
        use_amp=bool(payload.get("use_amp", True)),
        silence_samples_ab=int(payload.get("silence_samples_ab", 1600)),
    )
