"""Prompt loading utilities."""

from __future__ import annotations

import json
from pathlib import Path


def load_task_prompt(prompts_path: Path, task_name: str) -> str:
    """Load a task prompt from a JSON prompt file.

    Args:
        prompts_path: Path to JSON file containing task-prompt mappings.
        task_name: Task key.

    Returns:
        Prompt template string for the requested task.
    """
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    if task_name not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Prompt task '{task_name}' not found. Available: {available}")
    return str(payload[task_name])
