"""Compatibility shim for the legacy `asa.data` import surface.

The contents of this module used to live in one ~720-line file. As of the
phase-3 refactor it's split into:

- `asa.audio` — TARGET_SR, AUDIO_PLACEHOLDER, AUDIO_SPECIAL, load_audio
- `asa.prompts` — PROMPT_TEMPLATE, MOS expert-prompt builder
- `asa.datasets` — SFTDataset, DPODataset
- `asa.collators` — Qwen2AudioCollator, ALLDDPOCollator

This shim re-exports the public names so existing imports like
`from asa.data import SFTDataset` keep working. New code should import
from the focused modules directly.
"""

from asa.audio import (
    AUDIO_PLACEHOLDER,
    AUDIO_SPECIAL,
    TARGET_SR,
    load_audio,
)
from asa.collators import ALLDDPOCollator, Qwen2AudioCollator
from asa.datasets import DPODataset, SFTDataset
from asa.prompts import (
    DIMENSION_DEFINITIONS_MOS,
    EXPERT_FEW_SHOT_EXAMPLES_MOS,
    EXPERT_TASK_MOS,
    PROMPT_TEMPLATE,
    build_expert_prompt_MOS,
)


__all__ = [
    "AUDIO_PLACEHOLDER",
    "AUDIO_SPECIAL",
    "ALLDDPOCollator",
    "DIMENSION_DEFINITIONS_MOS",
    "DPODataset",
    "EXPERT_FEW_SHOT_EXAMPLES_MOS",
    "EXPERT_TASK_MOS",
    "PROMPT_TEMPLATE",
    "Qwen2AudioCollator",
    "SFTDataset",
    "TARGET_SR",
    "build_expert_prompt_MOS",
    "load_audio",
]
