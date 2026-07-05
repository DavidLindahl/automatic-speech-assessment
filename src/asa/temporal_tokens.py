"""Discrete anchor/offset time tokens for temporal localization.

This module implements the TimeAudio (arXiv 2511.11039) timestamp tokenization
idea, adapted to Qwen2-Audio. Continuous timestamps in seconds are encoded as a
pair of discrete tokens:

    anchor ``<aN>``  -> the integer-second part N (0..MAX_SECONDS-1)
    offset ``<fK>``  -> the tenth-of-a-second part K (0..9)

For example ``2.88`` rounds to ``2.9`` and encodes as ``<a2><f9>``. Decoding the
same pair returns ``2.9``. Offset resolution is fixed at 0.1 s (one fractional
digit), which is sufficient for temporal IoU on multi-second windows and keeps
the offset vocabulary small (10 tokens).

Why discrete tokens instead of free-text ``<|2.88|>``? A decoder LLM has to spell
free-text timestamps digit by digit, with no inductive bias toward smooth time,
so small digit errors map to large time errors. Discrete time tokens give the
model one atomic symbol per time value. Following TimeAudio Eq. 1, the new token
embeddings are seeded from the LLM's existing numeral embeddings rather than
randomly, so the model starts with its pretrained "sense of numbers" instead of
cold weights.

Tokens are registered as REGULAR added tokens (not special tokens) so that they
survive ``batch_decode(skip_special_tokens=True)``; the eval path must be able to
read them back regardless of that flag.
"""

from __future__ import annotations

import re
from typing import List

# Max integer seconds we can encode. NISQA mixes are <= 60 s (Qwen2-Audio's own
# encoder caps at 1500 frames @ 25 fps = 60 s), so 60 anchor tokens cover the
# full addressable range.
MAX_SECONDS: int = 60

# Offset resolution: one fractional digit -> 10 offset tokens, 0.1 s steps.
NUM_OFFSETS: int = 10
OFFSET_STEP: float = 1.0 / NUM_OFFSETS  # 0.1 s

ANCHOR_PREFIX = "a"
OFFSET_PREFIX = "f"


def anchor_token(second: int) -> str:
    """Return the anchor token string for an integer second."""
    return f"<{ANCHOR_PREFIX}{second}>"


def offset_token(tenth: int) -> str:
    """Return the offset token string for a tenth-of-a-second bucket."""
    return f"<{OFFSET_PREFIX}{tenth}>"


def all_time_tokens() -> List[str]:
    """Return every time token to register, anchors then offsets."""
    anchors = [anchor_token(s) for s in range(MAX_SECONDS)]
    offsets = [offset_token(k) for k in range(NUM_OFFSETS)]
    return anchors + offsets


# Matches a single ``<a2><f9>`` pair and captures the two integers.
_PAIR_RE = re.compile(rf"<{ANCHOR_PREFIX}(\d+)>\s*<{OFFSET_PREFIX}(\d+)>")


def encode_time(seconds: float) -> str:
    """Encode a non-negative timestamp in seconds as ``<aN><fK>``.

    The value is rounded to the nearest 0.1 s. Anchors saturate at
    ``MAX_SECONDS - 1`` so out-of-range values degrade gracefully rather than
    minting unknown tokens.

    Args:
        seconds: Timestamp in seconds (clamped to ``>= 0``).

    Returns:
        Two-token string, e.g. ``"<a2><f9>"``.
    """
    if seconds < 0:
        seconds = 0.0
    # Round to the nearest tenth, then split into whole + tenth.
    total_tenths = int(round(seconds * NUM_OFFSETS))
    anchor = total_tenths // NUM_OFFSETS
    tenth = total_tenths % NUM_OFFSETS
    if anchor >= MAX_SECONDS:
        anchor = MAX_SECONDS - 1
        tenth = NUM_OFFSETS - 1
    return f"{anchor_token(anchor)}{offset_token(tenth)}"


def decode_all_times(text: str) -> List[float]:
    """Decode every ``<aN><fK>`` pair in text, left to right.

    Args:
        text: Text that may contain anchor/offset token pairs.

    Returns:
        List of decoded second values in order of appearance.
    """
    values: List[float] = []
    for match in _PAIR_RE.finditer(text):
        anchor = int(match.group(1))
        tenth = int(match.group(2))
        values.append(anchor + tenth * OFFSET_STEP)
    return values
