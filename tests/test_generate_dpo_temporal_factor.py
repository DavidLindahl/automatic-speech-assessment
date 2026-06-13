import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from generate_dpo_temporal_factor import (
    detect_timestamp_format,
    rewrite_timestamps,
    shift_interval,
)


def test_detect_format_freetext_and_anchoroffset() -> None:
    assert detect_timestamp_format("between <|3.57|> and <|4.72|> and noisy") == "freetext"
    assert detect_timestamp_format("noisy. Degradation between <a3><f6> and <a4><f7>.") == "anchoroffset"
    assert detect_timestamp_format("no timestamps here") is None


def test_rewrite_freetext_keeps_caption() -> None:
    chosen = "The degradation in the clip is between <|3.57|> and <|4.72|> and is noisy. MOS 1.4."
    out = rewrite_timestamps(chosen, 3.07, 4.22)
    assert out == "The degradation in the clip is between <|3.07|> and <|4.22|> and is noisy. MOS 1.4."


def test_rewrite_anchoroffset_keeps_caption() -> None:
    # Caption-last layout, ARM A format. 2.0s -> <a2><f0>, 3.5s -> <a3><f5>.
    chosen = "This speech is noisy. MOS 1.4. The degradation in the clip is between <a4><f1> and <a5><f5>."
    out = rewrite_timestamps(chosen, 2.0, 3.5)
    assert out == "This speech is noisy. MOS 1.4. The degradation in the clip is between <a2><f0> and <a3><f5>."
    assert out.count("<a") == 2


def test_rewrite_returns_none_without_two_timestamps() -> None:
    assert rewrite_timestamps("only one <|3.5|> here", 1.0, 2.0) is None
    assert rewrite_timestamps("no timestamps", 1.0, 2.0) is None


def test_shift_interval_clamps_and_rejects_degenerate() -> None:
    # Normal shift.
    assert shift_interval(2.0, 3.0, 1.0, 10.0) == (3.0, 4.0)
    # Clamped onto the gold interval -> None.
    assert shift_interval(0.0, 2.0, -0.005, 10.0) is None
    # Degenerate (both clamped to duration).
    assert shift_interval(9.5, 9.9, 5.0, 10.0) is None
