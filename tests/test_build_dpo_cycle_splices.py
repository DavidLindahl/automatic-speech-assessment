import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from build_dpo_cycle_splices import (
    extract_mos_value,
    split_head_and_clause,
    swap_mos,
)

# A representative ARM A target: caption with the MOS inside, trailing ts clause.
GOLD = (
    "This synthesized speech is quite noisy and discontinuous, with moderate "
    "distortion. Although the loudness is just soft but understandable, the "
    "overall MOS score is only 1.4. The degradation in the clip is between "
    "<a3><f6> and <a4><f7>."
)
GOLD_HEAD = (
    "This synthesized speech is quite noisy and discontinuous, with moderate "
    "distortion. Although the loudness is just soft but understandable, the "
    "overall MOS score is only 1.4."
)
GOLD_CLAUSE = "The degradation in the clip is between <a3><f6> and <a4><f7>."


def test_split_head_and_clause_clean() -> None:
    split = split_head_and_clause(GOLD)
    assert split is not None
    head, clause = split
    assert head == GOLD_HEAD
    assert clause == GOLD_CLAUSE


def test_split_handles_no_trailing_period() -> None:
    target = "Noisy speech, MOS 2.0. The degradation in the clip is between <a1><f0> and <a2><f0>"
    split = split_head_and_clause(target)
    assert split is not None
    head, clause = split
    assert head == "Noisy speech, MOS 2.0."
    assert clause.endswith("<a2><f0>")


def test_split_returns_none_on_degenerate() -> None:
    # Token salad, no well-formed clause.
    assert split_head_and_clause("2><f3><f8><f5><f2><a4><f4>") is None
    # Lowercase connector before the clause is not the canonical template tail.
    assert split_head_and_clause("clarity. however degradation between <a2><f4> and <a3><f0>.") is None
    # No timestamps at all.
    assert split_head_and_clause("Just a caption with MOS 3.0 and no interval.") is None
    # Empty head (clause only) -> unusable.
    assert split_head_and_clause("The degradation in the clip is between <a1><f0> and <a2><f0>.") is None


def test_extract_mos_value_handles_phrasings() -> None:
    assert extract_mos_value("the overall MOS score is only 1.4.") == 1.4
    assert extract_mos_value("Given the MOS of 2.0, it is poor.") == 2.0
    assert extract_mos_value("a MOS score of 3.0") == 3.0
    # No MOS keyword: fall back to last number (matches evaluate.extract_mos).
    assert extract_mos_value("It scores 2.5 overall") == 2.5
    # Nothing numeric at all -> None (not 0.0), so it cannot fake a real rating.
    assert extract_mos_value("no numbers here") is None


def test_swap_mos_rewrites_only_the_number() -> None:
    out = swap_mos(GOLD_HEAD, 2.6)
    assert out is not None
    assert "MOS score is only 2.6." in out
    # Only the number changed; the surrounding caption is byte-identical.
    assert out == GOLD_HEAD.replace("only 1.4.", "only 2.6.")
    # And it still parses back to the new value.
    assert extract_mos_value(out) == 2.6


def test_swap_mos_formats_to_one_decimal() -> None:
    # A sampled MOS of 3.0 must render as "3.0", not "3".
    out = swap_mos(GOLD_HEAD, 3.0)
    assert out is not None
    assert "only 3.0." in out


def test_swap_mos_returns_none_without_mos() -> None:
    assert swap_mos("A caption with no rating mention at all.", 2.0) is None


def test_swap_mos_handles_integer_written_gold() -> None:
    # Some gold captions write the MOS without a decimal digit, so the sentence
    # period doubles as the "decimal point": "...MOS score of 1." Here the regex
    # captures just "1" and the swap must produce a well-formed "1.8." without a
    # doubled period or an orphaned digit.
    head = "Very poor speech, with an overall MOS score of 1."
    out = swap_mos(head, 1.8)
    assert out is not None
    assert out == "Very poor speech, with an overall MOS score of 1.8."
    assert ".." not in out
    assert extract_mos_value(out) == 1.8


def test_mos_cycle_corrupts_only_the_mos() -> None:
    # Reconstructing the MOS-cycle rejected: gold head with swapped MOS + gold clause.
    corrupted_head = swap_mos(GOLD_HEAD, 2.6)
    assert corrupted_head is not None
    rejected = f"{corrupted_head} {GOLD_CLAUSE}"
    # Timestamps identical to gold (localization untouched).
    assert GOLD_CLAUSE in rejected
    # Caption text identical except the single MOS token.
    assert rejected.replace("2.6", "1.4") == GOLD


def test_timestamp_cycle_corrupts_only_the_interval() -> None:
    # Reconstructing the timestamp-cycle rejected: gold head + a different clause.
    sampled_clause = "The degradation in the clip is between <a1><f7> and <a3><f1>."
    rejected = f"{GOLD_HEAD} {sampled_clause}"
    # Head (caption + MOS) byte-identical to gold; only the interval differs.
    split = split_head_and_clause(rejected)
    assert split is not None
    head, clause = split
    assert head == GOLD_HEAD
    assert clause == sampled_clause
    assert clause != GOLD_CLAUSE
