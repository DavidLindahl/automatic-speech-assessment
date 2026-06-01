"""CPU-safe tests for anchor/offset time token encode/decode round trips."""

from __future__ import annotations

import pytest

from asa import temporal_tokens as tt


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "<a0><f0>"),
        (2.9, "<a2><f9>"),
        (2.88, "<a2><f9>"),  # rounds to 2.9
        (2.84, "<a2><f8>"),  # rounds to 2.8
        (4.73, "<a4><f7>"),
        (10.06, "<a10><f1>"),  # 10.06 -> 100.6 tenths -> round to 101 -> 10.1
    ],
)
def test_encode_time(seconds, expected):
    assert tt.encode_time(seconds) == expected


def test_encode_negative_clamps_to_zero():
    assert tt.encode_time(-3.0) == "<a0><f0>"


def test_encode_out_of_range_saturates():
    # Beyond MAX_SECONDS the anchor saturates instead of minting unknown tokens.
    out = tt.encode_time(tt.MAX_SECONDS + 5)
    assert out == f"<a{tt.MAX_SECONDS - 1}><f{tt.NUM_OFFSETS - 1}>"


@pytest.mark.parametrize("seconds", [0.0, 0.1, 2.9, 4.7, 12.3, 33.6, 59.9])
def test_round_trip(seconds):
    decoded = tt.decode_first_time(tt.encode_time(seconds))
    assert decoded is not None
    assert decoded == pytest.approx(seconds, abs=0.05)


def test_decode_interval_in_sentence():
    text = (
        "The overall speech is clear, but there is distortion between "
        "<a2><f9> and <a4><f7>."
    )
    interval = tt.decode_interval(text)
    assert interval is not None
    start, end = interval
    assert start == pytest.approx(2.9, abs=0.05)
    assert end == pytest.approx(4.7, abs=0.05)


def test_decode_interval_orders_start_before_end():
    text = "noise between <a5><f0> and <a1><f0>"
    start, end = tt.decode_interval(text)
    assert start == pytest.approx(1.0, abs=0.05)
    assert end == pytest.approx(5.0, abs=0.05)


def test_decode_missing_returns_none():
    assert tt.decode_first_time("no timestamps here") is None
    assert tt.decode_interval("only one <a3><f3> here") is None


def test_all_time_tokens_count():
    tokens = tt.all_time_tokens()
    assert len(tokens) == tt.MAX_SECONDS + tt.NUM_OFFSETS
    assert tokens[0] == "<a0>"
    assert tokens[tt.MAX_SECONDS] == "<f0>"
    # No duplicates.
    assert len(set(tokens)) == len(tokens)
