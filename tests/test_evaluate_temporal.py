from evaluate_temporal import (
    Interval,
    extract_ground_truth_interval,
    extract_interval,
    interval_iou,
    query_to_prompt,
    strip_non_timestamp_special_tokens,
)


def test_extract_interval_prefers_timestamp_tokens() -> None:
    text = "degradation between <|1.25|> and <|2.75|> seconds"
    interval, source = extract_interval(text, duration_seconds=10.0)

    assert interval == Interval(start=1.25, end=2.75)
    assert source == "token"


def test_extract_interval_from_range_expression() -> None:
    text = "the distortion happens from 3.2 to 4.4 seconds"
    interval, source = extract_interval(text, duration_seconds=8.0)

    assert interval == Interval(start=3.2, end=4.4)
    assert source == "range"


def test_extract_interval_does_not_pair_partial_timestamp_with_mos() -> None:
    text = "The MOS score is 1.8. The issue occurs between <|2.88|> and ."
    interval, source = extract_interval(text, duration_seconds=6.0)

    assert interval is None
    assert source == "none"


def test_extract_interval_plain_fallback_default_on() -> None:
    # Default (allow_plain=True), used for fine-tuned models: with no token and
    # no range phrasing, two plain numbers are paired into an interval.
    text = "the degraded part runs 1.5 then ends around 3.0"
    interval, source = extract_interval(text, duration_seconds=6.0)

    assert interval == Interval(start=1.5, end=3.0)
    assert source == "plain"


def test_extract_interval_zero_shot_suppresses_plain_fallback() -> None:
    # The zero-shot trap: a free-text answer with non-temporal numbers. With the
    # plain fallback ON it manufactures a bogus interval; with it OFF (zero-shot)
    # it must honestly return None instead of faking a localization.
    text = "I would rate this 3 out of 5. The 6 second clip has some noise."

    manufactured, manufactured_source = extract_interval(
        text, duration_seconds=6.0, allow_plain=True
    )
    assert manufactured is not None
    assert manufactured_source == "plain"

    honest, honest_source = extract_interval(
        text, duration_seconds=6.0, allow_plain=False
    )
    assert honest is None
    assert honest_source == "none"


def test_extract_interval_zero_shot_keeps_explicit_range() -> None:
    # Suppressing the plain fallback must NOT break the honest range path: a
    # clean "between X and Y seconds" still parses under zero-shot.
    text = "The degradation occurs between 1.2 and 3.4 seconds."
    interval, source = extract_interval(
        text, duration_seconds=6.0, allow_plain=False
    )

    assert interval == Interval(start=1.2, end=3.4)
    assert source == "range"


def test_interval_iou() -> None:
    pred = Interval(start=2.0, end=5.0)
    truth = Interval(start=3.0, end=6.0)

    assert interval_iou(pred, truth) == 0.5


def test_extract_ground_truth_interval_prefers_manifest_segments() -> None:
    record = {
        "mix_deg_segments": [{"start": 0.5, "end": 1.0}, {"start": 2.0, "end": 4.0}],
        "response": "between <|7.00|> and <|8.00|>",
        "duration_seconds": 10.0,
    }
    interval, source = extract_ground_truth_interval(record)

    assert interval == Interval(start=2.0, end=4.0)
    assert source == "mix_deg_segments"


def test_query_to_prompt_replaces_audio_placeholder() -> None:
    prompt = query_to_prompt("Please localize degradation<audio>and report timestamps.")
    assert "<|audio_bos|><|AUDIO|><|audio_eos|>" in prompt


def test_strip_non_timestamp_special_tokens_preserves_timestamps() -> None:
    text = "<|im_start|>assistant degradation <|1.25|> to <|2.75|><|im_end|>"
    cleaned = strip_non_timestamp_special_tokens(text)

    assert cleaned == "assistant degradation <|1.25|> to <|2.75|>"
