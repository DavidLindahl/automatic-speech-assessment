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


def test_whole_clip_baseline_mean_tiou() -> None:
    from evaluate_temporal import whole_clip_baseline_mean_tiou

    truths = [Interval(start=2.0, end=4.0), Interval(start=0.0, end=10.0)]
    durations = [10.0, 10.0]

    # Clip 1: IoU([0,10],[2,4]) = 2/10. Clip 2: IoU([0,10],[0,10]) = 1.0.
    assert whole_clip_baseline_mean_tiou(truths, durations) == 0.6


def test_whole_clip_baseline_skips_missing_durations() -> None:
    from evaluate_temporal import whole_clip_baseline_mean_tiou

    truths = [Interval(start=2.0, end=4.0), Interval(start=1.0, end=2.0)]

    score = whole_clip_baseline_mean_tiou(truths, [10.0, None])

    assert score == 0.2


def test_best_constant_baseline_finds_shared_window() -> None:
    from evaluate_temporal import best_constant_baseline

    truths = [
        Interval(start=2.0, end=3.0),
        Interval(start=2.0, end=3.0),
        Interval(start=2.25, end=3.25),
    ]

    interval, score = best_constant_baseline(truths)

    assert interval is not None
    assert interval.start <= 2.5 <= interval.end
    assert score > 0.5


def test_best_constant_baseline_empty_truths() -> None:
    from evaluate_temporal import best_constant_baseline

    interval, score = best_constant_baseline([])

    assert interval is None
    assert score == 0.0


def test_extract_caption_part_timestamp_first_layout() -> None:
    from evaluate_temporal import extract_caption_part

    text = (
        "The degradation in the clip is between <a3><f6> and <a4><f7> and "
        "is quite noisy. The overall MOS score is only 1.4."
    )

    assert extract_caption_part(text) == (
        "is quite noisy. The overall MOS score is only 1.4."
    )


def test_extract_caption_part_caption_last_layout() -> None:
    from evaluate_temporal import extract_caption_part

    text = (
        "This speech is clear. MOS score is 4.2. "
        "The degradation in the clip is between <|2.10|> and <|3.40|>."
    )

    assert extract_caption_part(text) == "This speech is clear. MOS score is 4.2."


def test_extract_caption_part_no_clause_passthrough() -> None:
    from evaluate_temporal import extract_caption_part

    assert extract_caption_part("  Just a   caption. ") == "Just a caption."


def test_extract_mos_from_response() -> None:
    from evaluate_temporal import extract_mos_from_response

    assert extract_mos_from_response("the overall MOS score is only 1.4.") == 1.4
    assert extract_mos_from_response("MOS of 4.3 overall") == 4.3
    assert extract_mos_from_response("no score mentioned here") is None


def test_caption_corpus_bleu_perfect_match_or_skipped() -> None:
    import pytest as _pytest

    from evaluate_temporal import caption_corpus_bleu

    captions = ["the speech is clear and natural"] * 3
    score = caption_corpus_bleu(captions, captions)
    if score is None:
        _pytest.skip("sacrebleu not installed")
    assert score == _pytest.approx(100.0)
    assert caption_corpus_bleu([], []) is None
