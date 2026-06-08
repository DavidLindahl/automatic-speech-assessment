from scripts.data.build_nisqa_temporal_json import (
    build_temporal_response,
    relabel_existing_temporal_records,
    splice_caption_from_verb,
)


def test_splice_caption_from_verb_strips_opener_keeps_caption_and_mos() -> None:
    caption = (
        "This synthesized speech has a moderate level of noise. However, it "
        "suffers from distortion. With an overall MOS score of 1.8, it falls "
        "into the somewhat unnatural category."
    )

    spliced = splice_caption_from_verb(caption)

    assert spliced.startswith("has a moderate level of noise")
    assert "MOS score of 1.8" in spliced
    assert "This synthesized speech" not in spliced


def test_splice_caption_from_verb_handles_alternate_openers() -> None:
    assert splice_caption_from_verb(
        "The synthesized speech is very clean and clear."
    ) == "is very clean and clear."
    assert splice_caption_from_verb(
        "This is extremely poor in quality."
    ) == "is extremely poor in quality."


def test_splice_caption_from_verb_falls_back_to_full_caption() -> None:
    caption = "Totally clean clip with no copula verb present."

    assert splice_caption_from_verb(caption) == caption


def test_build_temporal_response_global_caption_localization() -> None:
    caption = (
        "This synthesized speech has a moderate level of noise. "
        "With an overall MOS score of 1.8, it is somewhat unnatural."
    )

    response = build_temporal_response(
        base_caption=caption,
        start_time=2.877,
        end_time=4.726,
        degradation_phrase="background noise",
        label_style="global-caption-localization",
    )

    assert response == (
        "The degradation in the clip is between <|2.88|> and <|4.73|> and "
        "has a moderate level of noise. With an overall MOS score of 1.8, "
        "it is somewhat unnatural."
    )


def test_build_temporal_response_global_caption_anchoroffset() -> None:
    caption = (
        "This synthesized speech has a moderate level of noise. "
        "With an overall MOS score of 1.8, it is somewhat unnatural."
    )

    response = build_temporal_response(
        base_caption=caption,
        start_time=2.877,
        end_time=4.726,
        degradation_phrase="background noise",
        label_style="global-caption-anchoroffset",
    )

    assert response == (
        "The degradation in the clip is between <a2><f9> and <a4><f7> and "
        "has a moderate level of noise. With an overall MOS score of 1.8, "
        "it is somewhat unnatural."
    )


def test_build_temporal_response_clear_speech_localization_with_types() -> None:
    response = build_temporal_response(
        base_caption="Old global caption should be ignored.",
        start_time=1.234,
        end_time=5.678,
        degradation_phrase="background noise and codec artifacts",
    )

    assert response == (
        "The overall speech is clear, but the quality is interrupted by "
        "background noise and codec artifacts occurring between <|1.23|> "
        "and <|5.68|>."
    )
    assert "Old global caption" not in response


def test_relabel_existing_temporal_records_rewrites_query_and_response() -> None:
    records = [
        {
            "query": "old query",
            "response": "This synthesized speech is very poor. MOS score is 1.4.",
            "mix_deg_segments": [{"start": 2.0, "end": 3.5}],
            "source_degradation_types": ["bgn", "codec1"],
        }
    ]

    relabeled = relabel_existing_temporal_records(
        records=records,
        query="new query<audio>",
        label_style="clear-speech-localization",
    )

    assert relabeled[0]["query"] == "new query<audio>"
    assert relabeled[0]["response"] == (
        "The overall speech is clear, but the quality is interrupted by "
        "background noise and codec artifacts occurring between <|2.00|> "
        "and <|3.50|>."
    )
