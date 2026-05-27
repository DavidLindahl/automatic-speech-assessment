from scripts.data.build_nisqa_temporal_json import (
    build_temporal_response,
    relabel_existing_temporal_records,
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
