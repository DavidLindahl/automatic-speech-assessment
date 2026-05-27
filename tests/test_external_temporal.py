import json
from pathlib import Path

from typer.testing import CliRunner

from scripts.eval.external_temporal import app


def test_prepare_external_temporal_timeaudio_json(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not-a-real-wav")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "audios": [str(audio_path)],
                "query": "Please find timestamps<audio>",
                "duration_seconds": 6.0,
                "mix_deg_segments": [{"start": 1.2, "end": 2.8}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "external"
    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--data-root",
            str(tmp_path),
            "--model-format",
            "timeaudio",
        ],
    )

    assert result.exit_code == 0
    prepared = json.loads((output_dir / "dataset_timeaudio.json").read_text())
    assert prepared == [
        {
            "id": "sample-1",
            "audio": str(audio_path.resolve()),
            "question": "Please find timestamps",
            "answer": ("The localized degradation occurs between 1.20 - 2.80 seconds."),
            "duration": 6.0,
            "source_dataset": str(dataset_path),
            "source_model_format": "timeaudio",
        }
    ]


def test_prepare_external_temporal_fails_when_all_audio_is_missing(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "audios": ["missing/audio.wav"],
                "query": "Please find timestamps<audio>",
                "duration_seconds": 6.0,
                "mix_deg_segments": [{"start": 1.2, "end": 2.8}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "external"),
            "--data-root",
            str(tmp_path),
            "--model-format",
            "timeaudio",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "No records had resolvable audio files" in str(result.exception)


def test_score_external_temporal_predictions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "duration_seconds": 6.0,
                "mix_deg_segments": [{"start": 1.0, "end": 3.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "model_prediction": "The issue happens from 1.5 - 2.5 seconds.",
                }
            ]
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "scored.json"
    output_csv = tmp_path / "scored.csv"
    result = CliRunner().invoke(
        app,
        [
            "score",
            "--dataset-path",
            str(dataset_path),
            "--prediction-path",
            str(prediction_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0
    scored = json.loads(output_json.read_text())
    assert scored["metrics"]["samples_total"] == 1
    assert scored["metrics"]["samples_with_parsed_prediction_interval"] == 1
    assert scored["results"][0]["pred_start"] == 1.5
    assert output_csv.exists()
