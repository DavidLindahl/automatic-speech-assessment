import evaluate
import pytest
from typer.testing import CliRunner


def test_help_flags_present():
    runner = CliRunner()

    # Top-level help should list global options we added
    result = runner.invoke(evaluate.app, ["--help"])
    assert result.exit_code == 0
    assert "--model-path" in result.output
    assert "--output-dir" in result.output
    assert "--dataset-path" in result.output
    assert "--batch-size" in result.output

    # Subcommand help should include per-command options
    result2 = runner.invoke(evaluate.app, ["eval-mos", "--help"])
    assert result2.exit_code == 0
    assert "--dataset-path" in result2.output
    assert "--model-path" in result2.output
    assert "--zero-shot" in result2.output


# --- extract_mos -----------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # Fine-tuned format ("MOS of X"): the first regex, unchanged. These
        # guard against silently shifting numbers already in the thesis tables.
        ("The speech is very clean, with an overall MOS of 4.7.", 4.7),
        ("Overall, a high-quality synthesized speech with an MOS of 4.6.", 4.6),
        ("...so the MOS score is only 2.1.", 2.1),
        # Zero-shot baseline phrasing: "X out of 5" / "X/5". Must take the
        # numerator, NOT the "5" denominator (the old last-number fallback bug).
        ("...the quality can be rated as 3 out of 5. The noise...", 3.0),
        ("I would rate the overall quality as a 4 out of 5.", 4.0),
        ("rated as 2 out of 5", 2.0),
        ("3/5 overall.", 3.0),
        # Other rating phrasings.
        ("I'd give it a score of 2.", 2.0),
        ("My rating: 4", 4.0),
        ("I would rate it 2 due to heavy noise.", 2.0),
        # Honest parse failures: no confident score -> None (the old code would
        # have fabricated a number from a stray digit).
        ("The speech is noisy and distorted.", None),
        ("It contains background noise throughout.", None),
        ("", None),
    ],
)
def test_extract_mos(text, expected):
    assert evaluate.extract_mos(text) == expected
