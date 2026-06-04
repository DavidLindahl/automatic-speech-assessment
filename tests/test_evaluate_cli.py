import evaluate
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
