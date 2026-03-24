import pytest
from typer.testing import CliRunner
from scripts.data.data_cli import app

runner = CliRunner()

def test_cli_help():
    """Verify Typer CLI responds properly and correctly nested commands exist."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "generate-captions" in result.stdout
