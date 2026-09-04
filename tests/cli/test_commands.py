"""Tests for the current CLI command surface."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from enumeraite.cli.bench_commands import _read_wordlist
from enumeraite.cli.main import cli
from enumeraite.cli.run_command import _validator_for
from enumeraite.cli.ui import _display_count, print_debug_info
from enumeraite.core.config import Config


def test_cli_help_lists_primary_workflows():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AI-assisted" in result.output
    for command in ("run", "doctor", "bench", "generate", "analyze", "royale"):
        assert command in result.output


def test_run_command_help():
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--loop" in result.output
    assert "--validate" in result.output


def test_bench_command_help():
    runner = CliRunner()
    assert runner.invoke(cli, ["bench", "--help"]).exit_code == 0
    result = runner.invoke(cli, ["bench", "openapi", "--help"])
    assert result.exit_code == 0
    assert "--specs" in result.output
    assert "--budgets" in result.output
    replay = runner.invoke(cli, ["bench", "replay", "--help"])
    assert replay.exit_code == 0
    assert "no model calls" in replay.output


def test_replay_cli_rejects_oversized_manifest_before_reading(tmp_path: Path):
    source = tmp_path / "oversized.json"
    with source.open("wb") as handle:
        handle.seek(256 * 1024 * 1024)
        handle.write(b"{}")
    result = CliRunner().invoke(cli, ["bench", "replay", str(source)])
    assert result.exit_code != 0
    assert "256 MiB" in result.output


def test_benchmark_wordlist_read_is_bounded(tmp_path: Path):
    wordlist = tmp_path / "oversized.txt"
    with wordlist.open("wb") as handle:
        handle.seek(64 * 1024 * 1024)
        handle.write(b"x")
    with pytest.raises(click.ClickException, match="64 MiB"):
        _read_wordlist(wordlist)


def test_run_rejects_empty_seed(tmp_path: Path):
    seed = tmp_path / "seed.txt"
    seed.write_text("# only a comment\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(seed)])
    assert result.exit_code != 0
    assert "no usable items" in result.output


def test_path_loop_requires_target():
    try:
        _validator_for("path", "")
    except click.ClickException as exc:
        assert "--target" in str(exc)
    else:
        raise AssertionError("expected target validation failure")


def test_validator_uses_configured_network_limits(monkeypatch):
    captured = {}

    class Validator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("enumeraite.cli.run_command.PathHTTPValidator", Validator)
    _validator_for(
        "path",
        "example.test",
        Config(request_timeout=7.5, max_concurrent_requests=9),
    )
    assert captured == {
        "timeout": 7.5,
        "max_concurrent": 9,
        "verify_ssl": True,
        "soft_404_threshold": 0.92,
        "baseline_samples": 2,
        "baseline_depth": 1,
        "max_baseline_groups": 32,
    }


def test_version_is_single_source_release():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_debug_ui_sanitizes_remote_metadata_and_bad_telemetry(capsys):
    print_debug_info(
        {
            "provider": "remote\x1b[31m",
            "model": "model\u202e",
            "token_usage": {
                "input_tokens": "not-a-number",
                "output_tokens": 3,
                "total_tokens": float("nan"),
            },
        }
    )
    output = capsys.readouterr().err
    assert "\x1b[31m" not in output
    assert "\u202e" not in output
    assert "Output tokens" in output
    assert _display_count(float("inf")) == 0
