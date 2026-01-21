"""Tests for CLI commands."""
import pytest
import tempfile
import os
from click.testing import CliRunner
from enumeraite.cli.main import cli

def test_cli_help():
    """Test that main CLI help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'enumeraite' in result.output.lower()
    assert 'AI-Assisted' in result.output

def test_batch_command_help():
    """Test that batch command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['batch', '--help'])
    assert result.exit_code == 0
    assert 'target' in result.output
    assert 'input' in result.output

def test_batch_command_missing_target():
    """Test error handling for missing required options."""
    runner = CliRunner()
    result = runner.invoke(cli, ['batch', '--input', 'nonexistent.txt'])
    assert result.exit_code != 0
    assert 'Missing option' in result.output or 'Usage:' in result.output

def test_batch_command_nonexistent_file():
    """Test error handling for nonexistent input file."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        'batch',
        '--target', 'example.com',
        '--input', 'definitely_not_a_real_file.txt'
    ])
    assert result.exit_code != 0

def test_continuous_command_help():
    """Test that continuous command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['continuous', '--help'])
    assert result.exit_code == 0
    assert 'duration' in result.output
    assert 'batch-size' in result.output

def test_load_paths_from_file():
    """Test loading paths from input file."""
    from enumeraite.cli.commands import _load_paths_from_file

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("/api/user/profile\n")
        f.write("# This is a comment\n")
        f.write("/api/admin/dashboard\n")
        f.write("   /api/user/settings   \n")  # Test whitespace handling
        temp_path = f.name

    try:
        paths = _load_paths_from_file(temp_path)
        assert len(paths) == 3  # Should skip comment
        assert "/api/user/profile" in paths
        assert "/api/admin/dashboard" in paths
        assert "/api/user/settings" in paths
        assert "# This is a comment" not in paths
    finally:
        os.unlink(temp_path)

def test_output_simple():
    """Test simple output formatting."""
    from enumeraite.cli.commands import _output_simple
    from enumeraite.core.models import GenerationResult

    result = GenerationResult(
        paths=["/api/test1", "/api/test2"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test"}
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name

    try:
        _output_simple(result, temp_path)

        with open(temp_path, 'r') as f:
            content = f.read()
            assert "/api/test1" in content
            assert "/api/test2" in content
    finally:
        os.unlink(temp_path)

def test_output_advanced():
    """Test advanced JSON output formatting."""
    from enumeraite.cli.commands import _output_advanced
    from enumeraite.core.models import GenerationResult, ValidationResult
    import json

    generation_result = GenerationResult(
        paths=["/api/test1", "/api/test2"],
        confidence_scores=[0.8, 0.7],
        metadata={"provider": "test", "target": "example.com"}
    )

    validation_results = [
        ValidationResult(path="https://example.com/api/test1", status_code=200, exists=True, method="GET")
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        _output_advanced(generation_result, validation_results, temp_path)

        with open(temp_path, 'r') as f:
            data = json.load(f)
            assert data["generated_count"] == 2
            assert data["metadata"]["provider"] == "test"
            assert len(data["generated_paths"]) == 2
    finally:
        os.unlink(temp_path)