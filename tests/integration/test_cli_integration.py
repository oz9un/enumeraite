"""Integration tests for CLI commands."""
import pytest
import tempfile
import os
import json
from click.testing import CliRunner
from enumeraite.cli.main import cli


class TestCLIIntegration:
    """Test CLI commands end-to-end."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

        # Create temporary input file
        self.temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_input.write("/api/users\n")
        self.temp_input.write("/api/auth/login\n")
        self.temp_input.write("/admin/dashboard\n")
        self.temp_input.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_input.name)

    def test_batch_command_basic_functionality(self):
        """Test batch command with minimal options."""
        result = self.runner.invoke(cli, [
            'batch',
            '--target', 'example.com',
            '--input', self.temp_input.name,
            '--count', '3'
        ])

        # Should fail gracefully due to no API keys configured
        # But should not crash with syntax errors
        assert "Configuration error" in result.output or "Provider error" in result.output

    def test_batch_command_with_output_file(self):
        """Test batch command with output file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_output:
            temp_output_path = temp_output.name

        try:
            result = self.runner.invoke(cli, [
                'batch',
                '--target', 'example.com',
                '--input', self.temp_input.name,
                '--output', temp_output_path,
                '--count', '5'
            ])

            # Should fail gracefully due to no API keys
            assert "Configuration error" in result.output or "Provider error" in result.output

        finally:
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def test_batch_command_advanced_output(self):
        """Test batch command with advanced JSON output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_output:
            temp_output_path = temp_output.name

        try:
            result = self.runner.invoke(cli, [
                'batch',
                '--target', 'example.com',
                '--input', self.temp_input.name,
                '--output', temp_output_path,
                '--count', '5',
                '--advanced'
            ])

            # Should fail gracefully due to no API keys
            assert "Configuration error" in result.output or "Provider error" in result.output

        finally:
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def test_continuous_command_basic_functionality(self):
        """Test continuous command with minimal options."""
        result = self.runner.invoke(cli, [
            'continuous',
            '--target', 'example.com',
            '--input', self.temp_input.name,
            '--duration', '1m',  # Very short duration
            '--batch-size', '5',
            '--max-empty-rounds', '1'
        ])

        # Should fail gracefully due to no API keys configured
        # But should not crash with syntax errors
        assert "Configuration error" in result.output or "Provider error" in result.output

    def test_continuous_command_invalid_duration(self):
        """Test continuous command with invalid duration."""
        result = self.runner.invoke(cli, [
            'continuous',
            '--target', 'example.com',
            '--input', self.temp_input.name,
            '--duration', 'invalid_duration'
        ])

        assert result.exit_code != 0
        assert "Invalid duration" in result.output

    def test_continuous_command_nonexistent_input(self):
        """Test continuous command with nonexistent input file."""
        result = self.runner.invoke(cli, [
            'continuous',
            '--target', 'example.com',
            '--input', 'nonexistent_file.txt',
            '--duration', '1m'
        ])

        assert result.exit_code != 0

    def test_batch_command_missing_target(self):
        """Test error handling for missing target."""
        result = self.runner.invoke(cli, [
            'batch',
            '--input', self.temp_input.name
        ])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Usage:' in result.output

    def test_continuous_command_missing_target(self):
        """Test error handling for missing target in continuous mode."""
        result = self.runner.invoke(cli, [
            'continuous',
            '--input', self.temp_input.name
        ])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Usage:' in result.output

    def test_help_commands(self):
        """Test that all help commands work."""
        # Main help
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Enumeraite' in result.output

        # Batch help
        result = self.runner.invoke(cli, ['batch', '--help'])
        assert result.exit_code == 0
        assert 'target' in result.output

        # Continuous help
        result = self.runner.invoke(cli, ['continuous', '--help'])
        assert result.exit_code == 0
        assert 'duration' in result.output

    def test_version_command(self):
        """Test version command works."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_load_paths_from_file_with_comments(self):
        """Test loading paths file with comments and whitespace."""
        from enumeraite.cli.commands import _load_paths_from_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("/api/users\n")
            f.write("# This is a comment\n")
            f.write("   /api/admin/login   \n")  # Whitespace
            f.write("\n")  # Empty line
            f.write("/api/data/export\n")
            temp_path = f.name

        try:
            paths = _load_paths_from_file(temp_path)
            assert len(paths) == 3
            assert "/api/users" in paths
            assert "/api/admin/login" in paths
            assert "/api/data/export" in paths
            assert "# This is a comment" not in paths
        finally:
            os.unlink(temp_path)

    def test_output_functions(self):
        """Test output formatting functions."""
        from enumeraite.cli.commands import _output_simple, _output_advanced
        from enumeraite.core.models import GenerationResult, ValidationResult

        # Test simple output
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

        # Test advanced output
        validation_results = [
            ValidationResult(path="https://example.com/api/test1", status_code=200, exists=True, method="GET")
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            _output_advanced(result, validation_results, temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert data["generated_count"] == 2
                assert len(data["generated_paths"]) == 2
                assert data["generated_paths"][0]["path"] == "/api/test1"

        finally:
            os.unlink(temp_path)