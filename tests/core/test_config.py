"""Tests for configuration system."""

import json
import os
import tempfile

import pytest
from pydantic import ValidationError

from enumeraite.core.config import Config, load_config, load_merklemap_api_key


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERKLEMAP_API_KEY", raising=False)


def test_config_creation():
    """Test basic config creation with defaults."""
    config = Config()
    assert config.default_provider == "openai"
    assert config.default_count == 50
    assert config.max_concurrent_requests == 20
    assert config.request_timeout == 10
    assert config.http_verify_ssl is True
    assert config.http_baseline_depth == 1


def test_config_from_dict():
    """Test config creation from dictionary."""
    config_dict = {
        "providers": {"openai": {"api_key": "test-key", "model": "gpt-4"}},
        "default_provider": "openai",
        "default_count": 75,
    }
    config = Config.from_dict(config_dict)
    assert config.providers["openai"]["api_key"] == "test-key"
    assert config.default_count == 75


def test_get_provider_config():
    """Test getting provider-specific config."""
    config = Config.from_dict(
        {
            "providers": {
                "openai": {"api_key": "test-key", "model": "gpt-4"},
                "claude": {"api_key": "claude-key"},
            }
        }
    )

    openai_config = config.get_provider_config("openai")
    assert openai_config["api_key"] == "test-key"
    assert openai_config["model"] == "gpt-4"

    # Non-existent provider should return empty dict
    missing_config = config.get_provider_config("nonexistent")
    assert missing_config == {}


def test_load_config_from_file():
    """Test loading config from JSON file."""
    config_data = {
        "default_provider": "claude",
        "default_count": 100,
        "providers": {"claude": {"api_key": "test-claude-key"}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.default_provider == "claude"
        assert config.default_count == 100
        assert config.providers["claude"]["api_key"] == "test-claude-key"
    finally:
        os.unlink(temp_path)


def test_load_config_with_env_vars(monkeypatch):
    """Test loading config with environment variables."""
    # Set environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-claude-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Load config without file (should use env vars)
    config = load_config()

    assert "openai" in config.providers
    assert config.providers["openai"]["api_key"] == "env-openai-key"
    assert config.providers["openai"]["model"] == "gpt-3.5-turbo"

    assert "claude" in config.providers
    assert config.providers["claude"]["api_key"] == "env-claude-key"


def test_load_config_detects_minimax_without_persisting_it(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-test-key")
    config = load_config("/this/path/does/not/exist.json")
    assert config.providers["minimax"]["api_key"] == "minimax-test-key"
    assert config.providers["minimax"]["model"] == "MiniMax-M3"
    assert config.providers["minimax"]["hybrid"] is True


def test_generation_budget_environment_overrides(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    monkeypatch.setenv("ENUMERAITE_BATCH_SIZE", "17")
    monkeypatch.setenv("ENUMERAITE_MAX_BATCH_CALLS", "2")
    monkeypatch.setenv("ENUMERAITE_PROVIDER_TIMEOUT", "45.5")
    monkeypatch.setenv("ENUMERAITE_HYBRID_MODEL_FRACTION", "0")
    monkeypatch.setenv("ENUMERAITE_SEED_LIMIT", "123")
    monkeypatch.setenv("ENUMERAITE_SEED_CHAR_LIMIT", "4567")
    monkeypatch.setenv("ENUMERAITE_TEMPERATURE", "0")
    config = load_config("/this/path/does/not/exist.json")
    assert config.providers["minimax"]["batch_size"] == 17
    assert config.providers["minimax"]["max_batch_calls"] == 2
    assert config.providers["minimax"]["timeout"] == 45.5
    assert config.providers["minimax"]["hybrid_model_fraction"] == 0
    assert config.providers["minimax"]["seed_limit"] == 123
    assert config.providers["minimax"]["seed_char_limit"] == 4567
    assert config.providers["minimax"]["temperature"] == 0


def test_global_generation_defaults_apply_to_file_provider(tmp_path, monkeypatch):
    path = tmp_path / "enumeraite.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "minimax": {
                        "api_key": "file-secret",
                        "hybrid": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENUMERAITE_BATCH_SIZE", "13")
    monkeypatch.setenv("ENUMERAITE_HYBRID", "true")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://example.invalid/v1")
    config = load_config(str(path))
    provider = config.providers["minimax"]
    assert provider["api_key"] == "file-secret"
    assert provider["model"] == "MiniMax-M3"
    assert provider["batch_size"] == 13
    assert provider["base_url"] == "https://example.invalid/v1"
    assert provider["hybrid"] is False  # Explicit file values win.


def test_http_calibration_environment_overrides(monkeypatch):
    monkeypatch.setenv("ENUMERAITE_HTTP_VERIFY_SSL", "false")
    monkeypatch.setenv("ENUMERAITE_HTTP_SOFT_404_THRESHOLD", "0.88")
    monkeypatch.setenv("ENUMERAITE_HTTP_BASELINE_SAMPLES", "3")
    monkeypatch.setenv("ENUMERAITE_HTTP_BASELINE_DEPTH", "0")
    monkeypatch.setenv("ENUMERAITE_HTTP_MAX_BASELINE_GROUPS", "12")
    config = load_config("/this/path/does/not/exist.json")
    assert config.http_verify_ssl is False
    assert config.http_soft_404_threshold == 0.88
    assert config.http_baseline_samples == 3
    assert config.http_baseline_depth == 0
    assert config.http_max_baseline_groups == 12


def test_boolean_environment_values_are_strict(monkeypatch):
    monkeypatch.setenv("ENUMERAITE_HYBRID", "perhaps")
    with pytest.raises(ValueError, match="boolean"):
        load_config("/this/path/does/not/exist.json")


def test_load_config_nonexistent_file():
    """Test loading config when no file exists."""
    config = load_config("/this/path/does/not/exist.json")
    # Should return default config
    assert config.default_provider == "openai"
    assert config.default_count == 50
    assert config.providers["huggingface"]["trust_remote_code"] is False
    assert "model" not in config.providers["huggingface"]


def test_explicit_huggingface_model_is_preserved(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_MODEL", "org/custom-model")
    config = load_config("/this/path/does/not/exist.json")
    assert config.providers["huggingface"]["model"] == "org/custom-model"


def test_config_rejects_invalid_network_budgets():
    with pytest.raises(ValidationError):
        Config(max_concurrent_requests=0)
    with pytest.raises(ValidationError):
        Config(request_timeout=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_count": True},
        {"default_count": "50"},
        {"max_concurrent_requests": 2.5},
        {"request_timeout": False},
        {"request_timeout": "10"},
        {"http_verify_ssl": "false"},
        {"http_baseline_depth": 1.5},
    ],
)
def test_config_rejects_coerced_json_resource_types(kwargs):
    with pytest.raises(ValidationError):
        Config(**kwargs)


def test_default_count_cannot_exceed_the_generation_budget():
    with pytest.raises(ValidationError):
        Config(default_count=5_001)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"providers": []},
        {"providers": {"minimax": "not-an-object"}},
    ],
)
def test_load_config_rejects_malformed_json_structure(tmp_path, payload):
    path = tmp_path / "enumeraite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration root|providers must"):
        load_config(str(path))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0"])
def test_provider_timeout_environment_must_be_finite_and_positive(monkeypatch, value):
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    monkeypatch.setenv("ENUMERAITE_PROVIDER_TIMEOUT", value)
    with pytest.raises(ValueError, match="must be positive"):
        load_config("/this/path/does/not/exist.json")


def test_merklemap_file_precedence_and_explicit_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("MERKLEMAP_API_KEY", "fixture-env-key")
    path = tmp_path / ".config" / "enumeraite" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"MERKLEMAP_API_KEY": "fixture-file-key"}), encoding="utf-8")
    assert load_merklemap_api_key() == "fixture-file-key"

    project = tmp_path / "enumeraite.json"
    project.write_text(json.dumps({"MERKLEMAP_API_KEY": " \t"}), encoding="utf-8")
    assert load_merklemap_api_key() == ""

    # The selected file wins as a whole; do not merge keys from a lower-priority file.
    project.write_text("{}", encoding="utf-8")
    assert load_merklemap_api_key() == "fixture-env-key"
    assert load_merklemap_api_key(str(path)) == "fixture-file-key"


def test_merklemap_file_key_is_not_serialized_as_application_config(tmp_path):
    path = tmp_path / "enumeraite.json"
    path.write_text(json.dumps({"MERKLEMAP_API_KEY": "fixture-private-key"}), encoding="utf-8")
    config = load_config()
    assert load_merklemap_api_key() == "fixture-private-key"
    assert "fixture-private-key" not in config.model_dump_json()
    assert "fixture-private-key" not in repr(config)


def test_merklemap_malformed_key_does_not_echo_credentials(tmp_path):
    path = tmp_path / "enumeraite.json"
    path.write_text(
        json.dumps({"MERKLEMAP_API_KEY": {"key": "fixture-private-key"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError) as error:
        load_merklemap_api_key()
    assert "fixture-private-key" not in str(error.value)
