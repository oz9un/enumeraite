"""Tests for configuration system."""
import pytest
import tempfile
import os
import json
from enumeraite.core.config import Config, load_config

def test_config_creation():
    """Test basic config creation with defaults."""
    config = Config()
    assert config.default_provider == "openai"
    assert config.default_count == 50
    assert config.max_concurrent_requests == 20
    assert config.request_timeout == 10

def test_config_from_dict():
    """Test config creation from dictionary."""
    config_dict = {
        "providers": {
            "openai": {"api_key": "test-key", "model": "gpt-4"}
        },
        "default_provider": "openai",
        "default_count": 75
    }
    config = Config.from_dict(config_dict)
    assert config.providers["openai"]["api_key"] == "test-key"
    assert config.default_count == 75

def test_get_provider_config():
    """Test getting provider-specific config."""
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "test-key", "model": "gpt-4"},
            "claude": {"api_key": "claude-key"}
        }
    })

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
        "providers": {
            "claude": {"api_key": "test-claude-key"}
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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

def test_load_config_nonexistent_file():
    """Test loading config when no file exists."""
    config = load_config("/this/path/does/not/exist.json")
    # Should return default config
    assert config.default_provider == "openai"
    assert config.default_count == 50