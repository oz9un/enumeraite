"""Tests for provider factory and registry."""
import pytest
from enumeraite.core.factory import ProviderFactory
from enumeraite.core.config import Config
from enumeraite.providers.openai_provider import OpenAIProvider
from enumeraite.providers.claude_provider import ClaudeProvider

def test_create_openai_provider():
    """Test creating OpenAI provider."""
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "test-key", "model": "gpt-4"}
        }
    })

    factory = ProviderFactory(config)
    provider = factory.create_provider("openai")

    assert isinstance(provider, OpenAIProvider)
    assert provider.config["api_key"] == "test-key"
    assert provider.config["model"] == "gpt-4"

def test_create_claude_provider():
    """Test creating Claude provider."""
    config = Config.from_dict({
        "providers": {
            "claude": {"api_key": "claude-key", "model": "anthropic/claude-sonnet-4"}
        }
    })

    factory = ProviderFactory(config)
    provider = factory.create_provider("claude")

    assert isinstance(provider, ClaudeProvider)
    assert provider.config["api_key"] == "claude-key"

def test_get_available_providers():
    """Test getting list of available providers."""
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "key1"},
            "claude": {"api_key": "key2"}
        }
    })

    factory = ProviderFactory(config)
    providers = factory.get_available_providers()

    assert "openai" in providers
    assert "claude" in providers
    assert len(providers) == 2

def test_get_default_provider():
    """Test getting default provider."""
    config = Config.from_dict({
        "default_provider": "claude",
        "providers": {
            "openai": {"api_key": "key1"},
            "claude": {"api_key": "key2"}
        }
    })

    factory = ProviderFactory(config)
    provider = factory.get_default_provider()

    assert isinstance(provider, ClaudeProvider)

def test_get_default_provider_fallback():
    """Test fallback to first available provider."""
    config = Config.from_dict({
        "default_provider": "nonexistent",
        "providers": {
            "openai": {"api_key": "key1"}
        }
    })

    factory = ProviderFactory(config)
    provider = factory.get_default_provider()

    assert isinstance(provider, OpenAIProvider)

def test_create_unknown_provider():
    """Test error handling for unknown provider."""
    config = Config.from_dict({
        "providers": {
            "openai": {"api_key": "key1"}
        }
    })

    factory = ProviderFactory(config)

    with pytest.raises(ValueError, match="Unknown provider 'unknown'"):
        factory.create_provider("unknown")

def test_create_provider_no_config():
    """Test error handling when provider has no config."""
    config = Config.from_dict({
        "providers": {}  # No providers configured
    })

    factory = ProviderFactory(config)

    with pytest.raises(ValueError, match="No configuration found for provider 'openai'"):
        factory.create_provider("openai")

def test_no_providers_available():
    """Test error when no providers are available."""
    config = Config.from_dict({
        "providers": {}  # No providers
    })

    factory = ProviderFactory(config)

    with pytest.raises(ValueError, match="No providers available"):
        factory.get_default_provider()