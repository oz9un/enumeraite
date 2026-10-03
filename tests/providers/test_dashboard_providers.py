"""Dashboard endpoint pinning; SDK construction only, never network calls."""

from unittest.mock import patch

import pytest

from enumeraite.core.scan_budget import SpendLedger, attach_scan_guard, price_for
from enumeraite.providers.anthropic_provider import AnthropicProvider
from enumeraite.providers.openai_provider import OpenAIProvider


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("no network permitted")

    monkeypatch.setattr("socket.socket.connect", denied)
    monkeypatch.setattr("socket.getaddrinfo", denied)


def test_anthropic_constructor_explicitly_pins_default_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.example.test")
    with patch("enumeraite.providers.anthropic_provider.anthropic.Anthropic") as constructor:
        provider = AnthropicProvider({"api_key": "fixture-not-a-real-key"})
    assert constructor.call_args.kwargs["base_url"] == "https://api.anthropic.com"
    assert provider.config["base_url"] == "https://api.anthropic.com"
    assert provider.model == "claude-sonnet-4-6"


def test_anthropic_sdk_ignores_endpoint_environment_when_default_is_pinned(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.example.test")
    # SDK initialization is local; patch the transport's send method to make
    # accidental network requests fail loudly if that ever changes.
    with patch("socket.create_connection", side_effect=AssertionError("no network permitted")):
        provider = AnthropicProvider({"api_key": "fixture-not-a-real-key"})
        try:
            assert str(provider.client.base_url).rstrip("/") == "https://api.anthropic.com"
            attach_scan_guard(
                provider, SpendLedger(1, price_for("anthropic", provider.model)), None
            )
        finally:
            provider.client.close()


def test_anthropic_explicit_custom_endpoint_remains_possible_but_requires_price_override(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://environment.example.test")
    with patch("enumeraite.providers.anthropic_provider.anthropic.Anthropic") as constructor:
        provider = AnthropicProvider(
            {"api_key": "fixture-not-a-real-key", "base_url": "https://configured.example.test"}
        )
    assert constructor.call_args.kwargs["base_url"] == "https://configured.example.test"
    assert provider.config["base_url"] == "https://configured.example.test"
    with pytest.raises(ValueError, match="custom Anthropic"):
        attach_scan_guard(provider, SpendLedger(1, price_for("anthropic", provider.model)), None)


def test_openai_sdk_default_base_url_is_used_for_registry_endpoint_check(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with patch("socket.create_connection", side_effect=AssertionError("no network permitted")):
        provider = OpenAIProvider({"api_key": "fixture-not-a-real-key", "model": "gpt-4.1-mini"})
        try:
            assert not provider.config["base_url"]
            assert str(provider.client.base_url).rstrip("/") == "https://api.openai.com/v1"
            attach_scan_guard(provider, SpendLedger(1, price_for("openai", provider.model)), None)
        finally:
            provider.client.close()


def test_openai_sdk_environment_endpoint_cannot_reuse_official_registry_price(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example.test/v1")
    with patch("socket.create_connection", side_effect=AssertionError("no network permitted")):
        provider = OpenAIProvider({"api_key": "fixture-not-a-real-key", "model": "gpt-4.1-mini"})
        try:
            assert not provider.config["base_url"]
            assert (
                str(provider.client.base_url).rstrip("/") == "https://environment.example.test/v1"
            )
            with pytest.raises(ValueError, match="custom OpenAI"):
                attach_scan_guard(
                    provider, SpendLedger(1, price_for("openai", provider.model)), None
                )
        finally:
            provider.client.close()


def test_explicit_openai_endpoint_beats_environment_and_keeps_official_price(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example.test/v1")
    with patch("socket.create_connection", side_effect=AssertionError("no network permitted")):
        provider = OpenAIProvider(
            {
                "api_key": "fixture-not-a-real-key",
                "model": "gpt-4.1",
                "base_url": "https://api.openai.com/v1",
            }
        )
        try:
            assert provider.config["base_url"] == "https://api.openai.com/v1"
            assert str(provider.client.base_url).rstrip("/") == "https://api.openai.com/v1"
            attach_scan_guard(provider, SpendLedger(1, price_for("openai", provider.model)), None)
        finally:
            provider.client.close()
