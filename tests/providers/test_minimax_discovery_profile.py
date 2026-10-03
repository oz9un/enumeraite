"""Subdomain fast profiles are scoped requests, never a new backend call path."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from enumeraite.core.strategies import SubdomainDiscoveryStrategy
from enumeraite.providers.claude_provider import ClaudeProvider
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.providers.openai_compatible import OpenAICompatibleProvider
from enumeraite.providers.openai_provider import OpenAIProvider


def completion():
    return SimpleNamespace(
        model="MiniMax-M3",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "candidates": [
                                {"value": "next.example.test", "confidence": 0.8},
                                {"value": "/next", "confidence": 0.8},
                            ]
                        }
                    )
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def provider(monkeypatch, **config):
    client = Mock()
    client.chat.completions.create.return_value = completion()
    factory = Mock(return_value=client)
    monkeypatch.setattr("enumeraite.providers.openai_compatible.openai.OpenAI", factory)
    result = OpenAICompatibleProvider(
        {"preset": "minimax", "api_key": "test", "max_batch_calls": 1, **config}
    )
    return result, client, factory


def generate(instance):
    return instance.generate_paths(
        ["known.example.test"], "example.test", 1, strategy=SubdomainDiscoveryStrategy()
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, "disabled"),
        ({"subdomain_thinking": "disabled"}, "disabled"),
        ({"subdomain_thinking": "default"}, "default"),
    ],
)
def test_m3_subdomain_thinking_profile_is_explicit_scoped_and_reported(
    monkeypatch, config, expected
):
    instance, client, _ = provider(monkeypatch, **config)
    original_config = dict(instance.config)
    result = generate(instance)
    assert result.paths == ["next.example.test"]
    assert result.metadata["requested_thinking_modes"] == [expected]
    extra = client.chat.completions.create.call_args.kwargs["extra_body"]
    assert extra == (
        {"reasoning_split": True, "thinking": {"type": "disabled"}}
        if expected == "disabled"
        else {"reasoning_split": True}
    )
    assert client.chat.completions.create.call_count == 1
    assert instance._runtime_generation_kind is None
    assert instance.config == original_config


def test_ordinary_chat_path_generation_and_analysis_hook_keep_existing_behavior(monkeypatch):
    instance, client, _ = provider(monkeypatch)
    generate(instance)
    ordinary = instance.chat("system", "user", None)
    assert ordinary.requested_thinking_mode == ""
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True
    }
    result = instance.generate_paths(["/known"], "example.test", 1)
    assert result.paths == ["/next"]
    assert "requested_thinking_modes" not in result.metadata
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True
    }
    instance._make_api_call("s", "u")
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True
    }
    assert instance._runtime_generation_kind is None


@pytest.mark.parametrize(
    "preset, model",
    [
        ("minimax", "MiniMax-M2.7"),
        ("minimax", "MiniMax-M2.7-highspeed"),
        ("minimax", "MiniMax-M4"),
        ("minimax", "minimax-m3"),
        ("minimax", "MiniMax-M3-highspeed"),
        ("openrouter", "MiniMax-M3"),
        ("openai", "MiniMax-M3"),
        ("ollama", "MiniMax-M3"),
    ],
)
def test_other_models_and_presets_never_receive_minimax_thinking_switch(monkeypatch, preset, model):
    instance, client, _ = provider(monkeypatch, preset=preset, model=model)
    result = generate(instance)
    assert "thinking" not in client.chat.completions.create.call_args.kwargs.get("extra_body", {})
    assert "requested_thinking_modes" not in result.metadata
    assert instance._runtime_generation_kind is None


@pytest.mark.parametrize(
    "value",
    [True, False, None, "true", "false", "enabled", "off", "DISABLED", " disabled ", 0, 1, [], {}],
)
def test_invalid_thinking_enum_rejected_before_sdk_client_creation(monkeypatch, value):
    factory = Mock()
    monkeypatch.setattr("enumeraite.providers.openai_compatible.openai.OpenAI", factory)
    with pytest.raises(ValueError, match="subdomain_thinking"):
        OpenAICompatibleProvider(
            {"preset": "minimax", "api_key": "test", "subdomain_thinking": value}
        )
    factory.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unsupported parameter: thinking"),
        RuntimeError("unsupported response_format"),
        KeyboardInterrupt(),
    ],
)
def test_fast_profile_failure_restores_scope_without_paid_fallback(monkeypatch, error):
    instance, client, _ = provider(monkeypatch)
    client.chat.completions.create.side_effect = error
    with pytest.raises(type(error)):
        generate(instance)
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["extra_body"]["thinking"] == {
        "type": "disabled"
    }
    assert instance._runtime_generation_kind is None


class LegacyChat(LLMProvider):
    def __init__(self):
        super().__init__({"model": "legacy", "max_batch_calls": 1})
        self.seen = []

    def chat(self, system, user, schema):
        self.seen.append(self._runtime_generation_kind)
        if self._runtime_generation_kind == "subdomain":
            self._chat_for_discovery("path", system, user, schema)
            self.seen.append(self._runtime_generation_kind)
        return ChatResult('{"candidates": [{"value":"next.example.test"}]}')

    def get_provider_name(self):
        return "legacy"


def test_legacy_three_argument_chat_and_nested_generation_restore_prior_scope():
    instance = LegacyChat()
    generate(instance)
    assert instance.seen == ["subdomain", "path", "subdomain"]
    assert instance._runtime_generation_kind is None
    instance._runtime_generation_kind = "path"
    instance._chat_for_discovery("subdomain", "s", "u", None)
    assert instance._runtime_generation_kind == "path"


def test_legacy_openai_and_claude_provider_classes_remain_unchanged(monkeypatch):
    _, client, _ = provider(monkeypatch)
    legacy = OpenAIProvider({"api_key": "test", "model": "MiniMax-M3", "max_batch_calls": 1})
    generate(legacy)
    assert "extra_body" not in client.chat.completions.create.call_args.kwargs
    anthropic_client = Mock()
    anthropic_client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text='{"candidates":[{"value":"next.example.test"}]}')
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        model="claude",
        stop_reason="stop",
    )
    monkeypatch.setattr(
        "enumeraite.providers.anthropic_provider.anthropic.Anthropic",
        Mock(return_value=anthropic_client),
    )
    claude = ClaudeProvider({"api_key": "test", "max_batch_calls": 1})
    assert generate(claude).paths == ["next.example.test"]
    assert "thinking" not in anthropic_client.messages.create.call_args.kwargs
    assert claude._runtime_generation_kind is None


def test_legacy_chat_result_without_optional_thinking_metadata_still_works():
    instance = LegacyChat()
    response = SimpleNamespace(
        text='{"candidates":[{"value":"next.example.test"}]}',
        input_tokens=1,
        cached_input_tokens=0,
        output_tokens=1,
        model="legacy",
        finish_reason="stop",
        reasoning_tokens=0,
        token_logprobs=[],
    )
    instance.chat = Mock(return_value=response)
    assert generate(instance).paths == ["next.example.test"]
