"""Scoped generation limits must not silently refill a discovery pass."""

import json

import pytest

from enumeraite.core.strategies import SubdomainDiscoveryStrategy
from enumeraite.providers.llm_provider import ChatResult, LLMProvider


class RecordingLLM(LLMProvider):
    def __init__(self, values=None, config=None):
        super().__init__({"model": "fake", "batch_size": 2, "max_batch_calls": 4, **(config or {})})
        self.values = values
        self.calls = 0

    def chat(self, system, user, schema):
        self.calls += 1
        values = self.values
        if values is None:
            values = [f"host{self.calls}.example.test"]
        return ChatResult(
            text=json.dumps(
                {
                    "candidates": [
                        {"value": value, "confidence": 0.8, "rationale": "test"} for value in values
                    ]
                }
            ),
            input_tokens=10,
            cached_input_tokens=3,
            output_tokens=5,
        )

    def get_provider_name(self):
        return "fake"


def generate(provider, count=5):
    return provider.generate_paths(
        ["known.example.test"],
        "example.test",
        count,
        strategy=SubdomainDiscoveryStrategy(),
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["new.example.test"], ["new.example.test"]),
        ([], []),
        (["known.example.test"], []),
        (["new.example.test", "new.example.test"], ["new.example.test"]),
    ],
    ids=["underfilled", "empty", "already-known", "duplicates"],
)
def test_one_batch_limit_never_refills(values, expected):
    provider = RecordingLLM(values)
    provider.set_runtime_batch_limit(1)

    result = generate(provider)

    assert provider.calls == 1
    assert result.paths == expected
    assert result.metadata["batch_calls"] == 1
    assert result.metadata["token_usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 15,
    }


@pytest.mark.parametrize("limit", [True, False, 0, -1, 101, 1.0, "1", float("nan"), []])
def test_runtime_limit_rejects_malformed_values_without_losing_existing_limit(limit):
    provider = RecordingLLM()
    provider.set_runtime_batch_limit(1)

    with pytest.raises(ValueError, match="runtime_batch_limit"):
        provider.set_runtime_batch_limit(limit)

    assert generate(provider).metadata["batch_calls"] == 1


@pytest.mark.parametrize("limit", [1, 100])
def test_runtime_limit_accepts_inclusive_bounds(limit):
    provider = RecordingLLM()
    provider.set_runtime_batch_limit(limit)
    assert generate(provider).metadata["batch_calls"] == min(limit, 4)


@pytest.mark.parametrize("clear_with_default", [False, True])
def test_clearing_restores_default_behavior_without_config_mutation(clear_with_default):
    provider = RecordingLLM()
    initial_config = dict(provider.config)
    provider.set_runtime_batch_limit(1)
    assert generate(provider).metadata["batch_calls"] == 1

    if clear_with_default:
        provider.set_runtime_batch_limit()
    else:
        provider.set_runtime_batch_limit(None)
    result = generate(provider)

    assert result.metadata["batch_calls"] == 4
    assert len(result.paths) == 4
    assert provider.config == initial_config
    assert provider.max_batch_calls == 4


def test_runtime_limit_cannot_increase_configured_call_budget():
    provider = RecordingLLM(config={"max_batch_calls": 2})
    provider.set_runtime_batch_limit(100)
    assert generate(provider).metadata["batch_calls"] == 2


def test_runtime_limit_cannot_increase_calculated_call_budget():
    provider = RecordingLLM(config={"max_batch_calls": 100, "batch_size": 50})
    provider.set_runtime_batch_limit(100)
    assert generate(provider).metadata["batch_calls"] == 3


def test_runtime_limit_keeps_existing_early_empty_response_stop():
    provider = RecordingLLM([], config={"max_batch_calls": 100})
    provider.set_runtime_batch_limit(100)
    assert generate(provider, count=100).metadata["batch_calls"] == 3


def test_default_generation_behavior_is_unchanged():
    provider = RecordingLLM()
    assert generate(provider).metadata["batch_calls"] == 4


def test_runtime_limit_does_not_force_model_calls_for_full_structural_pool():
    provider = RecordingLLM(config={"hybrid": True})
    provider.set_runtime_batch_limit(1)

    result = provider.generate_paths(
        ["api1.example.test", "api2.example.test"],
        "example.test",
        1,
        strategy=SubdomainDiscoveryStrategy(),
    )

    assert len(result.paths) == 1
    assert provider.calls == 0
    assert result.metadata["batch_calls"] == 0
    assert result.metadata["model_skipped_reason"] == "structural pool filled candidate budget"


def test_runtime_limit_preserves_exclusions_and_scope_checks():
    provider = RecordingLLM(["excluded.example.test", "outside.test", "new.example.test"])
    provider.set_runtime_batch_limit(1)
    provider.set_runtime_exclusions(["excluded.example.test"])

    result = generate(provider)

    assert result.paths == ["new.example.test"]
    assert result.metadata["runtime_rejected"] == 1
    assert result.metadata["scope_rejected"] == 1
    assert result.metadata["batch_calls"] == 1
