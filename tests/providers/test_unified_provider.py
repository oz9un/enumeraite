import json
import math

import pytest

from enumeraite.core.config import Config
from enumeraite.core.factory import ProviderFactory
from enumeraite.core.strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.providers.openai_compatible import (
    OpenAICompatibleProvider,
    _safe_int,
    _validated_base_url,
)
from enumeraite.providers.openai_provider import OpenAIProvider


class FakeLLM(LLMProvider):
    def __init__(self, config=None):
        super().__init__({"model": "fake", "max_batch_calls": 1, **(config or {})})
        self.last_user = ""

    def chat(self, system, user, schema):
        self.last_user = user
        return ChatResult(
            text=json.dumps(
                {
                    "candidates": [
                        {
                            "value": "/api/v2/users",
                            "confidence": 0.91,
                            "rationale": "v1 exists",
                        },
                        {
                            "value": "/api/v2/orders",
                            "confidence": 0.42,
                            "rationale": "users pattern",
                        },
                    ]
                }
            ),
            input_tokens=10,
            output_tokens=20,
            model="fake",
        )

    def get_provider_name(self):
        return "fake"


def test_unified_generation_uses_model_confidence_and_context():
    provider = FakeLLM()
    result = provider.generate_paths(
        ["/api/v1/users", "/api/v1/orders"],
        "example.com",
        10,
        strategy=PathDiscoveryStrategy(),
    )
    assert result.confidence_scores == [0.91, 0.42]
    assert result.metadata["token_usage"]["total_tokens"] == 30
    assert result.metadata["token_usage"]["cached_input_tokens"] == 0
    assert "common path prefix: /api/v1" in provider.last_user


def test_confidence_modes_are_controls():
    uniform = FakeLLM({"confidence": "uniform"}).generate_paths(
        ["/api/v1/users", "/api/v1/orders"],
        "example.com",
        2,
        strategy=PathDiscoveryStrategy(),
    )
    assert uniform.confidence_scores == [1.0, 1.0]
    heuristic = FakeLLM({"confidence": "heuristic"}).generate_paths(
        ["/api/v1/users", "/api/v1/orders"],
        "example.com",
        2,
        strategy=PathDiscoveryStrategy(),
    )
    assert heuristic.confidence_scores != [0.91, 0.42]


def test_logprob_confidence_aligns_exact_candidate_span():
    logprobs = [
        ('{"value":"', -4.0),
        ("/api", -0.1),
        ("/users", -0.3),
        ('","rationale":"unlikely"}', -8.0),
    ]
    score = FakeLLM._logprob_confidence("/api/users", logprobs)
    assert score == pytest.approx(math.exp(-0.2))
    assert FakeLLM._logprob_confidence("/not-present", logprobs) == 0.5


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"temperature": float("nan")}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"timeout": float("inf")}, "timeout"),
        ({"max_retries": 11}, "max_retries"),
        ({"hybrid_model_fraction": float("nan")}, "hybrid_model_fraction"),
        ({"batch_size": 501}, "batch_size"),
        ({"max_batch_calls": 101}, "max_batch_calls"),
        ({"seed_limit": 5001}, "seed_limit"),
        ({"seed_char_limit": 1_000_001}, "seed_char_limit"),
        ({"max_tokens": "100"}, "max_tokens"),
        ({"temperature": True}, "temperature"),
    ],
)
def test_provider_resource_limits_reject_unsafe_config(config, message):
    with pytest.raises(ValueError, match=message):
        FakeLLM(config)


@pytest.mark.parametrize(
    "config",
    [
        {"context": "sometimes"},
        {"hybrid": "enabled"},
        {"context": object()},
    ],
)
def test_provider_rejects_ambiguous_boolean_settings(config):
    with pytest.raises(ValueError, match="boolean provider settings"):
        FakeLLM(config)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("count", True, "count"),
        ("count", 2.5, "count"),
        ("max_depth", False, "max_depth"),
        ("max_depth", 1.5, "max_depth"),
        ("known_paths", "/seed", "known_paths"),
        ("target", None, "target"),
    ],
)
def test_unified_provider_rejects_ambiguous_direct_api_inputs(argument, value, message):
    arguments = {
        "known_paths": ["/seed"],
        "target": "example.test",
        "count": 2,
        "max_depth": 2,
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=message):
        FakeLLM().generate_paths(**arguments)


def test_unified_provider_enforces_max_depth_on_model_outputs():
    provider = FakeLLM()

    def chat(system, user, schema):
        return ChatResult(
            text=json.dumps(
                {
                    "candidates": [
                        {"value": "/allowed", "confidence": 0.7, "rationale": "one"},
                        {"value": "/too/deep", "confidence": 0.9, "rationale": "two"},
                    ]
                }
            )
        )

    provider.chat = chat
    result = provider.generate_paths(["/seed"], "example.com", 2, max_depth=1)
    assert result.paths == ["/allowed"]
    assert result.metadata["depth_rejected"] == 1


def test_openai_registry_interlock():
    factory = ProviderFactory(Config(providers={"openai": {"api_key": "test"}}))
    assert factory._registry["openai"] is OpenAIProvider
    assert "openrouter" in factory._registry


def test_subdomain_generation_rejects_cross_apex_candidates():
    provider = FakeLLM()

    def chat(system, user, schema):
        return ChatResult(
            text=json.dumps(
                {
                    "candidates": [
                        {
                            "value": "api.example.com",
                            "confidence": 0.8,
                            "rationale": "in scope",
                        },
                        {
                            "value": "api.example.net",
                            "confidence": 0.9,
                            "rationale": "wrong apex",
                        },
                    ]
                }
            )
        )

    provider.chat = chat
    result = provider.generate_paths(
        ["dev.example.com", "prod.example.com"],
        "example.com",
        10,
        strategy=SubdomainDiscoveryStrategy(),
    )
    assert result.paths == ["api.example.com"]
    assert result.metadata["scope_rejected"] == 1


def test_subdomain_generation_does_not_prompt_with_cross_apex_seeds():
    provider = FakeLLM()
    result = provider.generate_paths(
        ["dev.example.com", "outside.test"],
        "example.com",
        2,
        strategy=SubdomainDiscoveryStrategy(),
    )
    assert "outside.test" not in provider.last_user
    assert result.metadata["seeds_valid"] == 2
    assert result.metadata["seeds_in_scope"] == 1
    assert result.metadata["seed_scope_rejected"] == 1


def test_generation_fills_large_budgets_across_bounded_batches():
    class BatchLLM(FakeLLM):
        def __init__(self):
            super().__init__({"batch_size": 2, "max_batch_calls": 4})
            self.call_number = 0

        def chat(self, system, user, schema):
            self.call_number += 1
            start = (self.call_number - 1) * 2
            return ChatResult(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "value": f"/generated/item-{index}",
                                "confidence": 1 - index / 10,
                                "rationale": "batch",
                            }
                            for index in range(start, start + 2)
                        ]
                    }
                ),
                input_tokens=5,
                output_tokens=7,
            )

    result = BatchLLM().generate_paths(["/known"], "example.com", 5)
    assert len(result.paths) == 5
    assert result.metadata["batch_calls"] == 3
    assert result.metadata["token_usage"]["total_tokens"] == 36
    assert result.metadata["fill_rate"] == 1.0


def test_generation_evenly_bounds_large_seed_prompts():
    provider = FakeLLM({"seed_limit": 3, "seed_char_limit": 1000})
    result = provider.generate_paths(
        [f"/seed/{index}" for index in range(10)],
        "example.test",
        2,
    )
    assert result.metadata["seeds_received"] == 10
    assert result.metadata["seeds_used"] == 3
    assert result.metadata["seeds_truncated"] == 7
    assert '"/seed/0"' in provider.last_user
    assert '"/seed/4"' in provider.last_user
    assert '"/seed/9"' in provider.last_user
    assert '"/seed/1"' not in provider.last_user


def test_generation_requires_at_least_one_valid_seed():
    provider = FakeLLM()
    try:
        provider.generate_paths(["not-a-path"], "example.test", 2)
    except ValueError as exc:
        assert "no valid path seeds" in str(exc)
    else:
        raise AssertionError("expected invalid seed rejection")


def test_generation_rejects_known_route_shape_with_another_id():
    provider = FakeLLM()
    provider.chat = lambda system, user, schema: ChatResult(
        text=json.dumps(
            {
                "candidates": [
                    {
                        "value": "/users/456",
                        "confidence": 0.9,
                        "rationale": "same route",
                    },
                    {
                        "value": "/orders/456",
                        "confidence": 0.8,
                        "rationale": "new route",
                    },
                ]
            }
        )
    )
    result = provider.generate_paths(["/users/{userId}"], "example.com", 2)
    assert result.paths == ["/orders/456"]
    assert result.metadata["known_rejected"] == 1


def test_hybrid_pool_ranks_high_evidence_structure_with_source_metadata():
    provider = FakeLLM({"hybrid": True})
    result = provider.generate_paths(
        ["/api/v1/users/{userId}/profile", "/api/v1/orders"],
        "example.com",
        3,
    )
    assert result.paths[0] == "/api/v1/users/123"
    assert result.metadata["candidate_sources"][result.paths[0]] == "structural"


def test_hybrid_skips_paid_call_when_structure_fills_small_budget():
    provider = FakeLLM({"hybrid": True})
    calls = {"count": 0}

    def chat(system, user, schema):
        calls["count"] += 1
        return ChatResult(text='{"candidates": []}')

    provider.chat = chat
    result = provider.generate_paths(
        ["/api/v1/users/{userId}/profile", "/api/v1/orders/123/history"],
        "example.test",
        2,
    )
    assert len(result.paths) == 2
    assert calls["count"] == 0
    assert result.metadata["model_budget"] == 0
    assert result.metadata["model_skipped_reason"] == "structural pool filled candidate budget"


def test_provider_never_accepts_more_model_candidates_than_requested():
    result = FakeLLM().generate_paths(["/seed"], "example.com", 1)
    assert result.paths == ["/api/v2/users"]
    assert result.metadata["overflow_rejected"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -3, True, "12", None])
def test_provider_token_counts_are_nonnegative_and_json_safe(value):
    assert _safe_int(value) == 0


def test_chat_result_normalizes_backend_telemetry():
    result = ChatResult(
        text="ok",
        input_tokens=100,
        cached_input_tokens=200,
        output_tokens=float("nan"),
        reasoning_tokens=-2,
    )
    assert result.input_tokens == 100
    assert result.cached_input_tokens == 100
    assert result.output_tokens == 0
    assert result.reasoning_tokens == 0


def test_provider_base_url_protects_hosted_api_keys():
    assert (
        _validated_base_url("https://api.example.test/v1/", local=False)
        == "https://api.example.test/v1"
    )
    assert (
        _validated_base_url("http://127.0.0.1:8000/v1", local=False) == "http://127.0.0.1:8000/v1"
    )
    assert (
        _validated_base_url("http://model-server:8000/v1", local=True)
        == "http://model-server:8000/v1"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        _validated_base_url("http://api.example.test/v1", local=False)
    with pytest.raises(ValueError, match="credentials"):
        _validated_base_url("https://token@api.example.test/v1", local=False)
    with pytest.raises(ValueError, match="query"):
        _validated_base_url("https://api.example.test/v1?debug=1", local=False)
    with pytest.raises(ValueError, match="invalid hostname"):
        _validated_base_url("https://api.example%0a.test/v1", local=False)
    with pytest.raises(ValueError, match="invalid port"):
        _validated_base_url("https://api.example.test:notaport/v1", local=False)


def test_openai_compatible_rejects_cleartext_remote_host_before_client_creation():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleProvider(
            {
                "preset": "minimax",
                "api_key": "placeholder",
                "base_url": "http://api.example.test/v1",
            }
        )
