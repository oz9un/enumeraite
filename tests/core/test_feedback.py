import pytest

from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.feedback import (
    FeedbackLoop,
    ValidationOutcome,
    dns_validator_adapter,
    http_path_validator_adapter,
)
from enumeraite.core.http_validator import PathHTTPResult
from enumeraite.core.models import GenerationResult
from enumeraite.core.provider import BaseProvider


class RoundProvider(BaseProvider):
    def __init__(self):
        super().__init__({})
        self.round = 0
        self.contexts = []
        self.exclusions = []

    def set_runtime_context(self, value=""):
        self.contexts.append(value)

    def set_runtime_exclusions(self, values=()):
        self.exclusions.append(list(values))

    def generate_paths(self, known, target, count, max_depth=None, strategy=None):
        self.round += 1
        values = [f"/api/v{self.round}/hit", f"/api/v{self.round}/miss"]
        return GenerationResult(
            paths=values,
            confidence_scores=[0.8, 0.7],
            metadata={
                "token_usage": {"input_tokens": 10, "output_tokens": 5},
                "reasoning_tokens": 2,
                "batch_calls": 1,
            },
        )

    def get_provider_name(self):
        return "round"


def test_feedback_loop_refeeds_hits_and_stops_on_budget():
    provider = RoundProvider()
    validator = lambda values: [
        ValidationOutcome(value, value.endswith("/hit")) for value in values
    ]
    result = FeedbackLoop(provider, validator).run(
        ["/api/v0/seed"],
        "example.com",
        "path",
        batch_size=2,
        max_rounds=2,
        max_candidates=4,
    )
    assert result.validated_hits == ["/api/v1/hit", "/api/v2/hit"]
    assert len(result.rounds) == 2
    assert any("Validated hits" in context for context in provider.contexts)
    assert provider.contexts[-1] == ""
    assert provider.exclusions[0] == []
    assert provider.exclusions[1] == ["/api/v1/hit", "/api/v1/miss"]
    assert provider.exclusions[-1] == []
    assert result.metadata["token_usage"]["total_tokens"] == 30
    assert result.metadata["token_usage"]["cached_input_tokens"] == 0
    assert result.metadata["reasoning_tokens"] == 4


def test_feedback_loop_diminishing_returns():
    provider = RoundProvider()
    validator = lambda values: [ValidationOutcome(value, False) for value in values]
    result = FeedbackLoop(provider, validator).run(
        ["/seed"], "example.com", "path", max_rounds=10, max_empty_rounds=2
    )
    assert result.stop_reason == "diminishing returns"
    assert len(result.rounds) == 2


def test_feedback_loop_stops_without_learning_validator_failures_as_misses():
    provider = RoundProvider()
    result = FeedbackLoop(provider, lambda values: []).run(
        ["/seed"], "example.com", "path", max_rounds=5
    )
    assert result.stop_reason == "validation inconclusive"
    assert len(result.rounds) == 1
    assert result.rounds[0].misses == []
    assert result.rounds[0].inconclusive == ["/api/v1/hit", "/api/v1/miss"]
    assert "Inconclusive (2)" in result.rounds[0].feedback


def test_validation_adapters_preserve_transport_errors_as_inconclusive():
    class FailedDNS:
        def validate_subdomains(self, values, detect_wildcard, apex):
            return [DNSResult(values[0], False, [], 0.1, error="timeout\nspoofed")]

    class FailedHTTP:
        def validate_paths(self, target, values):
            return [
                PathHTTPResult(
                    values[0],
                    target + values[0],
                    None,
                    False,
                    0.1,
                    error="timeout\nspoofed",
                )
            ]

    dns_outcome = dns_validator_adapter(FailedDNS(), "example.com")(["api.example.com"])[0]
    http_outcome = http_path_validator_adapter(FailedHTTP(), "https://example.com")(["/api"])[0]
    assert dns_outcome.exists is None
    assert dns_outcome.status == "error"
    assert dns_outcome.detail == "timeout spoofed"
    assert http_outcome.exists is None
    assert http_outcome.detail == "timeout spoofed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_empty_rounds": 0},
        {"min_new_hits": 0},
        {"batch_size": 0},
    ],
)
def test_feedback_loop_rejects_non_positive_stop_budgets(kwargs):
    with pytest.raises(ValueError, match="batch_size|max_empty_rounds|min_new_hits"):
        FeedbackLoop(RoundProvider(), lambda values: []).run(
            ["/seed"], "example.test", "path", **kwargs
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("batch_size", True, "batch_size"),
        ("max_rounds", 1.5, "max_rounds"),
        ("max_candidates", 5_001, "max_candidates"),
        ("known", "/seed", "known"),
        ("target", None, "target"),
    ],
)
def test_feedback_loop_rejects_ambiguous_direct_inputs(argument, value, message):
    arguments = {
        "known": ["/seed"],
        "target": "example.test",
        "kind": "path",
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=message):
        FeedbackLoop(RoundProvider(), lambda values: []).run(**arguments)


def test_feedback_loop_sanitizes_untrusted_provider_telemetry():
    provider = RoundProvider()

    def generate(*args, **kwargs):
        return GenerationResult(
            paths=["/hit"],
            confidence_scores=[0.8],
            metadata={
                "token_usage": {
                    "input_tokens": float("nan"),
                    "cached_input_tokens": "not-a-number",
                    "output_tokens": -10,
                },
                "reasoning_tokens": float("inf"),
                "batch_calls": True,
            },
        )

    provider.generate_paths = generate
    result = FeedbackLoop(provider, lambda values: [ValidationOutcome(values[0], True)]).run(
        ["/seed"], "example.test", "path", max_rounds=1
    )
    assert result.metadata["token_usage"]["total_tokens"] == 0
    assert result.metadata["reasoning_tokens"] == 0
    assert result.metadata["batch_calls"] == 0
