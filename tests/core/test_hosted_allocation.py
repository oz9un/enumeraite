"""An explicitly selected hosted model gets room before the pattern frontier."""

import json
import threading

import pytest

from enumeraite.core import discovery_session
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.models import GenerationResult
from enumeraite.core.scan_budget import ModelPrice, SpendLedger
from enumeraite.providers.llm_provider import ChatResult, LLMProvider


class FixtureDNS:
    def __init__(self, hits):
        self.hits = set(hits)
        self.calls = []

    def validate(self, values, **kwargs):
        self.calls.extend(values)
        return [
            DNSResult(value, value in self.hits, ["192.0.2.1"] if value in self.hits else [], 0)
            for value in values
        ]


class FixtureModel(LLMProvider):
    def __init__(self, candidates=("model-only.example.test",)):
        super().__init__({"model": "fixture", "hybrid": False, "max_batch_calls": 1})
        self.candidates = candidates
        self.requests = []

    def chat(self, system, user, schema):
        self.requests.append(user)
        return ChatResult(
            json.dumps(
                {
                    "candidates": [
                        {
                            "value": value,
                            "confidence": 0.9,
                            "rationale": "Fixture contextual hypothesis",
                        }
                        for value in self.candidates
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=20,
        )

    def get_provider_name(self):
        return "fixture-model"


def session_with_verified(*, budget=1, max_candidates=100, stop=None):
    model = FixtureModel()
    ledger = SpendLedger(budget, ModelPrice(1, 2))
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=max_candidates),
        FixtureDNS({"ny-01-iphone.example.test"}),
        provider=model,
        ledger=ledger,
        stop_requested=stop,
    )
    session._enqueue("ny-01-iphone.example.test", source="seed")
    session.nodes["ny-01-iphone.example.test"].update(status="verified", attempts=1)
    session.state["attempted"] = 1
    return session, model, ledger


def test_many_known_hosts_do_not_starve_hosted_hypotheses_before_dns_validation():
    seeds = [f"ny-{number:02d}-iphone.example.test" for number in range(1, 41)]
    hypothesis = "model-only.example.test"
    events = []
    model = FixtureModel([hypothesis])
    ledger = SpendLedger(1, ModelPrice(1, 2))
    validator = FixtureDNS([*seeds, hypothesis])
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=70, max_rounds=1),
        validator,
        provider=model,
        ledger=ledger,
        event=lambda kind, data: events.append((kind, data)),
    )
    report = session.run(seeds)
    assert len(model.requests) == 3
    assert "ny-01-iphone.example.test" in model.requests[0]
    assert "ny-40-iphone.example.test" in model.requests[0]
    assert ledger.calls == 3 and ledger.spent_usd > 0
    assert session.nodes[hypothesis]["source"] == "model"
    assert session.nodes[hypothesis]["status"] == "verified"
    assert hypothesis in validator.calls
    assert report["known_verified"] == 40
    assert report["attempted"] == 70
    candidates = [data for kind, data in events if kind == "candidate"]
    model_position = next(i for i, data in enumerate(candidates) if data["value"] == hypothesis)
    pattern_position = next(i for i, data in enumerate(candidates) if data["source"] == "pattern")
    assert model_position < pattern_position
    assert candidates[model_position]["status"] == "pending"
    assert candidates[model_position]["reason"] == "Fixture contextual hypothesis"


def test_hosted_guesses_are_not_reported_as_findings_until_dns_verifies_them():
    session, model, _ = session_with_verified()
    assert session._expand()
    assert len(model.requests) == 3
    node = session.nodes["model-only.example.test"]
    assert node["source"] == "model" and node["status"] == "pending"
    assert node["parent"] is None
    assert "model-only.example.test" not in session.findings
    session._validate_batch(["model-only.example.test"])
    assert node["status"] == "miss"
    assert "model-only.example.test" not in session.findings


def test_empty_budget_leaves_all_remaining_capacity_for_free_patterns():
    session, model, ledger = session_with_verified(budget=0)
    assert session._expand()
    assert not model.requests
    assert ledger.calls == 0
    assert not session.model_enabled
    assert any(node["source"] == "pattern" for node in session.nodes.values())
    assert all(node["source"] != "model" for node in session.nodes.values())
    assert any("budget exhausted" in warning for warning in session.state["warnings"])


@pytest.mark.parametrize("stop_kind", ["pause", "deadline", "full", "goal", "blocked", "reserved"])
def test_no_provider_request_when_limits_or_budget_disallow_it(stop_kind):
    pause = threading.Event()
    session, model, ledger = session_with_verified(
        stop=pause.is_set, max_candidates=1 if stop_kind == "full" else 100
    )
    if stop_kind == "pause":
        pause.set()
    elif stop_kind == "deadline":
        session.deadline = 0
    elif stop_kind == "goal":
        session.options.find = 1
    elif stop_kind == "blocked":
        ledger.blocked = True
    elif stop_kind == "reserved":
        ledger.reserve(1, 1)
    before = ledger.calls
    session._expand()
    assert not model.requests
    assert ledger.calls == before
    assert all(node["source"] != "model" for node in session.nodes.values())


@pytest.mark.parametrize("remaining,expected", [(1, 1), (7, 5), (99, 60)])
def test_model_allocation_is_bounded_even_if_provider_returns_too_many(
    monkeypatch, remaining, expected
):
    session, _, _ = session_with_verified(max_candidates=remaining + 1)
    requests = []

    class OverflowEngine:
        def __init__(self, provider):
            pass

        def generate_paths(self, evidence, target, count, **kwargs):
            requests.append(count)
            return GenerationResult(
                paths=[f"pass{len(requests)}-host-{i}.example.test" for i in range(50)],
                confidence_scores=[0.9] * 50,
                metadata={},
            )

    monkeypatch.setattr(discovery_session, "GenerationEngine", OverflowEngine)
    session._expand()
    assert sum(requests) == expected and all(count <= 20 for count in requests)
    assert sum(node["source"] == "model" for node in session.nodes.values()) == expected
    assert len(session.nodes) <= remaining + 1


def test_free_scans_still_expand_without_constructing_a_generation_engine(monkeypatch):
    monkeypatch.setattr(
        discovery_session,
        "GenerationEngine",
        lambda provider: pytest.fail("Free discovery must not construct a model engine"),
    )
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=50),
        FixtureDNS({"ny-01-iphone.example.test", "ny-02-iphone.example.test"}),
    )
    result = session.run(["ny-01-iphone.example.test"])
    assert "ny-02-iphone.example.test" in result["findings"]
    assert session.nodes["ny-02-iphone.example.test"]["source"] == "pattern"
    assert result["ledger"] is None
