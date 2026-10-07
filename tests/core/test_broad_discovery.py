"""Offline regression replays for broad, budgeted hypothesis exploration."""

import copy
import json
import re
import threading
from types import SimpleNamespace

from anthropic import APITimeoutError as AnthropicTimeoutError
from openai import APITimeoutError as OpenAITimeoutError

import pytest

from enumeraite.core import discovery_session
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.scan_budget import ModelPrice, SpendLedger
from enumeraite.providers.llm_provider import ChatResult, LLMProvider

SEED = "api-prod-us.example.test"


class Model(LLMProvider):
    def __init__(self, responses=None, *, after_call=None, uncertain=False):
        super().__init__({"model": "fixture", "max_tokens": 40000})
        self.responses = responses
        self.requests = []
        self.after_call = after_call
        self.uncertain = uncertain

    def get_provider_name(self):
        return "fixture"

    def chat(self, system, user, schema):
        self.requests.append(user)
        index = len(self.requests) - 1
        count = int(re.search(r"Generate (\d+) new", user).group(1))
        names = (
            self.responses[index]
            if self.responses is not None
            else [f"batch{index}-guess{number}.example.test" for number in range(count)]
        )
        if isinstance(names, Exception):
            raise names
        if self.after_call:
            self.after_call()
        return ChatResult(
            json.dumps(
                {
                    "candidates": [
                        {"value": name, "confidence": 0.01, "rationale": "Speculative fixture"}
                        for name in names
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=100,
            usage_complete=not self.uncertain,
        )


class DNS:
    def __init__(self, hits=(SEED,)):
        self.hits = set(hits)
        self.calls = []

    def validate(self, values, **kwargs):
        self.calls.extend(values)
        return [
            DNSResult(value, value in self.hits, ["192.0.2.1"] if value in self.hits else [], 0)
            for value in values
        ]


def make_session(model=None, *, budget=1, options=None, **kwargs):
    scan = DiscoverySession(
        "example.test",
        options or ScanOptions(passive=False, max_candidates=1000),
        DNS(),
        provider=model or Model(),
        ledger=SpendLedger(budget, ModelPrice(1, 1)),
        **kwargs,
    )
    scan._enqueue(SEED, source="seed")
    scan.nodes[SEED].update(status="verified", attempts=1)
    scan.state.update(attempted=1, initialized=True)
    return scan


def test_three_distinct_passes_keep_low_confidence_names_and_checkpoint_each_batch():
    snapshots = []
    model = Model()
    scan = make_session(model, checkpoint=lambda state: snapshots.append(copy.deepcopy(state)))
    assert scan._model_candidates(scan.findings) == 60
    assert len(model.requests) == scan.ledger.calls == scan.state["model_batches"] == 3
    assert len(set(model.requests)) == 3
    assert scan.findings == [SEED]  # Hypotheses never become positive evidence.
    assert all(node["status"] == "pending" for name, node in scan.nodes.items() if name != SEED)
    # Each next paid reservation includes all earlier successful candidate records.
    reserved = [state for state in snapshots if float(state["ledger"]["reserved_usd"]) > 0]
    assert [len(state["nodes"]) for state in reserved] == [1, 21, 41]
    assert model._runtime_context == "" and model._runtime_exclusions == []
    assert model._runtime_batch_limit is None
    validate_checkpoint(scan.state, scan.target)


def test_duplicate_only_first_question_does_not_hide_productive_second_question():
    model = Model([[SEED], ["auth.example.test"], []])
    scan = make_session(model)
    assert scan._model_candidates(scan.findings) == 1
    assert len(model.requests) == 3
    assert scan.nodes["auth.example.test"]["status"] == "pending"


def test_successful_batch_survives_later_provider_failure_and_is_not_retried():
    snapshots = []
    model = Model([["auth.example.test"], RuntimeError("terminal provider failure")])
    scan = make_session(model, checkpoint=lambda state: snapshots.append(copy.deepcopy(state)))
    assert scan._model_candidates(scan.findings) == 1
    assert len(model.requests) == 2 and not scan.model_enabled
    assert snapshots[-1]["nodes"][-1]["value"] == "auth.example.test"
    assert snapshots[-1]["nodes"][-1]["status"] == "pending"
    assert scan.ledger.spent_usd > 0 and scan.ledger.reserved_usd == 0
    assert model._runtime_batch_limit is None


@pytest.mark.parametrize("error_type", [TimeoutError, OpenAITimeoutError, AnthropicTimeoutError])
def test_timeout_recovery_keeps_uncertain_charge_and_checkpoints_before_retry(error_type):
    error = (
        error_type("fixture timeout")
        if error_type is TimeoutError
        else error_type(request=SimpleNamespace(method="POST", url="https://fixture.test"))
    )
    snapshots = []
    model = Model([error, ["auth.example.test"], [], []])
    scan = make_session(model, checkpoint=lambda state: snapshots.append(copy.deepcopy(state)))

    assert scan._model_candidates(scan.findings) == 1
    assert scan.model_enabled
    assert scan.nodes["auth.example.test"]["status"] == "pending"
    assert scan.ledger.calls == 4 and scan.ledger.uncertain_calls == 1
    assert scan.ledger.reserved_usd == 0
    reservations = [
        state["ledger"] for state in snapshots if float(state["ledger"]["reserved_usd"])
    ]
    assert reservations[1]["uncertain_calls"] == 1
    assert float(reservations[1]["spent_usd"]) == float(reservations[0]["reserved_usd"])
    assert scan.ledger.spent_usd > float(reservations[0]["reserved_usd"])


def test_repeated_timeouts_end_open_ended_scan_without_claiming_discovery_exhausted(monkeypatch):
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    model = Model([TimeoutError("fixture timeout")] * 3)
    scan = make_session(
        model,
        options=ScanOptions(passive=False, max_candidates=0, max_rounds=0, time_limit=0),
    )
    result = scan.run()

    assert scan.ledger.calls == scan.ledger.uncertain_calls == 3
    assert scan.ledger.reserved_usd == 0
    assert not scan.model_enabled
    assert "timeout recovery exhausted" in result["stop_reason"]
    assert "deterministic frontier exhausted" in result["stop_reason"]
    assert result["findings"] == [SEED]


def test_terminal_provider_failure_still_verifies_deterministic_hypotheses():
    model = Model([RuntimeError("terminal provider failure")])
    scan = make_session(
        model,
        options=ScanOptions(
            passive=False,
            max_candidates=0,
            max_rounds=0,
            time_limit=0,
            max_depth=1,
            max_number=2,
            max_per_host=2,
        ),
    )
    known, sibling = "api-01.example.test", "api-02.example.test"
    scan._enqueue(known, source="seed")
    scan.nodes[known].update(status="verified", attempts=1)
    scan.validator.hits.add(sibling)
    result = scan.run()

    assert scan.ledger.calls == scan.ledger.uncertain_calls == 1
    assert sibling in result["findings"]
    assert scan.nodes[sibling]["source"] == "pattern"
    assert "provider failed" in result["stop_reason"]
    assert "deterministic frontier exhausted" in result["stop_reason"]


def test_timeout_recovery_cannot_bypass_exhausted_budget():
    model = Model([TimeoutError("fixture timeout")])
    scan = make_session(model, budget=0.02)

    assert scan._model_candidates(scan.findings) == 0
    assert scan.ledger.calls == scan.ledger.uncertain_calls == 1
    assert scan.ledger.reserved_usd == 0
    assert scan.ledger.spent_usd <= scan.ledger.limit_usd
    assert not scan.model_enabled
    assert any("budget exhausted" in warning for warning in scan.state["warnings"])


@pytest.mark.parametrize("stop_kind", ["pause", "deadline"])
def test_timeout_recovery_respects_stop_before_next_paid_attempt(stop_kind):
    pause = threading.Event()
    model = Model([TimeoutError("fixture timeout")])
    scan = make_session(model, stop_requested=pause.is_set)

    def stop_after_uncertain_charge(state):
        if state["ledger"]["uncertain_calls"]:
            if stop_kind == "pause":
                pause.set()
            else:
                scan.deadline = 0

    scan.checkpoint = stop_after_uncertain_charge
    result = scan.run()

    assert scan.ledger.calls == scan.ledger.uncertain_calls == 1
    assert scan.ledger.reserved_usd == 0
    assert result["stop_reason"] == (
        "paused; scan saved" if stop_kind == "pause" else "time limit reached"
    )


@pytest.mark.parametrize("open_ended", [False, True])
def test_successful_batch_survives_exhausted_budget_without_another_request(open_ended):
    model = Model(uncertain=True)
    options = (
        ScanOptions(passive=False, max_candidates=0, max_rounds=0, time_limit=0)
        if open_ended
        else None
    )
    scan = make_session(model, budget=0.02, options=options)
    assert scan._model_candidates(scan.findings) == 20
    assert len(model.requests) == 1 and not scan.model_enabled
    assert scan.ledger.uncertain_calls == 1
    assert scan.ledger.spent_usd <= scan.ledger.limit_usd
    assert any("budget exhausted" in warning for warning in scan.state["warnings"])


@pytest.mark.parametrize("stop_kind", ["pause", "deadline"])
def test_stop_between_passes_preserves_first_batch(stop_kind):
    pause = threading.Event()
    model = Model()
    scan = make_session(model, stop_requested=pause.is_set)
    model.after_call = pause.set if stop_kind == "pause" else lambda: setattr(scan, "deadline", 0)
    assert scan._model_candidates(scan.findings) == 20
    assert len(model.requests) == 1
    assert model._runtime_batch_limit is None


@pytest.mark.parametrize("remaining,expected", [(1, 1), (7, 5), (30, 20), (99, 60), (None, 60)])
def test_model_passes_leave_room_for_python_and_respect_small_candidate_caps(remaining, expected):
    scan = make_session(
        options=ScanOptions(
            passive=False,
            max_candidates=remaining + 1 if remaining is not None else 0,
            time_limit=0,
        )
    )
    assert scan._model_candidates(scan.findings) == expected
    assert len(scan.nodes) == expected + 1
    assert scan.ledger.calls == (expected + 19) // 20


def test_new_default_keeps_exploring_after_two_no_hit_rounds(monkeypatch):
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    scan = make_session(options=ScanOptions(passive=False, max_candidates=1000, max_rounds=4))
    result = scan.run()
    assert result["round"] == 4 and result["model_batches"] == 12
    assert result["attempted"] == 241
    assert result["stop_reason"] == "round limit reached"
    assert len(scan.validator.calls) == len(set(scan.validator.calls)) == 240


def test_explicit_stale_limit_remains_authoritative(monkeypatch):
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    scan = make_session(options=ScanOptions(passive=False, max_rounds=10, stale_rounds=2))
    result = scan.run()
    assert result["round"] == 2 and result["model_batches"] == 6
    assert result["stop_reason"] == "diminishing returns"


def test_late_verified_discovery_is_not_hidden_by_two_miss_rounds(monkeypatch):
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    late_hit = "batch8-guess0.example.test"
    broad = make_session(options=ScanOptions(passive=False, max_rounds=3, find=2))
    broad.validator.hits.add(late_hit)
    assert broad.run()["findings"] == [SEED, late_hit]
    stale = make_session(options=ScanOptions(passive=False, max_rounds=3, stale_rounds=2, find=2))
    stale.validator.hits.add(late_hit)
    assert stale.run()["findings"] == [SEED]
    assert late_hit not in stale.validator.calls


def test_resume_retains_batch_cursor_exclusions_and_spend():
    first = make_session()
    first._model_candidates(first.findings)
    state = copy.deepcopy(first.state)
    second_model = Model([["batch0-guess0.example.test", "new.example.test"], [], []])
    second = DiscoverySession(
        "example.test",
        first.options,
        DNS(),
        provider=second_model,
        ledger=SpendLedger.from_dict(state["ledger"]),
        state=state,
    )
    assert second.state["model_batches"] == 3
    assert second._model_candidates(second.findings) == 1
    assert second.state["model_batches"] == 6
    assert len(second.nodes) == len(first.nodes) + 1
    assert second.ledger.spent_usd > first.ledger.spent_usd
    validate_checkpoint(second.state, second.target)


@pytest.mark.parametrize("value", [True, -1, "2", None])
def test_checkpoint_rejects_invalid_batch_cursor(value):
    scan = make_session()
    scan.save()
    scan.state["model_batches"] = value
    with pytest.raises(ValueError, match="model batch"):
        validate_checkpoint(scan.state, scan.target)


def test_legacy_checkpoint_without_batch_cursor_loads():
    scan = make_session()
    scan.save()
    del scan.state["model_batches"]
    validate_checkpoint(scan.state, scan.target)
    restored = DiscoverySession("example.test", scan.options, DNS(), state=scan.state)
    assert restored.state["model_batches"] == 0


def test_old_pattern_frontier_reopens_once_without_erasing_history_or_limits():
    scan = make_session()
    scan._model_candidates(scan.findings)
    scan.nodes[SEED]["expanded_revision"] = 1
    scan.state.update(stale=2, round=2)
    scan.save()
    original = copy.deepcopy(scan.state)
    del original["pattern_paging_version"]
    restored = DiscoverySession("example.test", scan.options, DNS(), state=copy.deepcopy(original))
    assert restored.nodes[SEED]["expanded_revision"] == 0
    assert restored.state["stale"] == 0
    for field in ("attempted", "round", "options", "ledger", "model_batches"):
        assert restored.state[field] == original[field]
    assert set(restored.nodes) == set(scan.nodes)
    restored.nodes[SEED]["expanded_revision"] = 1
    restored.state["stale"] = 1
    again = DiscoverySession("example.test", scan.options, DNS(), state=restored.state)
    assert again.nodes[SEED]["expanded_revision"] == again.state["stale"] == 1


@pytest.mark.parametrize("value", [True, 0, "1", None])
def test_checkpoint_rejects_unknown_pattern_paging_version(value):
    scan = make_session()
    scan.save()
    scan.state["pattern_paging_version"] = value
    with pytest.raises(ValueError):
        validate_checkpoint(scan.state, scan.target)


def test_checkpoint_rejects_future_pattern_paging_version():
    scan = make_session()
    scan.save()
    scan.state["pattern_paging_version"] += 1
    with pytest.raises(ValueError):
        validate_checkpoint(scan.state, scan.target)
