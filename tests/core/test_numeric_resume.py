"""Numeric range ceilings constrain deterministic changes, not literal names."""

import copy
import json

import pytest

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_validator import DNSResult


class OfflineDNS:
    def __init__(self):
        self.calls = []

    def validate(self, values, *, deadline):
        self.calls.extend(values)
        return [DNSResult(value, False, [], 0) for value in values]


def session(*, ceiling=9, depth=4, state=None):
    return DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_number=ceiling, max_candidates=100, max_depth=depth),
        OfflineDNS(),
        state=state,
    )


def enqueue_pair(scan, parent, candidate, *, reason="", source="pattern", depth=1):
    parent += ".example.test"
    candidate += ".example.test"
    scan._enqueue(parent, source="seed")
    scan.nodes[parent].update(status="verified", attempts=1, addresses=["192.0.2.1"])
    scan.state["attempted"] += 1
    scan._enqueue(candidate, source=source, parent=parent, reason=reason, depth=depth)
    return scan.nodes[candidate]


@pytest.mark.parametrize(
    ("parent", "candidate", "ceiling", "allowed"),
    [
        ("worker9", "worker10", 9, False),
        ("worker9", "worker10", 10, True),
        ("worker09", "worker010", 9, False),
        ("worker09", "worker010", 10, True),
        ("worker099", "worker100", 99, False),
        ("209-64-116-001", "209-64-116-002", 2, True),
        ("209-64-116-001", "209-64-116-003", 2, False),
        ("rack209-worker01", "rack209-worker02", 2, True),
        ("rack209-worker01", "rack210-worker02", 2, False),
        ("ny-01-iphone", "ca-10-ipad", 9, False),
        ("ny-01-iphone", "ca-09-ipad", 9, True),
        ("api209-prod", "api209-stage", 1, True),
        ("app099", "app99", 1, True),
    ],
)
def test_only_changed_corresponding_numeric_slots_are_limited(parent, candidate, ceiling, allowed):
    scan = session(ceiling=ceiling)
    node = enqueue_pair(scan, parent, candidate)
    assert scan._pending_allowed(node) is allowed


@pytest.mark.parametrize("source", ["seed", "passive", "model", "bootstrap"])
def test_explicit_passive_and_model_names_are_not_subject_to_numeric_expansion_ceiling(source):
    scan = session(ceiling=1)
    node = enqueue_pair(scan, "worker1", "worker999", source=source)
    assert scan._pending_allowed(node)


def test_numbered_observed_children_and_fixed_parent_numbers_are_vocabulary_not_ranges():
    scan = session(ceiling=1)
    child = enqueue_pair(
        scan,
        "209-64-116-099",
        "v200.209-64-116-099",
        reason="child label observed below a host with the same naming pattern",
    )
    assert scan._pending_allowed(child)
    child["depth"] = 5
    assert not scan._pending_allowed(child)
    child["depth"] = 1
    scan.options.max_depth = 1
    assert not scan._pending_allowed(child)


@pytest.mark.parametrize("parent", [None, "missing.example.test", [], 123])
@pytest.mark.parametrize(
    "reason",
    [
        "numeric-slot hypothesis preserving observed zero padding",
        "Systematic numeric family enumeration; fixed tokens preserved",
    ],
)
def test_missing_numeric_provenance_keeps_unmistakable_above_ceiling_rules_dormant(parent, reason):
    scan = session(ceiling=2)
    scan._enqueue("worker003.example.test", source="pattern", reason=reason)
    node = scan.nodes["worker003.example.test"]
    node["parent"] = parent
    assert not scan._pending_allowed(node)
    scan.options.max_number = 3
    assert scan._pending_allowed(node)


def test_malformed_parent_record_is_conservative_only_for_unmistakable_numeric_generation():
    scan = session(ceiling=2)
    node = enqueue_pair(scan, "worker001", "worker003", reason="numeric-slot hypothesis")
    scan.nodes[node["parent"]]["value"] = "different.example.test"
    assert not scan._pending_allowed(node)
    node["reason"] = "child label observed below a host with the same naming pattern"
    assert scan._pending_allowed(node)


def test_unmappable_numeric_generation_is_not_guessed_from_shuffled_literal_structure():
    scan = session(ceiling=2)
    node = enqueue_pair(scan, "worker001.a", "a.worker003", reason="numeric-slot hypothesis")
    assert not scan._pending_allowed(node)
    node["reason"] = "legacy untyped naming hypothesis"
    assert scan._pending_allowed(node)


def test_reduction_keeps_saved_numeric_queue_dormant_and_later_increase_wakes_it_once():
    first = session(ceiling=5, depth=1)
    for name in ["app01.example.test", "app02.example.test"]:
        first._enqueue(name, source="seed")
        first.nodes[name].update(status="verified", attempts=1, addresses=["192.0.2.1"])
    first.state.update(initialized=True, attempted=2)
    assert first._expand()
    queued = {name for name, node in first.nodes.items() if node["status"] == "pending"}
    assert queued == {f"app0{number}.example.test" for number in (3, 4, 5)}
    original_fields = set(first.nodes["app03.example.test"])
    # A real serialized checkpoint must not require new mandatory fields.
    narrowed = session(ceiling=2, depth=1, state=json.loads(json.dumps(first.state)))
    summary = narrowed.run()
    assert summary["stop_reason"] == "remaining queued work is outside current limits"
    assert summary["queued"] == len(queued)
    assert narrowed.validator.calls == []
    assert narrowed.findings == ["app01.example.test", "app02.example.test"]
    assert all(narrowed.nodes[name]["status"] == "pending" for name in queued)
    assert all(narrowed.nodes[name]["attempts"] == 0 for name in queued)
    assert set(narrowed.nodes["app03.example.test"]) == original_fields
    validate_checkpoint(narrowed.state, "example.test")
    widened = session(ceiling=5, depth=1, state=copy.deepcopy(narrowed.state))
    widened.run()
    assert set(widened.validator.calls) == queued
    assert len(widened.validator.calls) == len(queued)
    assert all(widened.nodes[name]["attempts"] == 1 for name in queued)
    validate_checkpoint(widened.state, "example.test")


def test_numeric_retry_allowance_is_not_spent_while_dormant():
    scan = session(ceiling=2)
    node = enqueue_pair(scan, "app01", "app03", reason="numeric-slot hypothesis")
    node.update(status="inconclusive", attempts=1)
    scan.state.update(initialized=True, attempted=2)
    scan.run()
    assert node["status"] == "pending" and node["attempts"] == 1
    assert "app03.example.test" not in scan.validator.calls
    resumed = session(ceiling=3, state=copy.deepcopy(scan.state))
    resumed.run()
    assert resumed.validator.calls.count("app03.example.test") == 1
    assert resumed.nodes["app03.example.test"]["attempts"] == 2
