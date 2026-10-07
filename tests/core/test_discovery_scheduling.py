"""Offline fairness checks: admission and parent paging are not DNS evidence."""

import copy
import math

import pytest

from enumeraite.core import discovery_session
from enumeraite.core.discovery_plans import DiscoveryPlan
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint


class NoNetwork:
    def validate(self, *args, **kwargs):
        pytest.fail("Scheduling tests must never make DNS requests")


def scan_with_families(count, *, max_depth=1, max_candidates=5000):
    scan = DiscoverySession(
        "example.test",
        ScanOptions(
            passive=False,
            max_candidates=max_candidates,
            max_rounds=30,
            max_number=9999,
            max_depth=max_depth,
        ),
        NoNetwork(),
    )
    families = [f"rack{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(count)]
    names = [f"{family}{number:02d}.example.test" for family in families for number in (1, 2)] + [
        "zzpayments-dev.example.test",
        "zzpayments-prod.example.test",
    ]
    for value in names:
        scan._enqueue(value, source="seed")
        scan.nodes[value].update(status="verified", attempts=1)
    scan.state.update(initialized=True, attempted=len(names))
    return scan, families


def miss_wave(scan):
    before = set(scan.nodes)
    assert scan._expand()
    added = [node for value, node in scan.nodes.items() if value not in before]
    for node in added:
        node.update(status="miss", attempts=1)
    scan.state["attempted"] += len(added)
    validate_checkpoint(scan.state, scan.target)
    return added


def test_thirty_numeric_families_and_lexical_family_all_receive_work():
    scan, families = scan_with_families(30)
    counts = {family: 0 for family in families}
    lexical = 0
    for _ in range(10):
        added = miss_wave(scan)
        assert len(added) <= 40
        for node in added:
            if node["parent"].startswith("zzpayments-"):
                lexical += 1
            else:
                family = next(family for family in families if node["parent"].startswith(family))
                counts[family] += 1
    assert lexical > 0
    assert min(counts.values()) > 0
    # Before fair admission, one arbitrary hash-first family received 170 of
    # 400 proposals; six numeric families and the lexical family received zero.
    assert max(counts.values()) <= 40
    assert len(scan.nodes) == len(set(scan.nodes))


def test_later_parents_are_considered_after_the_first_two_hundred(monkeypatch):
    scan, _ = scan_with_families(110, max_depth=4)
    original = discovery_session.expand_host
    considered = []

    def track(host, *args, **kwargs):
        considered.append(host)
        return original(host, *args, **kwargs)

    monkeypatch.setattr(discovery_session, "expand_host", track)
    miss_wave(scan)
    assert len(considered) == 200
    assert not any(value.startswith("zzpayments-") for value in considered)
    considered.clear()
    miss_wave(scan)
    assert len(considered) == 200
    assert "zzpayments-dev.example.test" in considered
    assert "zzpayments-prod.example.test" in considered
    assert all(node["expanded_revision"] == 0 for node in scan.nodes.values())


def test_resume_reproduces_parent_cursor_admission_and_allocation():
    original, _ = scan_with_families(110, max_depth=4)
    miss_wave(original)
    saved = copy.deepcopy(original.state)
    resumed = DiscoverySession(
        original.target, original.options, NoNetwork(), state=copy.deepcopy(saved)
    )
    expected = miss_wave(original)
    actual = miss_wave(resumed)
    assert [(node["value"], node["parent"]) for node in actual] == [
        (node["value"], node["parent"]) for node in expected
    ]
    assert not set(node["value"] for node in actual).intersection(
        node["value"] for node in saved["nodes"]
    )
    assert resumed.state["round"] == 2


def test_legacy_checkpoint_needs_no_new_scheduler_fields():
    first, _ = scan_with_families(30)
    first.state["round"] = 7
    saved = copy.deepcopy(first.state)
    for field in ("pattern_paging_version", "model_batches", "predictions"):
        saved.pop(field, None)
    resumed = DiscoverySession(first.target, first.options, NoNetwork(), state=saved)
    validate_checkpoint(resumed.state, resumed.target)
    assert resumed.state["round"] == 7
    assert miss_wave(resumed)
    assert resumed.state["round"] == 8


def test_plan_admission_reserves_priority_and_exploration_slots():
    scan, _ = scan_with_families(1)
    high = [DiscoveryPlan(f"high-{index:03d}", "supported", priority=0.9) for index in range(30)]
    low = [DiscoveryPlan(f"low-{index:03d}", "tentative", priority=0.2) for index in range(40)]
    plans = [*high, *low]
    seen = set()
    for round_number in range(1, math.ceil(len(plans) / 6) + 1):
        scan.state["round"] = round_number
        chosen = scan._admit_plans(plans, 24)
        assert len(chosen) == 24
        assert sum(plan.priority == 0.9 for plan in chosen) >= 18
        seen.update(plan.id for plan in chosen)
    assert seen == {plan.id for plan in plans}


def test_equal_priority_first_allocation_rotates_without_mutating_input():
    scan, _ = scan_with_families(1)
    plans = [DiscoveryPlan(str(index), "same score", priority=0.9) for index in range(8)]
    identities = [plan.id for plan in plans]
    leaders = []
    for round_number in range(1, 9):
        scan.state["round"] = round_number
        leaders.append(scan._admit_plans(plans, 24)[0].id)
    assert len(set(leaders)) == 8
    assert [plan.id for plan in plans] == identities


@pytest.mark.parametrize("limit", [0, 1, 2, 6, 24])
def test_plan_admission_never_exceeds_available_candidate_capacity(limit):
    scan, _ = scan_with_families(1)
    plans = [DiscoveryPlan(str(index), "fixture", priority=0.9) for index in range(30)]
    for round_number in range(1, 6):
        scan.state["round"] = round_number
        assert len(scan._admit_plans(plans, limit)) == limit


def test_tight_candidate_limit_remains_global_with_rotated_admission():
    scan, _ = scan_with_families(30, max_candidates=67)
    added = miss_wave(scan)
    assert len(added) == 5
    assert len(scan.nodes) == scan.options.max_candidates
    assert all(node["value"].endswith(".example.test") for node in added)
    assert all(node["depth"] <= scan.options.max_depth for node in added)
    assert scan._stop() == "candidate limit reached"


def test_single_remaining_slot_keeps_highest_priority_instead_of_hash_exploration():
    scan, _ = scan_with_families(1)
    low = DiscoveryPlan("aaa", "speculative child", priority=0.25)
    high = DiscoveryPlan("zzz", "supported sibling", priority=0.9)
    for round_number in range(1, 6):
        scan.state["round"] = round_number
        assert scan._admit_plans([low, high], 1) == [high]
