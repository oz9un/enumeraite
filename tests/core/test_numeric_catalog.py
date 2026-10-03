"""Admission-time numeric catalogs retain the eager planner's exact behavior."""

import copy
import random
from dataclasses import FrozenInstanceError, asdict
from itertools import chain
from types import SimpleNamespace

import pytest

from enumeraite.core import discovery_plans, discovery_session
from enumeraite.core.discovery_plans import (
    DiscoveryPlan,
    _ipv4_encoding_matches,
    _numeric_buckets,
    build_numeric_catalog,
    numeric_plans,
    plan_id,
)
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from enumeraite.core.subdomain_patterns import PatternProposal

TARGET = "example.test"


def node(label, *, status="verified", depth=0, addresses=(), round_number=0):
    return {
        "value": f"{label}.{TARGET}",
        "status": status,
        "depth": depth,
        "round": round_number,
        "attempts": 0 if status == "pending" else 1,
        "addresses": list(addresses),
    }


def nodes_for(*labels):
    return {row["value"]: row for row in map(node, labels)}


def reference_eager(nodes, target, *, max_number, max_depth, count):
    """Pre-catalog page construction; family detection itself is unchanged.

    Keep an independent eager construction path here so equivalence tests do
    not merely compare the new catalog to its own eager wrapper.
    """
    plans, membership = [], {}
    for key, items in _numeric_buckets(nodes, target):
        verified = [(number, row) for number, row in items if row["status"] == "verified"]
        numbers = {number for number, _ in verified}
        if len(numbers) < 2:
            continue
        anchors = [row for _, row in verified if row["depth"] < max_depth]
        if not anchors:
            continue
        pattern = "".join(key) + "." + target
        identity = plan_id(pattern)
        for _, row in verified:
            membership.setdefault(row["value"], identity)
        seen = {number for number, _ in items}
        anchor = min(anchors, key=lambda row: (row["depth"], row["round"], row["value"]))
        slot = next(index for index, piece in enumerate(key) if piece.startswith("<number:"))
        width = int(key[slot][len("<number:0") : -2])
        encoding_matches = _ipv4_encoding_matches(verified, slot)
        minimum = 0 if encoding_matches or 0 in numbers else 1
        maximum = min(max_number, 255) if encoding_matches else max_number
        plan = DiscoveryPlan(
            identity,
            f"Enumerate {pattern} within {minimum}–{maximum}; keep other tokens fixed",
            evidence=[row["value"] for _, row in verified[:5]],
            feedback={
                "verified": len(verified),
                "tested": sum(row["attempts"] > 0 for _, row in items),
                "miss": sum(row["status"] == "miss" for _, row in items),
                "inconclusive": sum(row["status"] == "inconclusive" for _, row in items),
                "range_min": minimum,
                "range_max": maximum,
                "observed_min": min(numbers),
                "observed_max": max(numbers),
            },
            priority=0.9,
        )
        if encoding_matches:
            plan.description += (
                f"; inferred IPv4 encoding from {encoding_matches} matching DNS answers, "
                "not a claim that untested names exist"
            )
            plan.feedback.update(
                range_basis="verified_ipv4_encoding",
                range_evidence_count=encoding_matches,
                user_range_max=max_number,
            )
        for number in chain(
            range(min(numbers), min(max(numbers), maximum) + 1),
            range(max(numbers) + 1, maximum + 1),
            range(minimum, min(min(numbers), maximum + 1)),
        ):
            if number in seen:
                continue
            pieces = list(key)
            pieces[slot] = str(number).zfill(width)
            value = "".join(pieces) + "." + target
            if value in nodes or value.count(".") - target.count(".") > max_depth:
                continue
            plan.proposals.append(
                PatternProposal(
                    value,
                    anchor["value"],
                    pattern,
                    "Systematic numeric family enumeration; fixed tokens preserved",
                    0.9,
                )
            )
            if len(plan.proposals) >= count:
                break
        if plan.proposals:
            plans.append(plan)
    return sorted(plans, key=lambda plan: (-plan.feedback["verified"], plan.id)), membership


def test_randomized_whole_plans_membership_previews_and_order_match_old_path():
    rng = random.Random(8735)
    labels = [
        f"worker{number:0{width}d}"
        for number in (0, 1, 2, 9, 10, 11, 99, 100, 101, 255, 256)
        for width in (1, 2, 3)
    ]
    labels += [
        f"rack{rack:02d}-worker{number:02d}"
        for rack in (7, 8)
        for number in (0, 1, 2, 9, 10, 99, 100)
    ]
    labels += [f"child.worker{number:02d}" for number in (1, 2, 3, 10)]
    labels += [f"209-64-116-{number:03d}" for number in (0, 1, 2, 99, 100, 255, 256)]
    checked_plans = 0
    for _ in range(350):
        rows = []
        for label in rng.sample(labels, rng.randrange(10, len(labels))):
            status = rng.choice(
                ("verified", "verified", "pending", "miss", "inconclusive", "wildcard")
            )
            addresses = ()
            if label.startswith("209-") and status == "verified":
                addresses = (".".join(str(int(value)) for value in label.split("-")),)
                if rng.randrange(5) == 0:
                    addresses = ("192.0.2.1",)
            rows.append(
                node(
                    label,
                    status=status,
                    depth=rng.randrange(3),
                    addresses=addresses,
                    round_number=rng.randrange(4),
                )
            )
        rng.shuffle(rows)
        nodes = {row["value"]: row for row in rows}
        nodes["outside.invalid"] = {**node("outside1"), "value": "outside.invalid"}
        options = dict(
            max_number=rng.choice((0, 1, 9, 10, 99, 255, 300)),
            max_depth=rng.randrange(5),
            count=rng.choice((0, 1, 2, 3, 8, 40, 200)),
        )
        expected, expected_members = reference_eager(nodes, TARGET, **options)
        catalog = build_numeric_catalog(nodes, TARGET, **options)
        actual = [blueprint.materialize() for blueprint in catalog.plans]
        eager, members = numeric_plans(nodes, TARGET, **options)
        assert [asdict(plan) for plan in actual] == [asdict(plan) for plan in expected]
        assert eager == expected
        assert dict(catalog.membership) == members == expected_members
        assert [plan.context() for plan in actual] == [plan.context() for plan in expected]
        checked_plans += len(expected)
    assert checked_plans > 100


@pytest.mark.parametrize("cap", [1, 99, 255, 9999])
@pytest.mark.parametrize("conflicting", [False, True])
def test_ipv4_inference_caps_feedback_and_preview_are_unchanged(cap, conflicting):
    rows = [
        node("209-64-116-001", addresses=["209.64.116.1"]),
        node("209-64-116-002", addresses=["209.64.116.2"]),
    ]
    if conflicting:
        rows[1]["addresses"] = ["192.0.2.1"]
    nodes = {row["value"]: row for row in rows}
    options = dict(max_number=cap, max_depth=4, count=40)
    expected, membership = reference_eager(nodes, TARGET, **options)
    catalog = build_numeric_catalog(nodes, TARGET, **options)
    assert [blueprint.materialize() for blueprint in catalog.plans] == expected
    assert dict(catalog.membership) == membership


def test_snapshot_survives_mutation_of_values_statuses_addresses_and_history():
    rows = [
        node("209-64-116-001", addresses=["209.64.116.1"]),
        node("209-64-116-002", addresses=["209.64.116.2"]),
    ]
    nodes = {row["value"]: row for row in rows}
    options = dict(max_number=9999, max_depth=4, count=40)
    expected, expected_members = reference_eager(nodes, TARGET, **options)
    catalog = build_numeric_catalog(nodes, TARGET, **options)
    for row in rows:
        row["addresses"].clear()
        row.update(status="miss", value="foreign.invalid", depth=10)
    nodes.clear()
    nodes[expected[0].proposals[0].value] = node("new-history", status="pending")
    assert [blueprint.materialize() for blueprint in catalog.plans] == expected
    assert dict(catalog.membership) == expected_members
    assert catalog.plans[0].feedback["range_max"] == 255


def test_materialized_lists_and_feedback_do_not_mutate_blueprint():
    catalog = build_numeric_catalog(
        nodes_for("app01", "app02"), TARGET, max_number=99, max_depth=4, count=40
    )
    blueprint = catalog.plans[0]
    expected = blueprint.materialize()
    modified = blueprint.materialize()
    modified.proposals.clear()
    modified.evidence.append("foreign.invalid")
    modified.feedback["verified"] = 999
    assert blueprint.materialize() == expected


def test_catalog_and_recipe_are_deeply_immutable_and_share_exclusions():
    catalog = build_numeric_catalog(
        nodes_for("app01", "app02", "worker01", "worker02"),
        TARGET,
        max_number=99,
        max_depth=4,
        count=40,
    )
    blueprint = catalog.plans[0]
    assert isinstance(catalog.plans, tuple)
    assert isinstance(blueprint.evidence, tuple)
    assert isinstance(blueprint._expansion.key, tuple)
    assert all(isinstance(span, range) for span in blueprint._expansion.spans)
    assert len({id(plan._expansion.excluded) for plan in catalog.plans}) == 1
    with pytest.raises(FrozenInstanceError):
        blueprint.priority = 0.1
    with pytest.raises(FrozenInstanceError):
        blueprint.first_proposal.value = "changed.example.test"
    with pytest.raises(TypeError):
        blueprint.feedback["verified"] = 0
    with pytest.raises(TypeError):
        catalog.membership["changed.example.test"] = "changed"
    with pytest.raises(AttributeError):
        blueprint._expansion.excluded.add("changed.example.test")
    with pytest.raises(AttributeError):
        blueprint._expansion.seen.add(99)


def test_exhausted_family_retains_membership_but_cannot_displace_active_plan():
    nodes = nodes_for("exhausted1", "exhausted2", "exhausted3", "active1", "active2")
    catalog = build_numeric_catalog(nodes, TARGET, max_number=3, max_depth=4, count=40)
    assert len(catalog.plans) == 1
    assert "exhausted1.example.test" in catalog.membership
    selected = DiscoverySession._admit_plans(
        SimpleNamespace(state={"round": 1}), list(catalog.plans), 1
    )
    assert selected[0].materialize().proposals[0].value == "active3.example.test"


def test_first_candidate_is_not_retraversed_and_only_selected_pages_are_built(monkeypatch):
    rows = [
        node(f"family{chr(97 + index // 26)}{chr(97 + index % 26)}{number:03d}")
        for index in range(100)
        for number in (1, 2)
    ]
    original = discovery_plans.PatternProposal
    constructed = []

    def proposal(*args, **kwargs):
        result = original(*args, **kwargs)
        constructed.append(result)
        return result

    monkeypatch.setattr(discovery_plans, "PatternProposal", proposal)
    catalog = build_numeric_catalog(
        {row["value"]: row for row in rows}, TARGET, max_number=9999, max_depth=4, count=40
    )
    assert len(constructed) == len(catalog.plans) == 100
    selected = DiscoverySession._admit_plans(
        SimpleNamespace(state={"round": 1}), list(catalog.plans), 24
    )
    plans = [blueprint.materialize() for blueprint in selected]
    assert len(constructed) == 100 + 24 * 39
    assert all(len(plan.proposals) == 40 for plan in plans)


def test_session_materializes_only_admitted_plans_before_prediction_and_ui(monkeypatch):
    scan = DiscoverySession(
        TARGET, ScanOptions(max_candidates=1000, max_number=99, passive=False), object()
    )
    for index in range(30):
        for number in (1, 2):
            value = f"family{chr(97 + index // 26)}{chr(97 + index % 26)}{number:03d}.{TARGET}"
            scan._enqueue(value, source="seed")
            scan.nodes[value].update(status="verified", attempts=1)
    expected, _ = reference_eager(scan.nodes, TARGET, max_number=99, max_depth=4, count=40)
    expected = DiscoverySession._admit_plans(SimpleNamespace(state={"round": 1}), expected, 24)
    before = copy.deepcopy(scan.state)
    built = []
    original = discovery_plans.NumericPlanBlueprint.materialize

    def materialize(blueprint):
        built.append(blueprint.id)
        return original(blueprint)

    def rank(plans):
        assert all(isinstance(plan, DiscoveryPlan) for plan in plans)
        assert [asdict(plan) for plan in plans] == [asdict(plan) for plan in expected]
        assert [plan.context() for plan in plans] == [plan.context() for plan in expected]
        assert all(len(plan.proposals) == 40 for plan in plans)
        return plans

    monkeypatch.setattr(discovery_plans.NumericPlanBlueprint, "materialize", materialize)
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    monkeypatch.setattr(scan, "_rank_plans", rank)
    assert scan._expand()
    assert built == [plan.id for plan in expected]
    assert len(built) == 24
    assert len(scan.nodes) == len(before["nodes"]) + 40


def test_pause_before_materialization_does_not_build_or_enqueue_full_pages(monkeypatch):
    stopped = [False]
    scan = DiscoverySession(
        TARGET,
        ScanOptions(max_candidates=100, passive=False),
        object(),
        stop_requested=lambda: stopped[0],
    )
    for value in ("worker01.example.test", "worker02.example.test"):
        scan._enqueue(value, source="seed")
        scan.nodes[value].update(status="verified", attempts=1)
    original = scan._admit_plans

    def admit(plans, limit):
        selected = original(plans, limit)
        stopped[0] = True
        return selected

    def forbidden(*args):
        pytest.fail("Paused scan must not materialize selected candidate pages")

    monkeypatch.setattr(scan, "_admit_plans", admit)
    monkeypatch.setattr(discovery_session, "expand_host", lambda *args, **kwargs: [])
    monkeypatch.setattr(discovery_plans.NumericPlanBlueprint, "materialize", forbidden)
    assert not scan._expand()
    assert len(scan.nodes) == 2
