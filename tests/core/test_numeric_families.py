"""Numeric width compatibility and evidence-backed IPv4 range hypotheses."""

import pytest

from enumeraite.core.discovery_plans import numeric_plans, plan_id

TARGET = "example.test"


def node(label, *, status="verified", addresses=(), attempts=None):
    value = f"{label}.{TARGET}"
    return {
        "value": value,
        "status": status,
        "depth": 0,
        "round": 0,
        "attempts": (0 if status == "pending" else 1) if attempts is None else attempts,
        "addresses": list(addresses),
    }


def build_plans(rows, *, max_number=9999, count=5000):
    return numeric_plans(
        {row["value"]: row for row in rows},
        TARGET,
        max_number=max_number,
        max_depth=4,
        count=count,
    )


@pytest.mark.parametrize(
    ("left", "right", "width", "next_label"),
    [
        ("worker9", "worker10", 1, "worker11"),
        ("worker09", "worker10", 2, "worker11"),
        ("worker009", "worker010", 3, "worker011"),
        ("worker99", "worker100", 2, "worker101"),
        ("worker099", "worker100", 3, "worker101"),
    ],
)
def test_two_verified_names_bridge_compatible_minimum_widths(left, right, width, next_label):
    plans, membership = build_plans([node(left), node(right)], count=3)
    assert len(plans) == 1
    expected_pattern = f"worker<number:0{width}d>.{TARGET}"
    assert plans[0].id == plan_id(expected_pattern)
    assert plans[0].proposals[0].pattern == expected_pattern
    assert plans[0].proposals[0].value == f"{next_label}.{TARGET}"
    assert plans[0].feedback["verified"] == 2
    assert len(membership) == 2


def test_distinct_padding_spellings_cannot_establish_a_numeric_axis_by_themselves():
    plans, membership = build_plans([node("worker1"), node("worker01"), node("worker001")])
    assert plans == []
    assert membership == {}


def test_independently_supported_padding_families_keep_separate_spellings_and_ids():
    rows = [node(f"worker{number:0{width}d}") for width in (1, 2, 3) for number in (1, 2)]
    plans, _ = build_plans(rows, count=1)
    assert {plan.id for plan in plans} == {
        plan_id(f"worker<number:0{width}d>.{TARGET}") for width in (1, 2, 3)
    }
    assert {plan.proposals[0].value for plan in plans} == {
        f"worker3.{TARGET}",
        f"worker03.{TARGET}",
        f"worker003.{TARGET}",
    }


def test_shorter_evidence_supersedes_redundant_higher_width_without_inventing_padding():
    original, _ = build_plans([node("worker10"), node("worker11")], count=1)
    assert len(original) == 1
    assert original[0].id == plan_id(f"worker<number:02d>.{TARGET}")
    expanded, _ = build_plans(
        [node("worker9"), node("worker10"), node("worker11")], max_number=11, count=20
    )
    assert len(expanded) == 1
    assert expanded[0].id == plan_id(f"worker<number:01d>.{TARGET}")
    assert {item.value for item in expanded[0].proposals} == {
        f"worker{number}.{TARGET}" for number in range(1, 9)
    }


def test_actual_padding_is_not_suppressed_by_a_supported_shorter_width():
    plans, _ = build_plans(
        [node("worker9"), node("worker09"), node("worker10"), node("worker11")],
        max_number=11,
        count=20,
    )
    assert {plan.id for plan in plans} == {
        plan_id(f"worker<number:01d>.{TARGET}"),
        plan_id(f"worker<number:02d>.{TARGET}"),
    }
    assert any(item.value == f"worker01.{TARGET}" for plan in plans for item in plan.proposals)


def test_dominated_width_suppression_is_independent_of_discovery_order():
    names = ["worker9", "worker10", "worker11", "worker100", "worker101"]
    forward, _ = build_plans([node(name) for name in names], count=3)
    backward, _ = build_plans([node(name) for name in reversed(names)], count=3)
    assert len(forward) == len(backward) == 1
    assert forward[0].id == backward[0].id == plan_id(f"worker<number:01d>.{TARGET}")
    assert [item.value for item in forward[0].proposals] == [
        item.value for item in backward[0].proposals
    ]


def test_existing_short_width_id_stays_stable_after_crossing_boundary():
    original, _ = build_plans([node("worker8"), node("worker9")], count=1)
    resumed, _ = build_plans([node("worker8"), node("worker9"), node("worker10")], count=1)
    assert len(resumed) == 1
    assert original[0].id == resumed[0].id
    assert resumed[0].feedback["verified"] == 3


@pytest.mark.parametrize("status", ["pending", "miss", "inconclusive", "wildcard"])
def test_nonpositive_names_cannot_establish_support_or_introduce_shorter_width(status):
    plans, _ = build_plans([node("worker9", status=status), node("worker10")], count=1)
    assert plans == []
    plans, _ = build_plans(
        [node("worker9", status=status), node("worker10"), node("worker11")], count=1
    )
    assert len(plans) == 1
    assert plans[0].id == plan_id(f"worker<number:02d>.{TARGET}")


@pytest.mark.parametrize("status", ["pending", "miss", "inconclusive", "wildcard"])
def test_cross_width_history_is_not_retried_and_contributes_feedback(status):
    plans, _ = build_plans(
        [node("worker8"), node("worker9"), node("worker10", status=status)], count=1
    )
    assert len(plans) == 1
    assert plans[0].proposals[0].value == f"worker11.{TARGET}"
    assert plans[0].feedback["tested"] == (2 if status == "pending" else 3)
    assert plans[0].feedback["miss"] == int(status == "miss")
    assert plans[0].feedback["inconclusive"] == int(status == "inconclusive")


def test_width_bridge_retains_other_numeric_literal_anchors():
    plans, _ = build_plans([node("rack07-worker9"), node("rack07-worker10")], count=4)
    assert len(plans) == 1
    assert all(item.value.startswith("rack07-worker") for item in plans[0].proposals)
    assert plans[0].proposals[0].pattern == f"rack07-worker<number:01d>.{TARGET}"


def encoded(number, *, status="verified", addresses=None):
    return node(
        f"209-64-116-{number:03d}",
        status=status,
        addresses=[f"209.64.116.{number}"] if addresses is None else addresses,
    )


def test_two_exact_address_matches_infer_octet_ceiling_and_include_zero():
    plans, _ = build_plans([encoded(1), encoded(2)])
    assert len(plans) == 1
    plan = plans[0]
    expected = {f"209-64-116-{number:03d}.{TARGET}" for number in range(256)}
    expected -= {f"209-64-116-001.{TARGET}", f"209-64-116-002.{TARGET}"}
    assert {item.value for item in plan.proposals} == expected
    assert plan.feedback["range_min"] == 0
    assert plan.feedback["range_max"] == 255
    assert plan.feedback["user_range_max"] == 9999
    assert plan.feedback["range_basis"] == "verified_ipv4_encoding"
    assert plan.feedback["range_evidence_count"] == 2
    assert "inferred IPv4" in plan.description
    assert "not a claim" in plan.description


def test_inferred_octet_range_never_expands_past_lower_user_cap():
    plans, _ = build_plans([encoded(1), encoded(2)], max_number=99)
    assert plans[0].feedback["range_max"] == 99
    assert {int(item.value.split(".")[0].split("-")[-1]) for item in plans[0].proposals} == (
        set(range(100)) - {1, 2}
    )


@pytest.mark.parametrize(
    "rows",
    [
        [encoded(1, addresses=[]), encoded(2, addresses=[])],
        [encoded(1), encoded(2, addresses=[])],
        [encoded(1), encoded(2, addresses=["192.0.2.1"])],
        [encoded(1), encoded(2), encoded(3, addresses=["192.0.2.1"])],
        [encoded(1), encoded(2), encoded(256, addresses=["192.0.2.1"])],
    ],
    ids=[
        "names-alone",
        "one-match",
        "conflicting-address",
        "contradictory-member",
        "invalid-octet",
    ],
)
def test_missing_or_contradictory_encoding_evidence_does_not_infer_octet_limit(rows):
    plans, _ = build_plans(rows, count=300)
    assert plans
    assert all(plan.feedback["range_max"] == 9999 for plan in plans)
    assert all("range_basis" not in plan.feedback for plan in plans)
    assert any(
        int(item.value.split(".")[0].split("-")[-1]) > 255
        for plan in plans
        for item in plan.proposals
    )


def test_matching_address_among_multiple_answers_supports_encoding():
    plans, _ = build_plans(
        [
            encoded(1, addresses=["192.0.2.1", "209.64.116.1", "2001:db8::1"]),
            encoded(2, addresses=["209.64.116.2", "192.0.2.1"]),
        ]
    )
    assert plans[0].feedback["range_max"] == 255


@pytest.mark.parametrize("status", ["miss", "pending", "inconclusive", "wildcard"])
def test_nonpositive_address_mappings_cannot_establish_encoding(status):
    plans, _ = build_plans(
        [encoded(1), encoded(2, addresses=[]), encoded(3, status=status)], count=300
    )
    assert plans[0].feedback["range_max"] == 9999
    assert "range_basis" not in plans[0].feedback


@pytest.mark.parametrize(
    "labels",
    [
        ("ip209-64-116-001", "ip209-64-116-002"),
        ("209-64-116-001-server", "209-64-116-002-server"),
        ("209-64-116-001-7", "209-64-116-002-7"),
        ("209.64.116.001", "209.64.116.002"),
    ],
    ids=["prefix-literal", "suffix-literal", "extra-number", "separate-labels"],
)
def test_nonexact_leftmost_labels_never_infer_ipv4_range(labels):
    plans, _ = build_plans(
        [node(labels[0], addresses=["209.64.116.1"]), node(labels[1], addresses=["209.64.116.2"])],
        count=300,
    )
    assert plans
    assert all(plan.feedback["range_max"] == 9999 for plan in plans)
    assert all("range_basis" not in plan.feedback for plan in plans)


def test_other_numeric_slots_after_encoded_leftmost_label_are_not_octet_limited():
    plans, _ = build_plans(
        [
            node("209-64-116-001.rack1", addresses=["209.64.116.1"]),
            node("209-64-116-001.rack2", addresses=["209.64.116.1"]),
        ],
        count=300,
    )
    assert len(plans) == 1
    assert plans[0].feedback["range_max"] == 9999
    assert "range_basis" not in plans[0].feedback
    assert all(item.value.startswith("209-64-116-001.rack") for item in plans[0].proposals)


def test_nonpositive_contradictory_members_do_not_override_verified_range_evidence():
    plans, _ = build_plans([encoded(1), encoded(2), encoded(256, status="miss", addresses=[])])
    assert plans[0].feedback["range_max"] == 255
    assert len(plans[0].proposals) == 254
