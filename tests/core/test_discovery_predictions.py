"""Offline replay of family enumeration and optional Jev priority decisions."""

import copy
import json

import pytest

from enumeraite.core.discovery_plans import allow_child_trial, numeric_plans
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.scan_budget import BudgetExhausted, ModelPrice, SpendLedger
from enumeraite.core.subdomain_patterns import PatternProposal, describe_patterns


def host(number):
    return f"192-00-002-{number:03d}.example.test"


class DNS:
    def __init__(self, hits):
        self.hits = set(hits)
        self.calls = []

    def validate(self, values, **kwargs):
        self.calls.extend(values)
        return [
            DNSResult(value, value in self.hits, ["192.0.2.1"] if value in self.hits else [], 0)
            for value in values
        ]


class Predictor:
    model = "jev-1.13.0"

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def score_plans(self, state, plans, **kwargs):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(plans)))
        if self.error:
            raise self.error
        return {plan["id"]: 0.9 if "Enumerate" in plan["description"] else 0.1 for plan in plans}


def session(*, predictor=None, options=None, hits=None, **kwargs):
    return DiscoverySession(
        "example.test",
        options or ScanOptions(passive=False, max_candidates=300, max_rounds=5),
        DNS(hits if hits is not None else [host(i) for i in range(100)]),
        predictor=predictor,
        prediction_ledger=SpendLedger(0.05, ModelPrice(0.042, 0)) if predictor else None,
        **kwargs,
    )


def test_numeric_family_continues_after_54_without_changing_constant_slots():
    scan = session()
    result = scan.run([host(i) for i in range(55)])
    assert host(55) in result["findings"] and host(99) in result["findings"]
    assert host(100) not in scan.nodes  # explicit numeric ceiling still wins
    numeric = [node for node in scan.nodes.values() if node["value"].split(".")[0][0].isdigit()]
    assert all(node["value"].startswith("192-00-002-") for node in numeric)
    assert len(scan.validator.calls) == len(set(scan.validator.calls))
    for label in ("api", "www", "admin", "dev", "mail"):
        assert sum(value.startswith(label + ".192-") for value in scan.validator.calls) <= 2
    assert len(scan.validator.calls) <= 120
    patterns = describe_patterns(result["findings"], scan.target)
    assert any(item["pattern"] == "192-00-002-<number:03d>.example.test" for item in patterns)


def test_jev_receives_bounded_plans_and_real_feedback_not_a_call_per_host():
    predictor = Predictor()
    scan = session(predictor=predictor)
    result = scan.run([host(i) for i in range(55)])
    assert host(99) in result["findings"]
    assert 1 < len(predictor.calls) <= 5
    assert all(len(plans) <= 24 for _, plans in predictor.calls)
    assert any(item["status"] == "miss" for item in predictor.calls[1][0]["recent_results"])
    assert result["ledger"] is None and result["prediction_ledger"] is not None
    assert result["predictions"] and all(
        item["status"] == "scored" for item in result["predictions"]
    )
    validate_checkpoint(scan.state, scan.target)


@pytest.mark.parametrize(
    "error", [RuntimeError("secret-key-never-print"), BudgetExhausted("empty")]
)
def test_prediction_failure_falls_back_once_without_blocking_dns_or_leaking(error):
    predictor = Predictor(error)
    scan = session(predictor=predictor)
    result = scan.run([host(i) for i in range(55)])
    assert host(99) in result["findings"]
    assert len(predictor.calls) == 1
    assert "secret-key-never-print" not in str(result)
    assert all(
        item["score"] is None and item["status"] == "fallback" for item in result["predictions"]
    )


def test_resume_numeric_history_skips_misses_and_continues_beyond_previous_wave():
    first = session(options=ScanOptions(passive=False, max_candidates=75, max_rounds=5))
    first.run([host(i) for i in range(55)])
    second = session(state=copy.deepcopy(first.state))
    result = second.run()
    assert host(99) in result["findings"]
    assert not set(first.validator.calls).intersection(second.validator.calls)
    validate_checkpoint(second.state, second.target)


def test_rank_cannot_mark_a_name_verified_or_expand_scope():
    predictor = Predictor()
    scan = session(predictor=predictor, hits=[host(0), host(1)])
    result = scan.run([host(0), host(1)])
    assert result["findings"] == [host(0), host(1)]
    assert all(value.endswith(".example.test") for value in scan.validator.calls)
    assert result["predictions"]


def test_prediction_scores_change_which_family_is_tested_first():
    class PreferZeta(Predictor):
        def score_plans(self, state, plans, **kwargs):
            return {plan["id"]: 1.0 if "zeta" in plan["description"] else 0.0 for plan in plans}

    names = [f"{prefix}{i:02d}.example.test" for prefix in ("alpha", "zeta") for i in (1, 2)]
    scan = session(
        predictor=PreferZeta(),
        hits=names,
        options=ScanOptions(passive=False, max_candidates=10, max_depth=1, max_rounds=1),
    )
    scan.run(names)
    assert scan.validator.calls[4] == "zeta03.example.test"
    assert any(value.startswith("alpha") for value in scan.validator.calls[4:])


def test_real_adapter_session_integration_uses_mock_wire_and_separate_durable_spend(monkeypatch):
    from enumeraite.providers.jev import JEV_MODEL, JEV_PRICE, JevPredictor

    key = "fixture-jev-integration-key"
    predictor = JevPredictor(key)
    snapshots, requests = [], []
    ledger = SpendLedger(0.05, JEV_PRICE)

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, request):
            self.request = request

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, **kwargs):
            yield json.dumps(
                {
                    "model": JEV_MODEL,
                    "answers": {
                        identity: {"type": "noul", "noul": 0.8}
                        for identity in self.request["questions"]
                    },
                    "usage": {"input_tokens": 1000, "output_tokens": 0},
                }
            ).encode()

    def post(url, **kwargs):
        assert ledger.reserved_usd > 0
        assert float(snapshots[-1]["prediction_ledger"]["reserved_usd"]) > 0
        assert key not in kwargs["data"].decode()
        request = json.loads(kwargs["data"])
        requests.append(request)
        return Response(request)

    monkeypatch.setattr(predictor._session, "post", post)
    scan = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_rounds=2, max_candidates=100),
        DNS([host(i) for i in range(100)]),
        predictor=predictor,
        prediction_ledger=ledger,
        checkpoint=lambda state: snapshots.append(copy.deepcopy(state)),
    )
    try:
        result = scan.run([host(i) for i in range(55)])
    finally:
        predictor.close()
    assert requests and ledger.calls == len(requests)
    assert ledger.spent_usd == pytest.approx(len(requests) * 1000 * 0.042 / 1_000_000)
    assert ledger.reserved_usd == 0 and ledger.uncertain_calls == 0
    assert result["ledger"] is None
    assert key not in json.dumps(snapshots)
    assert all(record["status"] == "scored" for record in result["predictions"])
    for snapshot in snapshots:
        validate_checkpoint(snapshot, scan.target)


def test_no_prediction_when_candidate_slots_already_full_or_paused():
    for paused in (False, True):
        predictor = Predictor()
        scan = session(
            predictor=predictor,
            options=ScanOptions(passive=False, max_candidates=2),
            stop_requested=lambda: paused,
        )
        scan.run([host(0), host(1)])
        assert not predictor.calls


def test_two_verified_examples_required_and_filled_gap_skipped_on_later_plan():
    scan = session()
    for i in (0, 2):
        scan._enqueue(host(i), source="seed")
        scan.nodes[host(i)].update(status="verified", attempts=1)
    plans, members = numeric_plans(scan.nodes, scan.target, max_number=5, max_depth=4, count=10)
    assert len(plans) == 1 and len(members) == 2
    assert [p.value for p in plans[0].proposals] == [host(1), host(3), host(4), host(5)]
    scan._enqueue(host(1), source="pattern")
    scan.nodes[host(1)].update(status="miss", attempts=1)
    plans, _ = numeric_plans(scan.nodes, scan.target, max_number=5, max_depth=4, count=10)
    assert host(1) not in [p.value for p in plans[0].proposals]
    scan.nodes[host(2)]["status"] = "wildcard"
    assert not numeric_plans(scan.nodes, scan.target, max_number=5, max_depth=4, count=10)[0]


def test_numeric_width_boundary_does_not_fill_wave_with_previously_checked_names():
    scan = session()
    for i in range(1, 50):
        name = f"host{i}.example.test"
        scan._enqueue(name, source="seed")
        scan.nodes[name].update(status="verified" if i < 10 else "miss", attempts=1)
    plans, _ = numeric_plans(scan.nodes, scan.target, max_number=99, max_depth=4, count=40)
    assert plans[0].proposals[0].value == "host50.example.test"
    assert len(plans[0].proposals) == 40


def test_passive_child_evidence_unlocks_a_family_rule_without_provenance_parent():
    scan = session()
    parents = [host(i) for i in range(4)]
    for parent in parents:
        scan._enqueue(parent, source="seed")
        scan.nodes[parent].update(status="verified", attempts=1)
    _, membership = numeric_plans(scan.nodes, scan.target, max_number=99, max_depth=4, count=10)
    for parent in parents[:2]:
        child = f"api.{parent}"
        scan._enqueue(child, source="pattern", parent=parent, depth=1)
        scan.nodes[child].update(status="miss", attempts=1)
    proposal = PatternProposal(f"api.{parents[3]}", parents[3], "test", "test", 0.2)
    assert not allow_child_trial(proposal, membership[parents[3]], membership, scan.nodes)
    child = f"api.{parents[2]}"
    scan._enqueue(child, source="passive")
    scan.nodes[child].update(status="verified", attempts=1)
    assert allow_child_trial(proposal, membership[parents[3]], membership, scan.nodes)


@pytest.mark.parametrize("value", [float("nan"), True, -0.1, 1.1, "high"])
def test_checkpoint_rejects_invalid_prediction_scores(value):
    scan = session(predictor=Predictor())
    scan.run([host(0), host(1)])
    scan.state["predictions"][0]["score"] = value
    with pytest.raises(ValueError, match="prediction"):
        validate_checkpoint(scan.state, scan.target)
