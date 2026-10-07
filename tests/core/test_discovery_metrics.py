import copy

from enumeraite.core.discovery_metrics import discovery_method_counts
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions


def node(source, status="pending", attempts=0, reason=""):
    return {"source": source, "status": status, "attempts": attempts, "reason": reason}


def test_counts_separate_origins_numeric_rules_and_dns_outcomes():
    nodes = [
        node("seed", "verified", 1),
        node("history", "verified", 1),
        node("passive", "miss", 1),
        node("bootstrap", "wildcard", 1),
        node("pattern", "verified", 2, "numeric-slot hypothesis: neighbors"),
        node("pattern", "miss", 1, "Systematic numeric family enumeration"),
        node("pattern", "inconclusive", 2, "observed vocabulary"),
        node("model", "pending", 0, "numeric-slot hypothesis: model text is not a rule"),
        node("legacy-origin", "pending", 0),
    ]
    original = copy.deepcopy(nodes)
    result = discovery_method_counts(nodes)
    assert nodes == original
    rows = {row["method"]: row for row in result}
    assert list(rows) == ["seed", "history", "passive", "bootstrap", "numeric", "pattern", "model", "other"]
    assert sum(row["candidates"] for row in result) == len(nodes)
    assert sum(row["checked"] for row in result) == 7
    assert rows["numeric"]["checked"] == 2  # A retry is not another candidate.
    assert rows["numeric"]["verified"] == rows["numeric"]["miss"] == 1
    assert rows["pattern"]["inconclusive"] == 1
    assert rows["pattern"]["miss"] == 0
    assert rows["model"]["pending"] == 1
    assert rows["bootstrap"]["wildcard"] == 1
    assert rows["bootstrap"]["verified"] == 0
    assert rows["history"]["verified"] == rows["history"]["checked"] == 1
    assert rows["other"]["verified"] == 0


def test_requeued_transient_error_is_checked_and_pending_but_not_a_miss():
    row = discovery_method_counts([node("passive", "pending", 1)])[0]
    assert row["candidates"] == row["checked"] == row["pending"] == 1
    assert row["miss"] == row["inconclusive"] == row["verified"] == 0


def test_empty_and_legacy_sources_are_bounded_and_results_are_independent():
    assert discovery_method_counts([]) == []
    nodes = [node(f"legacy-{i}", "inconclusive", 1) for i in range(1000)]
    first = discovery_method_counts(nodes)
    assert len(first) == 1
    assert first[0]["method"] == "other"
    assert first[0]["inconclusive"] == 1000
    first[0]["inconclusive"] = 0
    assert discovery_method_counts(nodes)[0]["inconclusive"] == 1000


def test_session_reports_first_enqueue_ownership_without_changing_checkpoint():
    session = DiscoverySession("example.test", ScanOptions(passive=False), object())
    assert session._enqueue("api.example.test", source="passive")
    assert not session._enqueue("api.example.test", source="model")
    session.nodes["api.example.test"].update(status="verified", attempts=1)
    before = copy.deepcopy(session.state)
    summary = session.summary()
    assert summary["discovery_methods"] == session.report()["discovery_methods"]
    assert summary["discovery_methods"][0]["method"] == "passive"
    assert summary["discovery_methods"][0]["verified"] == 1
    assert len(summary["discovery_methods"]) == 1
    assert session.state == before
    assert "discovery_methods" not in session.state
