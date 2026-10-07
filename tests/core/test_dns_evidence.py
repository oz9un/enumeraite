import copy

import pytest

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_evidence import new_dns_evidence, validate_dns_evidence
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.scan_dns import ScanDNSResult
from enumeraite.core.scan_state import ScanStore

NAME = "api.example.test"
TARGET = "example.test"


def evidence():
    value = new_dns_evidence(NAME)
    value["records"]["A"] = {"status": "answered", "addresses": ["192.0.2.1"], "error": ""}
    value["records"]["AAAA"]["status"] = "no_answer"
    value["wildcard"]["status"] = "not_detected"
    return value


class FixedDNS:
    def __init__(self, result):
        self.result = result

    def validate(self, values, *, deadline):
        assert values == [NAME]
        return [self.result]


def session_with(result, **kwargs):
    return DiscoverySession(
        TARGET,
        ScanOptions(passive=False, max_candidates=1),
        FixedDNS(result),
        **kwargs,
    )


@pytest.mark.parametrize("status", ["verified", "miss", "wildcard", "inconclusive"])
@pytest.mark.parametrize("aliases", [None, [], ["edge.example.test"]])
def test_dns_evidence_survives_private_checkpoint_and_resume(tmp_path, status, aliases):
    observed = evidence()
    if aliases is not None:
        observed["aliases"] = aliases
    if status == "miss":
        observed["records"]["A"] = {"status": "nxdomain", "addresses": [], "error": ""}
        observed["records"]["AAAA"]["status"] = "not_checked"
        observed["wildcard"]["status"] = "not_checked"
    elif status == "wildcard":
        observed["wildcard"]["status"] = "matched"
    elif status == "inconclusive":
        observed["wildcard"]["status"] = "inconclusive"
    result = ScanDNSResult(
        NAME,
        status == "verified",
        ["192.0.2.1"] if status != "miss" else [],
        0.01,
        error="wildcard calibration inconclusive" if status == "inconclusive" else None,
        wildcard=status == "wildcard",
        dns_evidence=observed,
    )
    store = ScanStore(tmp_path / "scan.json")
    session = session_with(result, checkpoint=store.save)
    session.run([NAME])
    loaded = store.load()
    validate_checkpoint(loaded, TARGET)
    assert loaded["nodes"][0]["status"] == status
    assert loaded["nodes"][0]["dns_evidence"] == observed
    resumed = session_with(result, state=loaded)
    assert resumed.nodes[NAME]["dns_evidence"] == observed


def test_generic_results_and_legacy_checkpoints_do_not_gain_invented_evidence():
    result = DNSResult(NAME, True, ["192.0.2.1"], 0.01)
    session = session_with(result)
    session.run([NAME])
    assert "dns_evidence" not in session.nodes[NAME]
    old_state = copy.deepcopy(session.state)
    validate_checkpoint(old_state, TARGET)
    resumed = session_with(result, state=old_state)
    assert "dns_evidence" not in resumed.nodes[NAME]
    assert ScanDNSResult(NAME, False, [], 0).dns_evidence is None


def test_nested_evidence_is_detached_from_validator_and_event_consumers():
    observed = evidence()
    events = []

    def event(kind, payload):
        if "dns_evidence" in payload:
            events.append(copy.deepcopy(payload))
            payload["dns_evidence"]["records"]["A"]["addresses"].append("192.0.2.99")
            payload["dns_evidence"]["wildcard"]["status"] = "matched"

    result = ScanDNSResult(NAME, True, ["192.0.2.1"], 0.01, dns_evidence=observed)
    session = session_with(result, event=event)
    session.run([NAME])
    assert session.nodes[NAME]["dns_evidence"] == observed
    assert len(events) == 2
    observed["records"]["A"]["addresses"].clear()
    assert session.nodes[NAME]["dns_evidence"]["records"]["A"]["addresses"] == ["192.0.2.1"]
    assert all(row["dns_evidence"]["records"]["A"]["addresses"] == ["192.0.2.1"] for row in events)
    validate_checkpoint(session.state, TARGET)


def test_partial_cancelled_observation_does_not_consume_attempt_or_verify(tmp_path):
    observed = evidence()
    observed["records"]["AAAA"]["status"] = "not_checked"
    observed["wildcard"]["status"] = "not_checked"
    result = ScanDNSResult(NAME, True, ["192.0.2.1"], 0.01, cancelled=True, dns_evidence=observed)
    store = ScanStore(tmp_path / "scan.json")
    session = session_with(result, checkpoint=store.save)
    session._enqueue(NAME, source="seed")
    assert session._validate_batch([NAME]) == 0
    node = session.nodes[NAME]
    assert node["status"] == "pending" and node["attempts"] == 0
    assert node["addresses"] == [] and session.state["attempted"] == 0
    assert node["dns_evidence"] == observed
    loaded = store.load()
    validate_checkpoint(loaded, TARGET)
    assert loaded["nodes"][0]["dns_evidence"] == observed


def test_retry_without_record_details_does_not_reuse_previous_evidence():
    observed = evidence()
    result = ScanDNSResult(NAME, False, [], 0.01, error="timeout", dns_evidence=observed)
    session = session_with(result)
    session._enqueue(NAME, source="seed")
    session._validate_batch([NAME])
    session.nodes[NAME]["status"] = "pending"
    session.validator = FixedDNS(DNSResult(NAME, True, ["192.0.2.2"], 0.02))
    session._validate_batch([NAME])
    assert session.nodes[NAME]["status"] == "verified"
    assert "dns_evidence" not in session.nodes[NAME]
    validate_checkpoint(session.state, TARGET)


@pytest.mark.parametrize(
    "path,value",
    [
        ((), None),
        (("extra",), "unexpected"),
        (("checked_at",), "2026-10-03"),
        (("checked_at",), "2026-10-03T12:00:00"),
        (("checked_at",), "2026-10-03T12:00:00-07:00"),
        (("checked_at",), "2026-02-30T12:00:00Z"),
        (("checked_at",), "2026-10-03T12:00:00Z\n"),
        (("duration_ms",), True),
        (("duration_ms",), "1"),
        (("duration_ms",), -1),
        (("duration_ms",), float("nan")),
        (("duration_ms",), float("inf")),
        (("duration_ms",), 10**1000),
        (("records",), []),
        (("records", "MX"), {}),
        (("records", "A"), []),
        (("records", "A", "status"), []),
        (("records", "A", "status"), "invented"),
        (("records", "A", "status"), "nxdomain"),
        (("records", "A", "error"), "unexpected error on an answered record"),
        (("records", "A", "error"), "x" * 181),
        (("records", "A", "addresses"), []),
        (("records", "A", "addresses"), ["192.0.2.1"] * 101),
        (("records", "A", "addresses"), [True]),
        (("records", "A", "addresses"), ["not an address"]),
        (("records", "A", "addresses"), ["2001:db8::1"]),
        (("records", "AAAA"), {"status": "answered", "addresses": ["192.0.2.1"], "error": ""}),
        (("records", "AAAA"), {"status": "answered", "addresses": ["fe80::1%en0"], "error": ""}),
        (("wildcard",), []),
        (("wildcard", "status"), {}),
        (("wildcard", "status"), "invented"),
        (("wildcard", "parent"), "outside.test"),
        (("wildcard", "parent"), "deeper.example.test"),
        (("wildcard", "parent"), "EXAMPLE.TEST"),
        (("aliases",), None),
        (("aliases",), "edge.example.test"),
        (("aliases",), ("edge.example.test",)),
        (("aliases",), [True]),
        (("aliases",), [[]]),
        (("aliases",), ["edge.example.test"] * 2),
        (("aliases",), [f"edge{i}.example.test" for i in range(17)]),
        (("aliases",), ["EDGE.example.test"]),
        (("aliases",), ["edge.example.test."]),
        (("aliases",), [" edge.example.test"]),
        (("aliases",), ["edge.example.test\n"]),
        (("aliases",), ["bad_label.example.test"]),
        (("aliases",), ["edge.outside.test"]),
        (("aliases",), ["edge.notexample.test"]),
        (("aliases",), [TARGET]),
        (("aliases",), [NAME]),
    ],
)
def test_checkpoint_rejects_malformed_or_out_of_scope_dns_evidence(path, value):
    observed = evidence()
    if not path:
        observed = value
    else:
        entry = observed
        for key in path[:-1]:
            entry = entry[key]
        entry[path[-1]] = value
    session = session_with(DNSResult(NAME, True, ["192.0.2.1"], 0))
    session.run([NAME])
    session.nodes[NAME]["dns_evidence"] = observed
    with pytest.raises(ValueError, match="DNS|wildcard"):
        validate_checkpoint(session.state, TARGET)


def test_utc_z_timestamp_and_ipv6_records_are_valid():
    observed = evidence()
    observed["checked_at"] = "2026-10-03T12:34:56.123Z"
    observed["records"]["AAAA"] = {
        "status": "answered",
        "addresses": ["2001:db8::1"],
        "error": "",
    }
    validate_dns_evidence(observed, NAME, TARGET)


def test_optional_dns_alias_limit_accepts_sixteen_and_rejects_seventeen():
    observed = evidence()
    observed["aliases"] = [f"edge{i}.example.test" for i in range(16)]
    validate_dns_evidence(observed, NAME, TARGET)
    observed["aliases"].append("overflow.example.test")
    with pytest.raises(ValueError, match="DNS aliases"):
        validate_dns_evidence(observed, NAME, TARGET)


def test_legacy_dns_evidence_validation_does_not_invent_aliases():
    observed = evidence()
    previous = copy.deepcopy(observed)
    validate_dns_evidence(observed, NAME, TARGET)
    assert observed == previous
    assert "aliases" not in observed
