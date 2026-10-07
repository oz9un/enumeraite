"""Exported findings distinguish DNS evidence from website availability."""

import pytest

from enumeraite.cli.scan_tree import render_tree, text_tree


@pytest.fixture
def report():
    return {
        "target": "example.com",
        "verified": 1,
        "known_verified": 0,
        "attempted": 1,
        "round": 1,
        "stop_reason": "verified target reached",
        "nodes": [
            {
                "value": "api.example.com",
                "status": "verified",
                "source": "model",
                "parent": None,
                "round": 1,
                "reason": "Related service hypothesis",
                "pattern": "",
                "attempts": 1,
                "addresses": ["192.0.2.10"],
            }
        ],
    }


@pytest.fixture
def evidence():
    return {
        "checked_at": "2026-10-03T12:00:00+00:00",
        "duration_ms": 18.5,
        "records": {
            "A": {"status": "answered", "addresses": ["192.0.2.10"], "error": ""},
            "AAAA": {"status": "error", "addresses": [], "error": "DNS lookup timed out"},
        },
        "wildcard": {"status": "not_detected", "parent": "example.com"},
    }


def test_legacy_export_does_not_invent_web_or_dns_checks(report):
    content = render_tree(report)
    assert "DNS resolved" in content
    assert "Not checked — no HTTP/HTTPS or TLS checks" in content
    assert "192.0.2.10" in content
    assert "Individual lookup details not recorded" in content
    assert "<dt>Last DNS check (UTC)</dt><dd>Not recorded</dd>" in content
    assert "<dt>Wildcard check</dt><dd>Not recorded</dd>" in content
    assert "not continuous availability monitoring" in content
    assert 'src="http' not in content


def test_export_preserves_partial_dns_result_and_time(report, evidence):
    report["nodes"][0]["dns_evidence"] = evidence
    content = render_tree(report)
    assert "IPv4 (A)" in content
    assert "IPv6 (AAAA)" in content
    assert "2026-10-03T12:00:00+00:00" in content
    assert "18.5 ms" in content
    assert "DNS lookup timed out" in content
    assert "Not detected in sampled sibling names" in content
    assert "calibration may be reused" in content
    assert "Not checked — no HTTP/HTTPS or TLS checks" in content


@pytest.mark.parametrize(
    ("status", "label"),
    [
        ("matched", "Candidate filtered as a wildcard response"),
        ("different_answer", "Wildcard addresses found; candidate returned a different address"),
        ("inconclusive", "Wildcard calibration was inconclusive"),
        ("not_checked", "Not checked for this candidate"),
    ],
)
def test_export_explains_wildcard_outcome(report, evidence, status, label):
    evidence["wildcard"]["status"] = status
    report["nodes"][0]["dns_evidence"] = evidence
    assert label in render_tree(report)


def test_export_escapes_dns_evidence_and_proposal_text(report, evidence):
    attack = "<img src=x onerror=alert(1)>"
    evidence["records"]["AAAA"]["error"] = attack
    report["nodes"][0].update(dns_evidence=evidence, reason=attack)
    content = render_tree(report)
    assert attack not in content
    assert "&lt;img src=x onerror=alert(1)&gt;" in content


def test_terminal_tree_states_dns_only_semantics(report):
    assert "1 DNS resolved (web not checked)" in text_tree(report)


def test_legacy_export_exposes_all_collectors_without_inventing_source_records(report):
    node = report["nodes"][0]
    node["source"] = "passive"
    report["passive_sources"] = {
        "crt.sh": [node["value"]],
        "merklemap": [node["value"]],
        "subfinder": ["other.example.com"],
    }
    content = render_tree(report)
    assert "Observed by crt.sh" in content
    assert "Observed by merklemap" in content
    assert "Observed by subfinder" not in content
    assert "Certificate ID" not in content
    assert 'href="https://crt.sh/?q=api.example.com"' in content
    assert "crt.sh, merklemap" in text_tree(report)


def test_origin_export_separates_historical_record_and_current_dns_and_escapes_metadata(report):
    node = report["nodes"][0]
    node["source"] = "seed"
    node["origin_evidence"] = {
        "queued_at": "2026-10-03T12:00:01+00:00",
        "passive": [
            {
                "source": "crt.sh",
                "query": "%.example.com",
                "collected_at": "2026-10-03T12:00:00+00:00",
                "matched_name": "*.API.example.com",
                "certificate_id": "12345",
                "issuer": "<img src=x onerror=alert(1)>",
                "entry_timestamp": "2025-10-01T10:00:00",
            },
            {
                "source": "merklemap",
                "query": "*.example.com",
                "collected_at": "2026-10-03T12:00:00+00:00",
                "matched_name": "api.example.com",
                "subject_common_name": "*.example.com",
                "first_seen": "2025-10-01T10:00:00+00:00",
            },
            {"source": "javascript:alert(1)", "certificate_id": '123"><script>alert(1)</script>'},
        ],
    }
    content = render_tree(report)
    assert "<dt>First queued by</dt><dd>seed</dd>" in content
    assert "Observed by crt.sh" in content and "Observed by merklemap" in content
    assert "*.example.com" in content and "192.0.2.10" in content
    assert "2025-10-01T10:00:00+00:00" in content and "2026-10-03T12:00:00+00:00" in content
    assert 'href="https://crt.sh/?id=12345"' in content
    assert 'href="javascript:' not in content
    assert "<img " not in content and "<script>alert(1)" not in content
    assert "&lt;img src=x onerror=alert(1)&gt;" in content
