import copy
import importlib
import json

import pytest
from click.testing import CliRunner

from enumeraite.cli.main import cli
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from enumeraite.core.dns_validator import DNSResult


class MixedDNS:
    def validate(self, names, *, deadline):
        results = []
        for name in names:
            label = name.split(".", 1)[0]
            if label == "missing":
                continue
            results.append(
                DNSResult(
                    name,
                    exists=label == "verified",
                    ip_addresses=["192.0.2.1"] if label == "verified" else [],
                    response_time=0,
                    error="DNS connection timed out" if label == "timeout" else None,
                    wildcard=label == "wild",
                )
            )
        return results


def test_summary_counts_distinguish_misses_wildcards_and_unresolved_dns():
    session = DiscoverySession(
        "example.com", ScanOptions(passive=False, max_candidates=5), MixedDNS()
    )
    result = session.run(
        [f"{label}.example.com" for label in ("verified", "miss", "wild", "timeout", "missing")]
    )
    assert result["attempted"] == 5
    assert result["verified"] == 1
    assert result["wildcard"] == 1
    assert result["inconclusive"] == 2
    assert result["queued"] == 0
    assert result["findings"] == ["verified.example.com"]
    assert session.nodes["miss.example.com"]["status"] == "miss"
    assert session.report()["inconclusive"] == 2


def test_summary_pending_names_are_not_counted_as_dns_failures():
    session = DiscoverySession("example.com", ScanOptions(passive=False), MixedDNS())
    session._enqueue("pending.example.com", source="seed")
    report = session.summary()
    assert report["queued"] == 1
    assert report["inconclusive"] == report["wildcard"] == report["attempted"] == 0


def test_summary_inconclusive_count_clears_after_successful_resume_retry():
    options = ScanOptions(passive=False, max_candidates=1)
    initial = DiscoverySession("example.com", options, MixedDNS())
    assert initial.run(["timeout.example.com"])["inconclusive"] == 1

    class WorkingDNS:
        def validate(self, names, *, deadline):
            return [DNSResult(name, True, ["192.0.2.1"], 0) for name in names]

    resumed = DiscoverySession(
        "example.com", options, WorkingDNS(), state=copy.deepcopy(initial.state)
    )
    report = resumed.run()
    assert report["inconclusive"] == 0
    assert report["verified"] == 1
    assert report["attempted"] == 1


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_reports_inconclusive_dns_without_presenting_it_as_absence(
    monkeypatch, tmp_path, quiet
):
    scan_module = importlib.import_module("enumeraite.cli.scan_command")

    class FailedDNS:
        def __init__(self, *args, **kwargs):
            pass

        def validate(self, names, *, deadline):
            return [DNSResult(name, False, [], 0, error="DNS unavailable") for name in names]

    monkeypatch.setattr(scan_module, "ScanDNS", FailedDNS)
    result = CliRunner().invoke(
        cli,
        [
            "scan",
            "example.com",
            "--no-passive",
            "--max-candidates",
            "2",
            "--state",
            str(tmp_path / "scan.json"),
            "--json",
            *(["--quiet"] if quiet else []),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["verified"] == 0
    assert report["inconclusive"] == 2
    assert report["wildcard"] == 0
    if quiet:
        assert result.stderr == ""
    else:
        assert "2 inconclusive, 0 wildcard" in result.stderr
        assert "results are incomplete, not confirmed absences" in result.stderr
        assert "--resume" in result.stderr
