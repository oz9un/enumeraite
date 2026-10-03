"""The small common-name fallback must not become per-host brute forcing."""

from enumeraite.core.discovery_session import BOOTSTRAP_LABELS, DiscoverySession, ScanOptions
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.prompt_spec import SUBDOMAIN_SPEC


class NoHitsDNS:
    def __init__(self):
        self.calls = []

    def validate(self, names, *, deadline):
        self.calls.extend(names)
        return [DNSResult(name, False, [], 0) for name in names]


def test_common_name_fallback_is_small_unique_valid_and_target_independent():
    assert len(BOOTSTRAP_LABELS) == len(set(BOOTSTRAP_LABELS)) <= 100
    assert BOOTSTRAP_LABELS[:4] == ("www", "mail", "api", "dev")
    assert BOOTSTRAP_LABELS[19] == "login"
    for label in BOOTSTRAP_LABELS:
        assert "." not in label
        assert SUBDOMAIN_SPEC.validator(f"{label}.example.test")


def test_no_evidence_root_scan_tries_only_scoped_fallback_once():
    dns = NoHitsDNS()
    scan = DiscoverySession("example.test", ScanOptions(passive=False), dns)
    result = scan.run()
    assert dns.calls == [f"{label}.example.test" for label in BOOTSTRAP_LABELS]
    assert result["verified"] == 0
    assert result["attempted"] == len(BOOTSTRAP_LABELS)
    assert all(node["source"] == "bootstrap" for node in scan.nodes.values())


def test_small_candidate_budget_preserves_old_fallback_order():
    dns = NoHitsDNS()
    scan = DiscoverySession("example.test", ScanOptions(passive=False, max_candidates=4), dns)
    scan.run()
    assert dns.calls == [f"{label}.example.test" for label in BOOTSTRAP_LABELS[:4]]


def test_small_scan_reserves_room_for_recursive_discovery():
    dns = NoHitsDNS()
    scan = DiscoverySession("example.test", ScanOptions(passive=False, max_candidates=100), dns)
    scan.run()
    assert dns.calls == [f"{label}.example.test" for label in BOOTSTRAP_LABELS[:20]]


def test_explicit_seed_only_scan_does_not_gain_generic_child_guesses():
    dns = NoHitsDNS()
    scan = DiscoverySession("example.test", ScanOptions(passive=False), dns)
    scan.run(["209-64-116-001.example.test"])
    assert dns.calls == ["209-64-116-001.example.test"]
