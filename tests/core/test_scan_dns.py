import threading
import time
from collections import Counter
from types import SimpleNamespace

import dns.exception
import dns.resolver
import pytest

from enumeraite.core import scan_dns
from enumeraite.core.scan_dns import ScanDNS


def install_resolver(monkeypatch, resolve):
    class Resolver:
        def resolve(self, name, kind, *, lifetime, search):
            assert 0 < lifetime <= 30
            assert search is False
            return resolve(name, kind, lifetime)

    monkeypatch.setattr(scan_dns.dns.resolver, "Resolver", Resolver)


def test_dns_workers_are_concurrent_but_respect_worker_limit(monkeypatch):
    guard = threading.Lock()
    active = 0
    peak = 0

    def resolve(name, kind, lifetime):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.015)
            raise dns.resolver.NXDOMAIN()
        finally:
            with guard:
                active -= 1

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=3, rate=1000)
    names = [f"host{i}.example.com" for i in range(9)]
    results = validator.validate(names, deadline=time.monotonic() + 2)
    assert 1 < peak <= 3
    assert [result.subdomain for result in results] == names
    assert all(not result.exists and not result.error for result in results)
    assert validator.query_count == 9


def test_all_dns_queries_and_wildcard_probes_share_rate_limiter(monkeypatch):
    now = [100.0]
    calls = []
    monkeypatch.setattr(
        scan_dns,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0], sleep=lambda duration: now.__setitem__(0, now[0] + duration)
        ),
    )

    def resolve(name, kind, lifetime):
        calls.append((name, kind, now[0]))
        if name.startswith("enumeraite-"):
            raise dns.resolver.NXDOMAIN()
        return ["192.0.2.10"] if kind == "A" else ["2001:db8::10"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, rate=20)

    def elapsed_wait(duration):
        now[0] += duration
        return False

    monkeypatch.setattr(validator._cancel, "wait", elapsed_wait)
    results = validator.validate(["api.example.com"], deadline=101)
    assert results[0].exists
    assert len(calls) == validator.query_count == 4
    assert [call[2] for call in calls] == pytest.approx([100, 100.05, 100.10, 100.15])
    assert len([call for call in calls if call[0].startswith("enumeraite-")]) == 2


def test_wildcard_calibration_is_per_parent_shared_and_cached(monkeypatch):
    probes = []

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            probes.append(name)
            raise dns.resolver.NXDOMAIN()
        return ["192.0.2.1"] if kind == "A" else []

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=3, rate=1000)
    results = validator.validate(
        ["a.dev.example.com", "b.dev.example.com", "c.prod.example.com"],
        deadline=time.monotonic() + 2,
    )
    assert all(result.exists for result in results)
    assert Counter(name.split(".", 1)[1] for name in probes) == {
        "dev.example.com": 2,
        "prod.example.com": 2,
    }
    assert len(set(probes)) == 4
    result = validator.validate(["d.dev.example.com"], deadline=time.monotonic() + 2)[0]
    assert result.exists
    assert len(probes) == 4
    assert validator.query_count == 12


def test_wildcard_answers_are_not_verified(monkeypatch):
    install_resolver(monkeypatch, lambda name, kind, lifetime: ["192.0.2.1"])
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["anything.example.com"], deadline=time.monotonic() + 2)[0]
    assert not result.exists
    assert result.wildcard
    assert not result.error


def test_distinct_explicit_host_survives_stable_wildcard(monkeypatch):
    def resolve(name, kind, lifetime):
        if kind == "AAAA":
            raise dns.resolver.NoAnswer()
        return ["192.0.2.99"] if name == "api.example.com" else ["192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]
    assert result.exists
    assert not result.wildcard
    assert result.ip_addresses == ["192.0.2.99"]


@pytest.mark.parametrize(
    ("candidate", "first_probe", "second_probe"),
    [
        (
            ["192.0.2.3", "2001:db8::1"],
            ["192.0.2.1", "2001:db8::1"],
            ["192.0.2.2", "2001:db8::1"],
        ),
        (
            ["192.0.2.1", "2001:db8::3"],
            ["192.0.2.1", "2001:db8::1"],
            ["192.0.2.1", "2001:db8::2"],
        ),
        (
            ["192.0.2.1", "192.0.2.4"],
            ["192.0.2.1", "192.0.2.2"],
            ["192.0.2.1", "192.0.2.3"],
        ),
        (
            ["192.0.2.4"],
            ["192.0.2.1", "192.0.2.2"],
            ["192.0.2.1"],
        ),
    ],
    ids=["rotating-a-stable-aaaa", "stable-a-rotating-aaaa", "overlapping-a", "subset-pool"],
)
def test_overlapping_rotating_wildcard_does_not_verify_new_pool_address(
    monkeypatch, candidate, first_probe, second_probe
):
    probes = {}

    def resolve(name, kind, lifetime):
        if name == "candidate.example.com":
            addresses = candidate
        else:
            index = probes.setdefault(name, len(probes))
            addresses = (first_probe, second_probe)[index]
        selected = [address for address in addresses if (":" in address) == (kind == "AAAA")]
        if not selected:
            raise dns.resolver.NoAnswer()
        return selected

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["candidate.example.com"], deadline=time.monotonic() + 2)[0]

    assert not result.exists
    assert result.wildcard
    assert result.error is None
    assert set(result.ip_addresses) == set(candidate)
    assert len(probes) == 2
    assert validator.query_count == 6
    assert validator._wildcards["example.com"][1] is True


def test_distinct_explicit_host_survives_identical_dual_stack_wildcard(monkeypatch):
    def resolve(name, kind, lifetime):
        if kind == "AAAA":
            return ["2001:db8::1"]
        return ["192.0.2.99"] if name == "api.example.com" else ["192.0.2.1", "192.0.2.2"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert result.exists
    assert not result.wildcard
    assert result.error is None
    assert result.ip_addresses == ["192.0.2.99", "2001:db8::1"]
    assert validator._wildcards["example.com"][1] is False


@pytest.mark.parametrize("empty_kind", ["nxdomain", "nodata"])
@pytest.mark.parametrize("positive_first", [False, True])
@pytest.mark.parametrize("candidate_distinct", [False, True])
def test_mixed_wildcard_probe_address_presence_is_inconclusive_and_retryable(
    monkeypatch, empty_kind, positive_first, candidate_distinct
):
    probes = {}
    consistently_negative = [False]
    events = []

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            index = probes.setdefault(name, len(probes))
            if consistently_negative[0]:
                raise dns.resolver.NXDOMAIN()
            positive_index = 0 if positive_first else 1
            if index != positive_index:
                if empty_kind == "nxdomain":
                    raise dns.resolver.NXDOMAIN()
                raise dns.resolver.NoAnswer()
            if kind == "AAAA":
                raise dns.resolver.NoAnswer()
            return ["192.0.2.1"]
        if kind == "AAAA":
            raise dns.resolver.NoAnswer()
        return ["192.0.2.99" if candidate_distinct else "192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS(
        "example.com", workers=1, rate=1000, event=lambda *args: events.append(args)
    )
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert not result.exists and not result.wildcard and not result.cancelled
    assert result.error == (
        "wildcard calibration inconclusive: inconsistent wildcard probe address presence"
    )
    assert result.ip_addresses == ["192.0.2.99" if candidate_distinct else "192.0.2.1"]
    assert "example.com" not in validator._wildcards
    assert len(probes) == 2
    expected_queries = 5 if empty_kind == "nxdomain" else 6
    assert validator.query_count == expected_queries
    assert len([event for kind, event in events if kind == "dns_start"]) == expected_queries

    # Inconsistency must not become a persistent cached wildcard or miss. A
    # later complete calibration can still establish an ordinary positive.
    consistently_negative[0] = True
    retried = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]
    assert retried.exists and not retried.wildcard and retried.error is None
    assert len(probes) == 4
    assert validator._wildcards["example.com"] == (set(), False, None)
    assert validator.query_count == expected_queries + 4


@pytest.mark.parametrize("candidate_distinct", [False, True])
def test_wildcard_probe_address_then_nxdomain_is_inconclusive_not_stable(
    monkeypatch, candidate_distinct
):
    events = []

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            if kind == "AAAA":
                raise dns.resolver.NXDOMAIN()
            return ["192.0.2.1"]
        if kind == "AAAA":
            raise dns.resolver.NoAnswer()
        return ["192.0.2.99" if candidate_distinct else "192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS(
        "example.com", workers=1, rate=1000, event=lambda *args: events.append(args)
    )
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert not result.exists and not result.wildcard and not result.cancelled
    assert result.error == (
        "wildcard calibration inconclusive: "
        "inconsistent wildcard probe: address answer followed by NXDOMAIN"
    )
    assert "example.com" not in validator._wildcards
    assert validator.query_count == 6
    contradictions = [event for kind, event in events if kind == "dns_end" and event["error"]]
    assert len(contradictions) == 2
    assert all(
        event["probe"] and event["query_kind"] == "AAAA" and event["status"] == "nxdomain"
        for event in contradictions
    )


def test_candidate_address_then_nxdomain_retains_ordinary_positive_semantics(monkeypatch):
    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-") or kind == "AAAA":
            raise dns.resolver.NXDOMAIN()
        return ["192.0.2.99"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, rate=1000)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert result.exists and not result.wildcard and result.error is None
    assert result.ip_addresses == ["192.0.2.99"]
    assert validator._wildcards["example.com"] == (set(), False, None)
    assert validator.query_count == 4


def test_partial_wildcard_probe_failure_is_not_success_or_stable_calibration(monkeypatch):
    probes = {}

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            index = probes.setdefault(name, len(probes))
            if index == 1:
                raise dns.exception.Timeout()
            return ["192.0.2.1"] if kind == "A" else ["2001:db8::1"]
        return ["192.0.2.99"] if kind == "A" else ["2001:db8::99"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert not result.exists
    assert not result.wildcard
    assert "wildcard calibration inconclusive" in result.error
    assert "example.com" not in validator._wildcards
    assert len(probes) == 2
    assert validator.query_count == 6


@pytest.mark.parametrize("failed_kind", ["A", "AAAA"])
@pytest.mark.parametrize("candidate_distinct", [False, True])
def test_partial_record_type_probe_failure_cannot_establish_wildcard_truth(
    monkeypatch, failed_kind, candidate_distinct
):
    probes = set()
    events = []

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            probes.add(name)
            if kind == failed_kind:
                raise dns.exception.Timeout()
        suffix = "99" if candidate_distinct and not name.startswith("enumeraite-") else "1"
        return ["192.0.2." + suffix] if kind == "A" else ["2001:db8::" + suffix]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS(
        "example.com", workers=1, rate=1000, event=lambda *args: events.append(args)
    )
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    # Neither matching partial addresses nor a new unseen address prove that a
    # host is explicit or wildcard-derived while a record type is unavailable.
    assert not result.exists and not result.wildcard and not result.cancelled
    assert "wildcard calibration inconclusive" in result.error
    assert "example.com" not in validator._wildcards
    assert len(probes) == 2
    assert validator.query_count == 6
    assert len([event for kind, event in events if kind == "dns_start"]) == 6
    failures = [event for kind, event in events if kind == "dns_end" and event["error"]]
    assert len(failures) == 2
    assert all(event["probe"] and event["query_kind"] == failed_kind for event in failures)


@pytest.mark.parametrize("failed_kind", ["A", "AAAA"])
def test_partial_candidate_answer_still_verifies_after_complete_negative_wildcard_checks(
    monkeypatch, failed_kind
):
    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            raise dns.resolver.NXDOMAIN()
        if kind == failed_kind:
            raise dns.exception.Timeout()
        return ["192.0.2.1"] if kind == "A" else ["2001:db8::1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, rate=1000)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert result.exists and not result.wildcard and result.error is None
    assert len(result.ip_addresses) == 1
    assert validator._wildcards["example.com"] == (set(), False, None)
    assert validator.query_count == 4


def test_partial_probe_at_deadline_is_inconclusive_and_does_not_spend_extra_queries(monkeypatch):
    now = [10.0]
    calls = []
    monkeypatch.setattr(scan_dns, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def resolve(name, kind, lifetime):
        calls.append((name, kind, lifetime))
        if name.startswith("enumeraite-"):
            now[0] = 11.0
        return ["192.0.2.1"] if kind == "A" else ["2001:db8::1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, rate=10)
    monkeypatch.setattr(
        validator._cancel, "wait", lambda duration: now.__setitem__(0, now[0] + duration) or False
    )
    result = validator.validate(["api.example.com"], deadline=11.0)[0]

    assert not result.exists and not result.wildcard and not result.cancelled
    assert "wildcard calibration inconclusive: scan deadline reached" in result.error
    assert not validator._wildcards
    assert validator.query_count == len(calls) == 3
    assert [kind for _, kind, _ in calls] == ["A", "AAAA", "A"]
    assert all(0 < lifetime <= 1 for _, _, lifetime in calls)


def test_cancellation_after_partial_probe_retains_unfinished_status(monkeypatch):
    cancelled = threading.Event()
    calls = []

    def resolve(name, kind, lifetime):
        calls.append((name, kind))
        if name.startswith("enumeraite-"):
            cancelled.set()
        return ["192.0.2.1"] if kind == "A" else ["2001:db8::1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, rate=1000, cancel_event=cancelled)
    result = validator.validate(["api.example.com"], deadline=time.monotonic() + 2)[0]

    assert result.cancelled and not result.exists and not result.wildcard
    assert "wildcard calibration inconclusive: scan cancelled" in result.error
    assert not validator._wildcards
    assert cancelled.is_set()
    assert validator.query_count == len(calls) == 3
    assert [kind for _, kind in calls] == ["A", "AAAA", "A"]


def test_failed_calibration_is_inconclusive_and_retried(monkeypatch):
    fail_probes = [True]
    probes = []

    def resolve(name, kind, lifetime):
        if name.startswith("enumeraite-"):
            probes.append(name)
            if fail_probes[0]:
                raise dns.exception.Timeout()
            raise dns.resolver.NXDOMAIN()
        return ["192.0.2.1"] if kind == "A" else []

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", rate=1000)
    result = validator.validate(["a.example.com"], deadline=time.monotonic() + 2)[0]
    assert not result.exists
    assert "inconclusive" in result.error
    assert "example.com" not in validator._wildcards
    prior_probes = len(probes)
    fail_probes[0] = False
    result = validator.validate(["b.example.com"], deadline=time.monotonic() + 2)[0]
    assert result.exists
    assert len(probes) == prior_probes + 2


def test_invalid_or_out_of_scope_names_never_reach_dns(monkeypatch):
    monkeypatch.setattr(
        scan_dns.dns.resolver, "Resolver", lambda: pytest.fail("unexpected DNS query")
    )
    validator = ScanDNS("example.com")
    names = ["example.com", "outside.test", "example.com.evil.test", "bad_.example.com"]
    results = validator.validate(names, deadline=time.monotonic() + 1)
    assert all(not result.exists and "scope" in result.error for result in results)
    assert validator.query_count == 0


def test_no_query_starts_after_deadline_and_remaining_lifetime_is_bounded(monkeypatch):
    now = [10.0]
    calls = []
    monkeypatch.setattr(
        scan_dns, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _: None)
    )

    def resolve(name, kind, lifetime):
        calls.append((name, lifetime))
        now[0] = 10.2
        raise dns.exception.Timeout()

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=1, timeout=5)
    results = validator.validate(["a.example.com", "b.example.com"], deadline=10.1)
    assert len(calls) == validator.query_count == 1
    assert calls[0][1] == pytest.approx(0.1)
    assert all(not result.exists and result.error for result in results)


def test_wildcard_parent_cap_is_visible_without_extra_probes(monkeypatch):
    validator = ScanDNS("example.com")
    validator._parent_locks = {str(i): threading.Lock() for i in range(1000)}
    monkeypatch.setattr(validator, "_resolve", lambda *_: pytest.fail("unexpected calibration"))
    addresses, detected, error = validator._wildcard("new.example.com", time.monotonic() + 2)
    assert not addresses
    assert not detected
    assert "parent limit" in error


@pytest.mark.parametrize(
    "options", [{"workers": 0}, {"workers": 101}, {"rate": 0}, {"timeout": 31}]
)
def test_invalid_dns_limits_are_rejected(options):
    with pytest.raises(ValueError):
        ScanDNS("example.com", **options)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interruption_wakes_rate_waiters_cancels_queued_work_and_allows_reuse(monkeypatch, failure):
    waiting = threading.Event()
    queries = []

    def resolve(name, kind, lifetime):
        queries.append(name)
        raise dns.resolver.NXDOMAIN()

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.com", workers=4, rate=0.1)
    original_wait = validator._cancel.wait

    def track_wait(duration):
        waiting.set()
        return original_wait(duration)

    monkeypatch.setattr(validator._cancel, "wait", track_wait)
    executor_class = scan_dns.ThreadPoolExecutor

    class InterruptingExecutor(executor_class):
        def map(self, *args, **kwargs):
            results = super().map(*args, **kwargs)
            assert waiting.wait(1), "workers did not reach a scheduled rate wait"
            # Keep the real iterator alive: its destructor must not be what
            # cancels queued futures. validate() owns that cleanup itself.
            self.results = results
            raise failure("simulated interruption")

    monkeypatch.setattr(scan_dns, "ThreadPoolExecutor", InterruptingExecutor)
    started = time.monotonic()
    with pytest.raises(failure, match="simulated interruption"):
        validator.validate(
            [f"host{i}.example.com" for i in range(12)], deadline=time.monotonic() + 60
        )
    assert time.monotonic() - started < 1
    assert len(queries) == validator.query_count == 1
    assert not validator._cancel.is_set()
    # Only the last actual request's 10-second interval remains reserved,
    # rather than the 40+ seconds of slots allocated to cancelled workers.
    assert validator._last_query is not None
    assert validator._next_query == pytest.approx(validator._last_query + 10, abs=0.001)

    monkeypatch.setattr(scan_dns, "ThreadPoolExecutor", executor_class)
    real_clock = time.monotonic
    monkeypatch.setattr(scan_dns, "time", SimpleNamespace(monotonic=lambda: real_clock() + 30))
    results = validator.validate(["resumed.example.com"], deadline=real_clock() + 31)
    assert len(results) == 1
    assert not results[0].exists
    assert results[0].error is None
    assert queries[-1] == "resumed.example.com"
    assert validator.query_count == 2


def test_interruption_waits_only_for_bounded_inflight_dns_and_skips_followup_queries(monkeypatch):
    entered = threading.Event()
    queries = []

    def resolve(name, kind, lifetime):
        entered.set()
        queries.append((name, kind))
        time.sleep(0.04)
        return ["192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    executor_class = scan_dns.ThreadPoolExecutor

    class InterruptingExecutor(executor_class):
        def map(self, *args, **kwargs):
            self.results = super().map(*args, **kwargs)
            assert entered.wait(1)
            raise KeyboardInterrupt()

    monkeypatch.setattr(scan_dns, "ThreadPoolExecutor", InterruptingExecutor)
    validator = ScanDNS("example.com", workers=1, rate=1000, timeout=0.1)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        validator.validate(["api.example.com", "mail.example.com"], deadline=started + 2)
    assert time.monotonic() - started < 0.5
    assert queries == [("api.example.com", "A")]
    assert not validator._cancel.is_set()
    assert not validator._wildcards
