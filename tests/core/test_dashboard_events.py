"""Live scanner telemetry and resumable cooperative pause behavior."""

import copy
import threading
import time

import dns.resolver

from enumeraite.core import scan_dns
from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.passive import PassiveResult
from enumeraite.core.scan_budget import ModelPrice, SpendLedger
from enumeraite.core.scan_dns import ScanDNS
from enumeraite.providers.llm_provider import ChatResult, LLMProvider


def install_resolver(monkeypatch, resolve):
    class Resolver:
        def resolve(self, name, kind, *, lifetime, search):
            return resolve(name, kind)

    monkeypatch.setattr(scan_dns.dns.resolver, "Resolver", Resolver)


def test_query_events_pair_every_lookup_and_distinguish_owned_wildcard_probes(monkeypatch):
    events = []

    def resolve(name, kind):
        if name.startswith("enumeraite-"):
            raise dns.resolver.NXDOMAIN()
        if kind == "AAAA":
            raise dns.resolver.NoAnswer()
        return ["192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS(
        "example.test", rate=1000, event=lambda kind, data: events.append((kind, data))
    )
    result = validator.validate(["api.example.test"], deadline=time.monotonic() + 2)[0]
    assert result.exists
    starts = {data["query_id"]: data for kind, data in events if kind == "dns_start"}
    ends = {data["query_id"]: data for kind, data in events if kind == "dns_end"}
    assert set(starts) == set(ends) == {1, 2, 3, 4}
    assert len(events) == 8
    assert [data["status"] for data in ends.values()] == [
        "answered",
        "no_answer",
        "nxdomain",
        "nxdomain",
    ]
    assert sum(data["probe"] for data in starts.values()) == 2
    for query_id, start in starts.items():
        end = ends[query_id]
        assert all(end[key] == value for key, value in start.items())
        assert start["worker_id"]
        assert start["candidate"] == "api.example.test"
        assert start["query_kind"] in {"A", "AAAA"}
        assert start["hostname"].endswith(".example.test")
        assert end["elapsed_seconds"] >= 0


def test_query_error_also_finishes_active_worker_telemetry(monkeypatch):
    events = []

    def resolve(name, kind):
        raise dns.resolver.NoNameservers()

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS(
        "example.test", rate=1000, event=lambda kind, data: events.append((kind, data))
    )
    result = validator.validate(["api.example.test"], deadline=time.monotonic() + 2)[0]
    assert result.error
    assert [kind for kind, _ in events] == ["dns_start", "dns_end", "dns_start", "dns_end"]
    assert all(data["status"] == "error" for kind, data in events if kind == "dns_end")


def test_external_pause_wakes_rate_waiters_and_is_never_cleared(monkeypatch):
    pause = threading.Event()
    waiting = threading.Event()
    queries = []

    def resolve(name, kind):
        queries.append(name)
        raise dns.resolver.NXDOMAIN()

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.test", workers=3, rate=0.1, cancel_event=pause)
    original_wait = pause.wait

    def wait(duration):
        waiting.set()
        return original_wait(duration)

    monkeypatch.setattr(pause, "wait", wait)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.extend(
            validator.validate(
                [f"host{i}.example.test" for i in range(10)], deadline=time.monotonic() + 60
            )
        )
    )
    worker.start()
    assert waiting.wait(1)
    started = time.monotonic()
    pause.set()
    worker.join(1)
    assert not worker.is_alive()
    assert time.monotonic() - started < 1
    assert len(queries) == 1
    assert len(outcomes) == 10
    assert sum(result.cancelled for result in outcomes) == 9
    assert pause.is_set()
    assert validator._last_query is not None
    assert validator._next_query <= validator._last_query + 10.01


def test_prepaused_validator_never_queries_or_emits_worker_starts(monkeypatch):
    pause = threading.Event()
    pause.set()
    events = []
    monkeypatch.setattr(scan_dns.dns.resolver, "Resolver", lambda: 1 / 0)
    validator = ScanDNS("example.test", cancel_event=pause, event=lambda *args: events.append(args))
    results = validator.validate(["api.example.test"], deadline=time.monotonic() + 1)
    assert results[0].cancelled
    assert not events
    assert pause.is_set()


def test_pause_saves_unqueried_names_as_pending_and_resume_reuses_completed_results(monkeypatch):
    pause = threading.Event()
    queries = []
    saved = []

    def resolve(name, kind):
        queries.append(name)
        raise dns.resolver.NXDOMAIN()

    def event(kind, data):
        if kind == "dns_end":
            pause.set()

    install_resolver(monkeypatch, resolve)
    options = ScanOptions(passive=False, max_candidates=3)
    first = DiscoverySession(
        "example.test",
        options,
        ScanDNS("example.test", workers=1, cancel_event=pause, event=event),
        stop_requested=pause.is_set,
        checkpoint=lambda state: saved.append(copy.deepcopy(state)),
    )
    report = first.run(["a.example.test", "b.example.test", "c.example.test"])
    assert report["stop_reason"] == "paused; scan saved"
    assert report["attempted"] == 1
    assert report["queued"] == 2
    assert first.nodes["a.example.test"]["status"] == "miss"
    for name in ("b.example.test", "c.example.test"):
        assert first.nodes[name]["status"] == "pending"
        assert first.nodes[name]["attempts"] == 0
    validate_checkpoint(saved[-1], "example.test")
    resumed = DiscoverySession(
        "example.test", options, ScanDNS("example.test", rate=1000), state=saved[-1]
    )
    assert resumed.run()["attempted"] == 3
    assert queries == ["a.example.test", "b.example.test", "c.example.test"]


def test_pause_during_wildcard_calibration_preserves_retry_allowance(monkeypatch):
    pause = threading.Event()

    def resolve(name, kind):
        if name.startswith("enumeraite-"):
            pause.set()
            raise dns.resolver.NXDOMAIN()
        return ["192.0.2.1"]

    install_resolver(monkeypatch, resolve)
    validator = ScanDNS("example.test", rate=1000, cancel_event=pause)
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=1),
        validator,
        stop_requested=pause.is_set,
    )
    result = session.run(["api.example.test"])
    assert result["stop_reason"] == "paused; scan saved"
    assert result["attempted"] == 0
    assert result["queued"] == 1
    assert session.nodes["api.example.test"]["attempts"] == 0
    assert not validator._wildcards
    validate_checkpoint(session.state, "example.test")


def test_pause_before_start_saves_valid_empty_checkpoint_without_passive_work():
    saved = []
    session = DiscoverySession(
        "example.test",
        ScanOptions(),
        object(),
        stop_requested=lambda: True,
        passive_fetch=lambda *args, **kwargs: 1 / 0,
        checkpoint=lambda state: saved.append(copy.deepcopy(state)),
    )
    assert session.run()["stop_reason"] == "paused; scan saved"
    assert not session.state["initialized"]
    validate_checkpoint(saved[-1], "example.test")


def test_pause_while_collecting_passive_saves_evidence_but_never_checks_dns():
    pause = threading.Event()

    def passive(*args, **kwargs):
        pause.set()
        return PassiveResult(["known.example.test"], {"fixture": ["known.example.test"]}, [])

    session = DiscoverySession(
        "example.test", ScanOptions(), object(), stop_requested=pause.is_set, passive_fetch=passive
    )
    assert session.run()["stop_reason"] == "paused; scan saved"
    assert session.nodes["known.example.test"]["status"] == "pending"
    assert session.state["initialized"]
    validate_checkpoint(session.state, "example.test")


def test_candidate_and_final_classification_events_are_detached_snapshots():
    events = []

    class Validator:
        def validate(self, values, *, deadline):
            return [DNSResult(value, True, ["192.0.2.1"], 0) for value in values]

    def event(kind, data):
        events.append((kind, copy.deepcopy(data)))
        if kind in {"candidate", "finding", "result"}:
            data["addresses"].append("192.0.2.99")
            data["status"] = "broken"

    session = DiscoverySession(
        "example.test", ScanOptions(passive=False, max_candidates=1), Validator(), event=event
    )
    session.run(["api.example.test"])
    assert session.nodes["api.example.test"]["status"] == "verified"
    assert session.nodes["api.example.test"]["addresses"] == ["192.0.2.1"]
    assert [data["status"] for kind, data in events if kind in {"candidate", "result"}] == [
        "pending",
        "verified",
    ]
    assert any(data.get("stage") == "dns" for kind, data in events if kind == "phase")


def test_model_pause_does_not_admit_followup_requests_or_spend_reservation():
    pause = threading.Event()

    class Model(LLMProvider):
        def get_provider_name(self):
            return "fixture"

        def chat(self, system, user, schema):
            pause.set()
            return ChatResult('{"candidates":[]}', input_tokens=5, output_tokens=5)

    ledger = SpendLedger(1, ModelPrice(1, 2))
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False),
        object(),
        provider=Model({"model": "fixture"}),
        ledger=ledger,
        stop_requested=pause.is_set,
    )
    session._enqueue("api.example.test", source="seed")
    session.nodes["api.example.test"].update(status="verified", attempts=1)
    session.state["attempted"] = 1
    assert session._model_candidates(session.findings) == 0
    assert ledger.calls == 1
    assert ledger.reserved_usd == 0
    assert not session.state["warnings"]
    assert session._stop() == "paused; scan saved"


def test_missing_results_during_pause_remain_pending():
    pause = threading.Event()

    class Validator:
        def validate(self, values, *, deadline):
            pause.set()
            return []

    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False, max_candidates=1),
        Validator(),
        stop_requested=pause.is_set,
    )
    assert session.run(["api.example.test"])["attempted"] == 0
    assert session.nodes["api.example.test"]["status"] == "pending"
    validate_checkpoint(session.state, "example.test")
