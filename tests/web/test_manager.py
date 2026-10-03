import json
import stat
import threading
import time
import uuid
from pathlib import Path

import pytest

from enumeraite.core.discovery_session import DiscoverySession
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.passive import PassiveResult
from enumeraite.core.scan_budget import SpendLedger, price_for
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.web.manager import ENDPOINTS, MAX_EVENTS, ScanManager, default_storage_dir


class FixtureDNS:
    def __init__(self, target, **kwargs):
        self.target = target
        self.event = kwargs["event"]
        self.cancel = kwargs["cancel_event"]
        self.calls = []

    def validate(self, values, **kwargs):
        self.calls.extend(values)
        return [
            DNSResult(
                value,
                value.startswith("ny-0"),
                ["192.0.2.1"] if value.startswith("ny-0") else [],
                0,
            )
            for value in values
        ]


def fixture_passive(target, **kwargs):
    names = [f"ny-01-iphone.{target}"]
    return PassiveResult(names, {"fixture": names}, [], False)


class FixtureProvider(LLMProvider):
    def chat(self, system, user, schema):
        return ChatResult('{"candidates": []}', input_tokens=100, output_tokens=10)

    def get_provider_name(self):
        preset = self.config.get("preset", "minimax")
        return "anthropic" if preset == "anthropic" else f"openai-compat:{preset}"


def make_manager(tmp_path, **kwargs):
    return ScanManager(tmp_path, dns_factory=FixtureDNS, passive_fetch=fixture_passive, **kwargs)


def wait_complete(manager, scan_id):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(scan_id)
        if snapshot["status"] not in {"running", "pausing"}:
            return snapshot
        time.sleep(0.005)
    pytest.fail("Fixture scan did not finish")


def start_free(manager, **kwargs):
    return manager.start({"target": "example.test", "options": {"max_candidates": 30}, **kwargs})


def test_free_scan_progress_patterns_export_and_private_checkpoint(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = start_free(manager)
    result = wait_complete(manager, scan_id)
    assert result["status"] == "completed", result["error"]
    assert result["summary"]["verified"] >= 1
    assert result["summary"]["known_verified"] == 1
    assert result["patterns"]
    assert result["summary"]["ledger"] is None
    assert result["events"] and not result["active"]
    assert result["can_resume"]
    assert len(scan_id) == 32
    report = manager.report(scan_id)
    assert report["findings"] and report["nodes"]
    methods = report["discovery_methods"]
    assert methods == result["summary"]["discovery_methods"]
    assert sum(row["candidates"] for row in methods) == len(report["nodes"])
    assert sum(row["verified"] for row in methods) == report["verified"]
    assert sum(row["checked"] for row in methods) == report["attempted"]
    assert str(tmp_path) not in json.dumps(report)
    path = tmp_path / f"{scan_id}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    saved = json.loads(path.read_text())
    assert saved["_ui"]["status"] == "completed"


def test_vocabulary_capability_is_advertised_and_defaults_off(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.list_state()["features"]["vocabulary_breadth"] is True
    scan_id = start_free(manager)
    result = wait_complete(manager, scan_id)
    assert result["options"]["vocabulary_breadth"] is False
    saved = json.loads((tmp_path / f"{scan_id}.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is False


def test_vocabulary_option_survives_reload_and_resume_until_explicitly_disabled(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = start_free(manager, options={"max_candidates": 1, "vocabulary_breadth": True})
    result = wait_complete(manager, scan_id)
    assert result["options"]["vocabulary_breadth"] is True
    saved = json.loads((tmp_path / f"{scan_id}.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is True
    manager.close()

    restored = make_manager(tmp_path)
    assert restored.snapshot(scan_id)["options"]["vocabulary_breadth"] is True
    restored.resume(scan_id, {"options": {"max_candidates": 2}})
    resumed = wait_complete(restored, scan_id)
    assert resumed["status"] == "completed", resumed["error"]
    assert resumed["options"]["vocabulary_breadth"] is True

    restored.resume(scan_id, {"options": {"vocabulary_breadth": False}})
    disabled = wait_complete(restored, scan_id)
    assert disabled["status"] == "completed", disabled["error"]
    assert disabled["options"]["vocabulary_breadth"] is False
    saved = json.loads((tmp_path / f"{scan_id}.json").read_text())
    assert saved["options"]["vocabulary_breadth"] is False


def test_hostname_input_starts_from_seed_and_expands_siblings(tmp_path):
    def no_passive(*args, **kwargs):
        pytest.fail("A seed-first scan must not collect public names by default")

    manager = ScanManager(tmp_path, dns_factory=FixtureDNS, passive_fetch=no_passive)
    scan_id = manager.start(
        {"target": "ny-01-iphone.example.com", "options": {"max_candidates": 8}}
    )
    result = wait_complete(manager, scan_id)
    assert result["target"] == "example.com"
    assert result["options"]["passive"] is False
    assert result["nodes"][0]["value"] == "ny-01-iphone.example.com"
    assert result["nodes"][0]["source"] == "seed"
    assert result["nodes"][0]["status"] == "verified"
    assert any(
        node["value"] == "ny-02-iphone.example.com" and node["status"] == "verified"
        for node in result["nodes"]
    )
    assert not any(node["source"] == "bootstrap" for node in result["nodes"])


def test_domain_without_seed_collects_passive_names_by_default(tmp_path):
    calls = []

    def passive(target, **kwargs):
        calls.append(target)
        return fixture_passive(target, **kwargs)

    manager = ScanManager(tmp_path, dns_factory=FixtureDNS, passive_fetch=passive)
    scan_id = manager.start({"target": "example.com", "options": {"max_candidates": 8}})
    result = wait_complete(manager, scan_id)
    assert result["options"]["passive"] is True
    assert calls == ["example.com"]
    assert result["nodes"][0]["source"] == "passive"


def test_seeded_scan_can_explicitly_collect_passive_names(tmp_path):
    calls = []

    def passive(target, **kwargs):
        calls.append(target)
        return fixture_passive(target, **kwargs)

    manager = ScanManager(tmp_path, dns_factory=FixtureDNS, passive_fetch=passive)
    scan_id = manager.start(
        {"target": "ny-01-iphone.example.com", "options": {"passive": True, "max_candidates": 8}}
    )
    result = wait_complete(manager, scan_id)
    assert result["options"]["passive"] is True
    assert calls == ["example.com"]
    assert result["nodes"][0]["source"] == "seed"


def test_explicit_seed_list_and_narrow_scope_remain_in_scope(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = manager.start(
        {
            "target": "ny-01-iphone.dev.example.com",
            "scope": "dev.example.com",
            "seeds": ["ny-01-iphone.dev.example.com", "ny-02-iphone.dev.example.com"],
            "options": {"max_candidates": 8},
        }
    )
    result = wait_complete(manager, scan_id)
    assert result["target"] == "dev.example.com"
    assert result["options"]["passive"] is False
    assert all(node["value"].endswith(".dev.example.com") for node in result["nodes"])
    assert len([node for node in result["nodes"] if node["source"] == "seed"]) == 2
    manager.resume(scan_id, {"options": {"max_candidates": 10}})
    resumed = wait_complete(manager, scan_id)
    assert resumed["target"] == "dev.example.com"
    assert resumed["options"]["passive"] is False


def test_invalid_seed_scope_rejected_before_background_launch(tmp_path):
    manager = make_manager(tmp_path)
    for payload in (
        {"target": "example.com", "seeds": ["api.foreign.com"]},
        {"target": "api.example.com", "scope": "foreign.com"},
        {"target": "api.example.com", "scope": "com"},
    ):
        with pytest.raises(ValueError):
            manager.start(payload)
    assert manager.list_state()["scans"] == []


def test_deep_starting_hostname_requires_sufficient_depth_before_launch(tmp_path):
    manager = make_manager(tmp_path)
    payload = {"target": "ny-01.api.dev.us.east.example.com"}
    with pytest.raises(ValueError, match="depth"):
        manager.start(payload)
    assert manager.list_state()["scans"] == []
    scan_id = manager.start({**payload, "options": {"max_depth": 5, "max_candidates": 2}})
    result = wait_complete(manager, scan_id)
    assert result["nodes"][0]["value"] == payload["target"]
    assert result["nodes"][0]["status"] == "verified"
    with pytest.raises(ValueError, match="depth"):
        manager.resume(scan_id, {"seeds": ["ny-01.a.b.c.d.e.example.com"]})


def test_snapshots_are_detached_and_history_compact(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = start_free(manager)
    snapshot = wait_complete(manager, scan_id)
    snapshot["nodes"][0]["status"] = "MUTATED"
    snapshot["options"]["workers"] = -1
    snapshot["summary"]["warnings"].append("MUTATED")
    assert "MUTATED" not in json.dumps(manager.snapshot(scan_id))
    listing = manager.list_state()
    assert listing["credentials"] == {
        "minimax": False,
        "openai": False,
        "anthropic": False,
        "jev": False,
    }
    assert not ({"nodes", "patterns", "events", "active"} & listing["scans"][0].keys())


def test_history_restores_without_credentials_and_resume_preserves_work(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = start_free(manager)
    previous = wait_complete(manager, scan_id)
    reloaded = make_manager(tmp_path)
    assert reloaded.snapshot(scan_id)["summary"]["attempted"] == previous["summary"]["attempted"]
    assert reloaded.list_state()["credentials"] == {
        "minimax": False,
        "openai": False,
        "anthropic": False,
        "jev": False,
    }
    reloaded.resume(scan_id, {"options": {"max_candidates": 40}})
    result = wait_complete(reloaded, scan_id)
    assert result["status"] == "completed"
    assert result["summary"]["attempted"] >= previous["summary"]["attempted"]
    assert result["summary"]["elapsed_seconds"] >= previous["summary"]["elapsed_seconds"]


def test_active_queries_live_nodes_pause_and_single_run(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class HoldingDNS(FixtureDNS):
        def validate(self, values, **kwargs):
            self.event(
                "dns_start",
                {
                    "query_id": "42",
                    "worker_id": 2,
                    "hostname": values[0],
                    "candidate": values[0],
                    "query_kind": "A",
                    "probe": False,
                },
            )
            entered.set()
            release.wait(3)
            self.event("dns_end", {"query_id": "42"})
            return []

    manager = ScanManager(tmp_path, dns_factory=HoldingDNS, passive_fetch=fixture_passive)
    scan_id = start_free(manager)
    assert entered.wait(2)
    snapshot = manager.snapshot(scan_id)
    assert snapshot["nodes"] and snapshot["summary"]["queued"]
    assert snapshot["active"][0]["query_type"] == "A"
    assert snapshot["active"][0]["candidate"] == "ny-01-iphone.example.test"
    with pytest.raises(RuntimeError):
        start_free(manager)
    with pytest.raises(RuntimeError):
        manager.resume(scan_id)
    manager.pause(scan_id)
    assert manager.snapshot(scan_id)["status"] == "pausing"
    release.set()
    result = wait_complete(manager, scan_id)
    assert result["status"] == "paused"
    assert result["summary"]["attempted"] == 0
    assert result["summary"]["queued"] > 0
    assert not result["active"]


def test_close_requests_pause_clears_key_and_rejects_further_work(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def holding_passive(target, **kwargs):
        entered.set()
        release.wait(2)
        return fixture_passive(target)

    manager = ScanManager(tmp_path, dns_factory=FixtureDNS, passive_fetch=holding_passive)
    manager.set_minimax_key("fake-secret-for-tests")
    scan_id = start_free(manager)
    assert entered.wait(2)
    manager.close()
    assert not manager.list_state()["credentials"]["minimax"]
    with pytest.raises(RuntimeError):
        start_free(manager)
    with pytest.raises(RuntimeError):
        manager.set_minimax_key("new-fake-for-tests")
    release.set()
    assert wait_complete(manager, scan_id)["status"] == "paused"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"target": "https://example.test"},
        {"target": "127.0.0.1"},
        {"target": "example.test", "path": "/tmp/secret"},
        {"target": "example.test", "options": {"max_candidates": 10001}},
        {"target": "example.test", "options": {"workers": True}},
        {"target": "example.test", "options": {"unknown": 1}},
        {"target": "example.test", "provider": []},
        {"target": "example.test", "provider": "openai"},
        {"target": "example.test", "model": "MiniMax-M3"},
        {"target": "example.test", "budget_usd": 1},
        {"target": "example.test", "budget_usd": float("nan")},
        {"target": "example.test", "seeds": ["outside.test"]},
        {"target": "example.test", "seeds": "a.example.test"},
    ],
)
def test_request_validation_is_strict(tmp_path, payload):
    manager = make_manager(tmp_path)
    with pytest.raises(ValueError):
        manager.start(payload)
    assert not manager.list_state()["scans"]


@pytest.mark.parametrize(
    "key",
    [
        None,
        "a",
        "a" * 15,
        "one two",
        "bad\nkey",
        "bad\x00key",
        "a" * 2049,
        "é",
        "cached_input_usd_per_million",
    ],
)
def test_key_validation_does_not_echo_input(tmp_path, key):
    manager = make_manager(tmp_path)
    with pytest.raises(ValueError) as error:
        manager.set_minimax_key(key)
    if isinstance(key, str) and len(key) >= 8:
        assert key not in str(error.value)


def test_memory_only_key_fixed_endpoint_no_environment_fallback(tmp_path, monkeypatch):
    seen = []

    def provider_factory(config):
        seen.append(config.copy())
        return FixtureProvider(config)

    monkeypatch.setenv("MINIMAX_API_KEY", "must-not-use-environment-key")
    manager = make_manager(tmp_path, provider_factory=provider_factory)
    payload = {
        "target": "example.test",
        "provider": "minimax",
        "model": "MiniMax-M2.7",
        "options": {"max_candidates": 1},
    }
    with pytest.raises(ValueError, match="API key"):
        manager.start(payload)
    assert manager.set_minimax_key("memory-only-test-secret")
    scan_id = manager.start(payload)
    snapshot = wait_complete(manager, scan_id)
    assert snapshot["status"] == "completed", snapshot["error"]
    assert seen[0]["api_key"] == "memory-only-test-secret"
    assert seen[0]["base_url"] == "https://api.minimax.io/v1"
    assert seen[0]["max_retries"] == 0
    assert snapshot["summary"]["ledger"]["limit_usd"] == "1"
    assert "memory-only-test-secret" not in json.dumps(snapshot)
    assert "memory-only-test-secret" not in (tmp_path / f"{scan_id}.json").read_text()
    assert manager.set_minimax_key("") is False
    with pytest.raises(ValueError, match="API key"):
        manager.resume(scan_id)


def test_arbitrary_secret_redacted_from_warnings_events_and_checkpoints(tmp_path):
    secret = "arbitrary-format-secret"

    class WarningSession(DiscoverySession):
        def run(self, seeds=()):
            self._warn(f"Provider returned {secret}")
            self.event("phase", {"message": f"debug {secret}"})
            return super().run(seeds)

    manager = make_manager(
        tmp_path, session_factory=WarningSession, provider_factory=FixtureProvider
    )
    manager.set_minimax_key(secret)
    scan_id = manager.start(
        {"target": "example.test", "provider": "minimax", "options": {"max_candidates": 1}}
    )
    snapshot = wait_complete(manager, scan_id)
    assert snapshot["status"] == "completed"
    assert secret not in json.dumps(snapshot)
    assert secret not in json.dumps(manager.report(scan_id))
    assert secret not in (tmp_path / f"{scan_id}.json").read_text()
    assert "REDACTED" in json.dumps(snapshot)


def test_provider_constructor_errors_are_generic(tmp_path):
    def failure(config):
        raise RuntimeError(f"response contains {config['api_key']} and /private/confidential")

    manager = make_manager(tmp_path, provider_factory=failure)
    manager.set_minimax_key("secret-test-only")
    scan_id = manager.start({"target": "example.test", "provider": "minimax"})
    snapshot = wait_complete(manager, scan_id)
    assert snapshot["status"] == "error"
    assert not snapshot["can_resume"]
    assert "secret-test-only" not in json.dumps(snapshot)
    assert "/private/confidential" not in json.dumps(snapshot)


def test_saved_reservation_is_charged_once_and_budget_is_cumulative(tmp_path):
    manager = make_manager(tmp_path, provider_factory=FixtureProvider)
    manager.set_minimax_key("fixture-key-for-tests")
    scan_id = manager.start(
        {
            "target": "example.test",
            "provider": "minimax",
            "options": {"max_candidates": 1},
            "budget_usd": 5,
        }
    )
    wait_complete(manager, scan_id)
    path = tmp_path / f"{scan_id}.json"
    saved = json.loads(path.read_text())
    ledger = SpendLedger(5, price_for("minimax", "MiniMax-M3"), spent_usd=1)
    ledger.reserve(1000, 2000)
    saved["ledger"] = ledger.to_dict()
    path.write_text(json.dumps(saved))
    restored = make_manager(tmp_path, provider_factory=FixtureProvider)
    restored.set_minimax_key("fixture-key-for-tests")
    with pytest.raises(ValueError, match="cumulative"):
        restored.resume(scan_id, {"budget_usd": 1})
    with pytest.raises(ValueError, match="same provider"):
        restored.resume(scan_id, {"model": "MiniMax-M2.7"})
    restored.resume(scan_id, {"budget_usd": 6})
    result = wait_complete(restored, scan_id)
    assert result["status"] == "completed", result["error"]
    assert result["summary"]["ledger"]["limit_usd"] == "6"
    assert float(result["summary"]["ledger"]["spent_usd"]) > 1
    assert result["summary"]["ledger"]["uncertain_calls"] == 1
    spent = result["summary"]["ledger"]["spent_usd"]
    restored.resume(scan_id)
    again = wait_complete(restored, scan_id)
    assert again["summary"]["ledger"]["spent_usd"] == spent


def test_invalid_symlink_and_untrusted_checkpoint_files_are_ignored(tmp_path):
    (tmp_path / f"{uuid.uuid4().hex}.json").write_text("malformed")
    (tmp_path / "not-a-scan.json").write_text("{}")
    outside = tmp_path / "private-data.txt"
    outside.write_text("confidential-data")
    (tmp_path / f"{uuid.uuid4().hex}.json").symlink_to(outside)
    manager = make_manager(tmp_path)
    assert manager.list_state()["scans"] == []
    for scan_id in ["../private-data.txt", "../" + "a" * 32, "a" * 32]:
        with pytest.raises(KeyError):
            manager.snapshot(scan_id)


def test_event_ring_and_active_queries_are_bounded(tmp_path):
    manager = make_manager(tmp_path)
    scan_id = start_free(manager)
    wait_complete(manager, scan_id)
    row = manager._records[scan_id]
    for i in range(MAX_EVENTS + 20):
        manager._on_event(row, "phase", {"message": f"event {i}"}, "")
        manager._on_event(row, "dns_start", {"query_id": str(i), "hostname": "a.example.test"}, "")
    snapshot = manager.snapshot(scan_id)
    assert len(snapshot["events"]) == MAX_EVENTS
    assert len(snapshot["active"]) == 200
    assert snapshot["events"][-1]["message"] == f"event {MAX_EVENTS + 19}"


def test_default_directory_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_storage_dir() == Path(tmp_path) / "enumeraite" / "ui"


@pytest.mark.parametrize(
    "provider,model",
    [("openai", "gpt-4.1-mini"), ("openai", "gpt-4.1"), ("anthropic", "claude-sonnet-4-6")],
)
def test_hosted_providers_use_independent_keys_and_fixed_endpoints(tmp_path, provider, model):
    seen = []

    def factory(config):
        seen.append(config.copy())
        return FixtureProvider(config)

    manager = make_manager(tmp_path, provider_factory=factory)
    manager.set_api_key("minimax", "minimax-only-test")
    with pytest.raises(ValueError, match="API key"):
        manager.start({"target": "example.test", "provider": provider, "model": model})
    manager.set_api_key(provider, "provider-specific-test")
    scan_id = manager.start(
        {
            "target": "example.test",
            "provider": provider,
            "model": model,
            "options": {"max_candidates": 1},
        }
    )
    snapshot = wait_complete(manager, scan_id)
    assert snapshot["status"] == "completed", snapshot["error"]
    assert seen[0]["preset"] == provider
    assert seen[0]["base_url"] == ENDPOINTS[provider]
    assert seen[0]["api_key"] == "provider-specific-test"
    assert seen[0]["hybrid"] is False
    assert seen[0]["max_batch_calls"] == 1
    assert manager.list_state()["credentials"]["minimax"]
    assert manager.list_state()["credentials"][provider]
    assert "provider-specific-test" not in json.dumps(snapshot)
    restored = make_manager(tmp_path, provider_factory=factory)
    assert restored.snapshot(scan_id)["provider"] == provider
    assert restored.snapshot(scan_id)["model"] == model
    with pytest.raises(ValueError, match="API key"):
        restored.resume(scan_id)
    restored.set_api_key(provider, "replacement-test")
    restored.resume(scan_id)
    assert wait_complete(restored, scan_id)["status"] == "completed"
    assert seen[-1]["api_key"] == "replacement-test"


def test_key_provider_names_are_restricted(tmp_path):
    manager = make_manager(tmp_path)
    for name in ["local", "unknown", [], None]:
        with pytest.raises(ValueError):
            manager.set_api_key(name, "test")


def test_replacing_key_during_run_still_redacts_captured_old_key(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SecretSession(DiscoverySession):
        def run(self, seeds=()):
            entered.set()
            release.wait(2)
            self._warn(
                "first-key-value-for-tests second-key-value-for-tests unrelated-provider-key"
            )
            return super().run(seeds)

    manager = make_manager(
        tmp_path, session_factory=SecretSession, provider_factory=FixtureProvider
    )
    manager.set_api_key("minimax", "first-key-value-for-tests")
    manager.set_api_key("openai", "unrelated-provider-key")
    scan_id = manager.start(
        {"target": "example.test", "provider": "minimax", "options": {"max_candidates": 1}}
    )
    assert entered.wait(2)
    manager.set_api_key("minimax", "second-key-value-for-tests")
    release.set()
    snapshot = wait_complete(manager, scan_id)
    assert snapshot["status"] == "completed"
    combined = json.dumps(snapshot) + (tmp_path / f"{scan_id}.json").read_text()
    for secret in ["first-key-value", "second-key-value", "unrelated-provider-key"]:
        assert secret not in combined


def test_redaction_preserves_fixed_schema_keys_and_redacts_dynamic_keys(tmp_path):
    manager = make_manager(tmp_path)
    # Fixed schema must remain valid even if a credential's text coincides with
    # a field. Real credential entry rejects such implausible values separately.
    data = manager._redact(
        {
            "cached_input_usd_per_million": "cached_input_usd_per_million",
            "dynamic-cached_input_usd_per_million": "value",
        },
        "cached_input_usd_per_million",
    )
    assert "cached_input_usd_per_million" in data
    assert data["cached_input_usd_per_million"] == "[REDACTED]"
    assert "dynamic-[REDACTED]" in data
    manager.set_minimax_key("valid-length-test-key")
    with pytest.raises(ValueError):
        manager.set_minimax_key("a")
    scan_id = start_free(manager)
    assert wait_complete(manager, scan_id)["status"] == "completed"


def test_two_dashboards_cannot_roll_back_saved_spend_or_progress(tmp_path):
    requests = []
    constructors = []

    class ChargingProvider(FixtureProvider):
        def chat(self, system, user, schema):
            requests.append(self.model)
            return super().chat(system, user, schema)

    class ChargingSession(DiscoverySession):
        def run(self, seeds=()):
            self.provider.chat("fixture", "fixture", None)
            return super().run(seeds)

    def factory(config):
        constructors.append(config["model"])
        return ChargingProvider(config)

    first = make_manager(tmp_path, session_factory=ChargingSession, provider_factory=factory)
    first.set_minimax_key("fixture-key-for-tests")
    scan_id = first.start(
        {"target": "example.test", "provider": "minimax", "options": {"max_candidates": 1}}
    )
    original = wait_complete(first, scan_id)
    second = make_manager(tmp_path, session_factory=ChargingSession, provider_factory=factory)
    second.set_minimax_key("fixture-key-for-tests")
    first.resume(scan_id, {"options": {"max_candidates": 2}, "budget_usd": 6})
    advanced = wait_complete(first, scan_id)
    assert advanced["status"] == "completed", advanced["error"]
    assert float(advanced["summary"]["ledger"]["spent_usd"]) > float(
        original["summary"]["ledger"]["spent_usd"]
    )
    assert advanced["summary"]["attempted"] > original["summary"]["attempted"]
    path = tmp_path / f"{scan_id}.json"
    current_disk = path.read_bytes()
    requests_before_stale_resume = len(requests)
    second.resume(scan_id)
    refused = wait_complete(second, scan_id)
    assert refused["status"] == "error"
    assert "updated in another dashboard" in refused["error"]
    assert refused["can_resume"]
    assert len(requests) == requests_before_stale_resume
    assert len(constructors) == 2
    assert path.read_bytes() == current_disk
    assert refused["summary"]["ledger"] == advanced["summary"]["ledger"]
    assert refused["summary"]["attempted"] == advanced["summary"]["attempted"]
    second.resume(scan_id)
    retried = wait_complete(second, scan_id)
    assert retried["status"] == "completed", retried["error"]
    assert len(requests) == requests_before_stale_resume + 1
    assert retried["summary"]["ledger"]["limit_usd"] == "6"
    assert float(retried["summary"]["ledger"]["spent_usd"]) > float(
        advanced["summary"]["ledger"]["spent_usd"]
    )
