"""Independent opt-in Jev credentials and cumulative budgets in the dashboard."""

import json
import threading
import time

import pytest

from enumeraite.core.discovery_session import DiscoverySession
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.scan_budget import SpendLedger
from enumeraite.providers.jev import JEV_MODEL, JEV_PRICE
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.web.manager import ScanManager


class FixtureDNS:
    def __init__(self, *args, **kwargs):
        pass

    def validate(self, values, **kwargs):
        return [
            DNSResult(
                value,
                "-01-" in value or "-02-" in value,
                ["192.0.2.1"] if "-01-" in value or "-02-" in value else [],
                0,
            )
            for value in values
        ]


class FixturePredictor:
    instances = []

    def __init__(self, api_key, model=JEV_MODEL):
        self.key = api_key
        self.model = model
        self.closed = False
        self.calls = 0
        self.instances.append(self)

    def score_plans(self, state, plans, *, ledger, checkpoint, **kwargs):
        reservation = ledger.reserve(100, 1)
        checkpoint()
        self.calls += 1
        ledger.settle(reservation, ChatResult("", input_tokens=100, output_tokens=0))
        checkpoint()
        return {plan["id"]: 0.8 for plan in plans}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fixtures():
    FixturePredictor.instances = []


def manager(tmp_path, **kwargs):
    return ScanManager(
        tmp_path, dns_factory=FixtureDNS, predictor_factory=FixturePredictor, **kwargs
    )


def wait(manager, scan_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = manager.snapshot(scan_id)
        if result["status"] not in {"running", "pausing"}:
            manager._records[scan_id].thread.join(1)
            return manager.snapshot(scan_id)
        time.sleep(0.005)
    pytest.fail("Fixture scan did not finish")


def start(manager, **kwargs):
    return manager.start(
        {
            "target": "ny-01-iphone.example.com",
            "prediction_provider": "jev",
            "options": {"max_candidates": 8},
            **kwargs,
        }
    )


def test_prediction_is_opt_in_with_independent_memory_only_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "environment-key-must-not-enable-ui")
    scans = manager(tmp_path)
    assert not scans.list_state()["credentials"]["jev"]
    with pytest.raises(ValueError, match="API key"):
        start(scans)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    free_id = scans.start({"target": "ny-01-iphone.example.com", "options": {"max_candidates": 2}})
    free = wait(scans, free_id)
    assert free["summary"]["prediction_ledger"] is None
    assert not FixturePredictor.instances
    scan_id = start(scans)
    result = wait(scans, scan_id)
    assert result["status"] == "completed", result["error"]
    assert result["summary"]["ledger"] is None
    assert result["summary"]["predictor"] == {"name": "jev", "model": JEV_MODEL}
    assert result["summary"]["prediction_ledger"]["limit_usd"] == "0.05"
    assert float(result["summary"]["prediction_ledger"]["spent_usd"]) > 0
    assert result["summary"]["predictions"]
    assert FixturePredictor.instances[0].closed
    assert "fixture-jev-key-never-real" not in json.dumps(result)
    assert "fixture-jev-key-never-real" not in (tmp_path / f"{scan_id}.json").read_text()
    assert scans.report(scan_id)["prediction_ledger"] == result["summary"]["prediction_ledger"]


def test_generation_and_prediction_have_distinct_ledgers(tmp_path):
    class Generator(LLMProvider):
        def get_provider_name(self):
            return "openai-compat:minimax"

        def chat(self, system, user, schema):
            return ChatResult('{"candidates":[]}', input_tokens=20, output_tokens=5)

    scans = manager(tmp_path, provider_factory=Generator)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    scans.set_api_key("minimax", "fixture-minimax-key-never-real")
    scan_id = start(scans, provider="minimax", budget_usd=5, prediction_budget_usd=0.07)
    result = wait(scans, scan_id)
    assert result["status"] == "completed", result["error"]
    assert result["summary"]["ledger"]["limit_usd"] == "5"
    assert result["summary"]["prediction_ledger"]["limit_usd"] == "0.07"
    assert (
        result["summary"]["ledger"]["pricing"] != result["summary"]["prediction_ledger"]["pricing"]
    )
    assert result["summary"]["ledger"]["calls"] >= 1
    assert result["summary"]["prediction_ledger"]["calls"] >= 1


@pytest.mark.parametrize(
    "payload",
    [
        {"prediction_provider": "other"},
        {"prediction_provider": []},
        {"prediction_provider": "none", "prediction_budget_usd": 1},
        {"prediction_budget_usd": -1},
        {"prediction_budget_usd": True},
        {"prediction_budget_usd": float("nan")},
        {"prediction_budget_usd": 1001},
        {"provider": "jev"},
    ],
)
def test_invalid_prediction_settings_rejected_before_launch(tmp_path, payload):
    scans = manager(tmp_path)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    with pytest.raises(ValueError):
        start(scans, **payload)
    assert scans.list_state()["scans"] == []


def test_resume_preserves_predictor_and_charges_interrupted_reservation_once(tmp_path):
    scans = manager(tmp_path)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    scan_id = start(scans)
    wait(scans, scan_id)
    path = tmp_path / f"{scan_id}.json"
    saved = json.loads(path.read_text())
    ledger = SpendLedger(0.05, JEV_PRICE, spent_usd=0.02)
    ledger.reserve(100, 1)
    saved["prediction_ledger"] = ledger.to_dict()
    path.write_text(json.dumps(saved))
    restored = manager(tmp_path)
    assert restored.snapshot(scan_id)["summary"]["predictor"] == {"name": "jev", "model": JEV_MODEL}
    assert not restored.list_state()["credentials"]["jev"]
    with pytest.raises(ValueError, match="API key"):
        restored.resume(scan_id)
    restored.set_api_key("jev", "fixture-replacement-jev-key")
    with pytest.raises(ValueError, match="same prediction provider"):
        restored.resume(scan_id, {"prediction_provider": "none"})
    with pytest.raises(ValueError, match="spent or reserved"):
        restored.resume(scan_id, {"prediction_budget_usd": 0.02})
    restored.resume(scan_id, {"prediction_budget_usd": 0.08})
    result = wait(restored, scan_id)
    assert result["status"] == "completed", result["error"]
    budget = result["summary"]["prediction_ledger"]
    assert budget["limit_usd"] == "0.08" and float(budget["spent_usd"]) > 0.02
    assert budget["uncertain_calls"] == 1
    restored.resume(scan_id)
    second = wait(restored, scan_id)
    assert second["summary"]["prediction_ledger"]["spent_usd"] == budget["spent_usd"]


def test_old_checkpoints_resume_without_prediction_or_new_optional_fields(tmp_path):
    scans = manager(tmp_path)
    scan_id = scans.start({"target": "ny-01-iphone.example.com", "options": {"max_candidates": 2}})
    wait(scans, scan_id)
    path = tmp_path / f"{scan_id}.json"
    saved = json.loads(path.read_text())
    for field in ("predictor", "prediction_ledger", "predictions"):
        saved.pop(field, None)
    path.write_text(json.dumps(saved))
    restored = manager(tmp_path)
    restored.set_api_key("jev", "fixture-jev-key-never-real")
    with pytest.raises(ValueError, match="same prediction provider"):
        restored.resume(scan_id, {"prediction_provider": "jev"})
    restored.resume(scan_id)
    result = wait(restored, scan_id)
    assert result["status"] == "completed", result["error"]
    assert result["summary"]["predictor"] is None
    assert result["summary"]["prediction_ledger"] is None
    assert not FixturePredictor.instances


def test_stale_prediction_checkpoint_is_not_overwritten_or_spent(tmp_path):
    first = manager(tmp_path)
    first.set_api_key("jev", "fixture-jev-key-never-real")
    scan_id = start(first)
    wait(first, scan_id)
    second = manager(tmp_path)
    second.set_api_key("jev", "fixture-jev-key-never-real")
    first.resume(scan_id, {"prediction_budget_usd": 0.09})
    advanced = wait(first, scan_id)
    path = tmp_path / f"{scan_id}.json"
    saved = path.read_bytes()
    instances = len(FixturePredictor.instances)
    second.resume(scan_id)
    refused = wait(second, scan_id)
    assert refused["status"] == "error"
    assert "updated in another dashboard" in refused["error"]
    assert path.read_bytes() == saved
    assert len(FixturePredictor.instances) == instances
    assert refused["summary"]["prediction_ledger"] == advanced["summary"]["prediction_ledger"]


def test_captured_prediction_key_stays_redacted_after_clearing(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class WarningSession(DiscoverySession):
        def run(self, seeds=()):
            entered.set()
            release.wait(2)
            self._warn("fixture-jev-key-never-real appears in a synthetic warning")
            return super().run(seeds)

    scans = manager(tmp_path, session_factory=WarningSession)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    scan_id = start(scans)
    assert entered.wait(1)
    scans.set_api_key("jev", "")
    release.set()
    result = wait(scans, scan_id)
    assert result["status"] == "completed", result["error"]
    assert "fixture-jev-key-never-real" not in json.dumps(result)
    assert "fixture-jev-key-never-real" not in (tmp_path / f"{scan_id}.json").read_text()


def test_prediction_cleanup_failure_does_not_leak_or_change_completed_scan(tmp_path):
    class FailingClose(FixturePredictor):
        def close(self):
            raise RuntimeError(self.key)

    scans = ScanManager(tmp_path, dns_factory=FixtureDNS, predictor_factory=FailingClose)
    scans.set_api_key("jev", "fixture-jev-key-never-real")
    scan_id = start(scans)
    result = wait(scans, scan_id)
    assert result["status"] == "completed", result["error"]
    assert any(
        "Prediction connection cleanup failed" == event["message"] for event in result["events"]
    )
    assert "fixture-jev-key-never-real" not in json.dumps(result)
