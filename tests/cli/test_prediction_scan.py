"""CLI prediction opt-in, separate spending, and backward-compatible resumes."""

import importlib
import json

import pytest
from click.testing import CliRunner

from enumeraite.core.discovery_session import DiscoverySession
from enumeraite.core.dns_validator import DNSResult
from enumeraite.core.scan_budget import SpendLedger
from enumeraite.providers import jev
from enumeraite.providers.llm_provider import ChatResult

scan = importlib.import_module("enumeraite.cli.scan_command")


class FixturePredictor:
    instances = []

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.closed = False
        self.instances.append(self)

    def score_plans(self, state, plans, *, ledger, checkpoint, **kwargs):
        reservation = ledger.reserve(100, 1)
        checkpoint()
        ledger.settle(reservation, ChatResult("", input_tokens=100, output_tokens=0))
        checkpoint()
        return {plan["id"]: 0.7 for plan in plans}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fixtures(monkeypatch):
    FixturePredictor.instances = []
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "JevPredictor", FixturePredictor)

    class DNS:
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

    monkeypatch.setattr(scan, "ScanDNS", DNS)
    monkeypatch.setattr(
        scan,
        "collect_passive",
        lambda *args, **kwargs: pytest.fail("Seed-first test must not access passive services"),
    )


def invoke(tmp_path, *args):
    return CliRunner().invoke(
        scan.scan_command,
        [
            "ny-01-iphone.example.com",
            "--state",
            str(tmp_path / "scan.json"),
            "--max-candidates",
            "8",
            *args,
        ],
    )


def test_prediction_requires_explicit_choice_even_with_environment_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")
    result = invoke(tmp_path, "--json")
    assert result.exit_code == 0, result.output
    assert not FixturePredictor.instances
    assert json.loads(result.stdout)["prediction_ledger"] is None


def test_jev_uses_environment_key_and_additional_budget_not_generation_budget(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")
    result = invoke(tmp_path, "--predictor", "jev", "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ledger"] is None
    assert report["predictor"] == {"name": "jev", "model": jev.JEV_MODEL}
    assert report["prediction_ledger"]["limit_usd"] == "0.05"
    assert float(report["prediction_ledger"]["spent_usd"]) > 0
    assert report["predictions"]
    assert "additional cumulative cap $0.05" in result.stderr
    assert "separate from the generation budget" in result.stderr
    assert "no paid AI calls" not in result.stderr
    assert "fixture-typesafe-key-never-real" not in result.output
    assert "fixture-typesafe-key-never-real" not in (tmp_path / "scan.json").read_text()
    assert FixturePredictor.instances[0].api_key == "fixture-typesafe-key-never-real"
    assert FixturePredictor.instances[0].closed


@pytest.mark.parametrize(
    "args,expected",
    [
        (("--predictor", "jev"), "TYPESAFE_API_KEY"),
        (("--prediction-budget-usd", "0.05"), "--predictor jev"),
        (("--predictor", "none", "--prediction-budget-usd", "0.05"), "--predictor jev"),
        (("--prediction-budget-usd", "nan"), "finite"),
        (("--prediction-budget-usd", "-1"), "not in the range"),
    ],
)
def test_invalid_or_unconfigured_prediction_fails_without_client(tmp_path, args, expected):
    result = invoke(tmp_path, *args)
    assert result.exit_code != 0
    assert expected in result.stderr
    assert not FixturePredictor.instances


@pytest.mark.parametrize("budget", ["0.08", "1001.0"])
def test_saved_predictor_and_budget_are_preserved_when_flags_omitted(tmp_path, monkeypatch, budget):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")
    initial = invoke(tmp_path, "--predictor", "jev", "--prediction-budget-usd", budget)
    assert initial.exit_code == 0, initial.output
    first = json.loads((tmp_path / "scan.json").read_text())
    result = invoke(tmp_path, "--resume", "--json")
    assert result.exit_code == 0, result.output
    resumed = json.loads(result.stdout)
    assert resumed["prediction_ledger"]["limit_usd"] == budget
    assert resumed["prediction_ledger"]["spent_usd"] == first["prediction_ledger"]["spent_usd"]
    assert len(FixturePredictor.instances) == 2
    assert all(client.closed for client in FixturePredictor.instances)
    assert invoke(tmp_path, "--resume", "--predictor", "none").exit_code != 0


def test_existing_free_checkpoint_cannot_silently_enable_prediction(tmp_path, monkeypatch):
    assert invoke(tmp_path).exit_code == 0
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")
    result = invoke(tmp_path, "--resume", "--predictor", "jev")
    assert result.exit_code != 0
    assert "same prediction provider" in result.stderr
    assert not FixturePredictor.instances


def test_interrupted_prediction_reservation_is_cumulative_and_not_double_charged(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")
    original = SpendLedger(0.05, jev.JEV_PRICE, spent_usd=0.02)
    original.reserve(100, 1)
    saved = {
        "predictor": {"name": "jev", "model": jev.JEV_MODEL},
        "prediction_ledger": original.to_dict(),
    }
    with pytest.raises(ValueError, match="spent or reserved"):
        scan._predictor_and_ledger(None, 0.02, saved)
    client, ledger = scan._predictor_and_ledger(None, 0.07, saved)
    assert ledger.limit_usd == 0.07
    assert ledger.spent_usd > 0.02 and ledger.uncertain_calls == 1
    saved["prediction_ledger"] = ledger.to_dict()
    second, restored = scan._predictor_and_ledger(None, None, saved)
    assert restored.to_dict() == ledger.to_dict()
    client.close()
    second.close()


def test_prediction_client_closes_when_later_generation_validation_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")

    def fail(*args):
        raise ValueError("Invalid generation settings")

    monkeypatch.setattr(scan, "_provider_and_ledger", fail)
    result = invoke(tmp_path, "--predictor", "jev")
    assert result.exit_code != 0
    assert FixturePredictor.instances[0].closed


def test_prediction_constructor_failure_never_echoes_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")

    def fail(api_key, model):
        raise ValueError("Secret " + api_key)

    monkeypatch.setattr(jev, "JevPredictor", fail)
    result = invoke(tmp_path, "--predictor", "jev")
    assert result.exit_code != 0
    assert "Could not initialize Jev" in result.stderr
    assert "fixture-typesafe-key-never-real" not in result.output


def test_prediction_cleanup_failure_is_generic_and_preserves_success(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")

    class FailingClose(FixturePredictor):
        def close(self):
            raise RuntimeError(self.api_key)

    monkeypatch.setattr(jev, "JevPredictor", FailingClose)
    result = invoke(tmp_path, "--predictor", "jev", "--json")
    assert result.exit_code == 0, result.output
    assert "prediction connection cleanup failed" in result.stderr
    assert "fixture-typesafe-key-never-real" not in result.output


def test_raw_prediction_key_is_redacted_from_events_report_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-typesafe-key-never-real")

    class WarningSession(DiscoverySession):
        def run(self, seeds=()):
            self._warn("fixture-typesafe-key-never-real in synthetic warning")
            return super().run(seeds)

    monkeypatch.setattr(scan, "DiscoverySession", WarningSession)
    result = invoke(tmp_path, "--predictor", "jev", "--json")
    assert result.exit_code == 0, result.output
    assert "fixture-typesafe-key-never-real" not in result.output
    assert "fixture-typesafe-key-never-real" not in (tmp_path / "scan.json").read_text()
    assert "REDACTED" in result.stdout
