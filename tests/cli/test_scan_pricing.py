"""Model-selection and cumulative-price resume boundaries, without API calls."""

import importlib
from dataclasses import replace
from unittest.mock import Mock

import pytest

from enumeraite.core.config import Config
from enumeraite.core.scan_budget import ModelPrice, SpendLedger, price_for
from enumeraite.providers.llm_provider import LLMProvider

scan_module = importlib.import_module("enumeraite.cli.scan_command")


class FakeProvider(LLMProvider):
    def __init__(self, name="openai", model="custom-model"):
        super().__init__({"model": model})
        self.name = name

    def chat(self, system, user, schema):
        raise AssertionError("provider selection must not make a model request")

    def get_provider_name(self):
        return self.name


@pytest.fixture
def providers(monkeypatch):
    factory = Mock()
    factory.create_provider.return_value = FakeProvider()
    config_loader = Mock(return_value=Config())
    monkeypatch.setattr(scan_module, "load_config", config_loader)
    monkeypatch.setattr(scan_module, "ProviderFactory", Mock(return_value=factory))
    monkeypatch.setattr(scan_module, "detect_provider", lambda config: "openai")
    return factory, config_loader


def checkpoint(ledger, name="openai", model="custom-model"):
    return {"provider": {"name": name, "model": model}, "ledger": ledger.to_dict()}


def select(saved, name="auto", model=None, budget=None, input_price=None, output_price=None):
    return scan_module._provider_and_ledger(
        name, model, budget, input_price, output_price, saved, Mock()
    )


def test_custom_model_resume_reuses_saved_prices_and_original_model(providers):
    factory, _ = providers
    prior = SpendLedger(5, ModelPrice(0.7, 2.1), spent_usd=0.5)
    ai, restored = select(checkpoint(prior))
    assert ai is factory.create_provider.return_value
    assert restored.to_dict() == prior.to_dict()
    assert factory.create_provider.call_args.args[1]["model"] == "custom-model"


def test_builtin_resume_preserves_dated_pricing_snapshot(providers):
    factory, _ = providers
    factory.create_provider.return_value = FakeProvider("openai-compat:minimax", "MiniMax-M3")
    old_price = replace(price_for("minimax", "MiniMax-M3"), checked_at="2026-09-01")
    prior = SpendLedger(5, old_price, spent_usd=0.5)
    _, restored = select(checkpoint(prior, "openai-compat:minimax", "MiniMax-M3"))
    assert restored.pricing == old_price
    assert restored.spent_usd == 0.5


def test_matching_explicit_rates_do_not_fail_on_provenance_difference(providers):
    factory, _ = providers
    factory.create_provider.return_value = FakeProvider("openai-compat:minimax", "MiniMax-M3")
    prior = SpendLedger(5, price_for("minimax", "MiniMax-M3"), spent_usd=0.5)
    _, restored = select(
        checkpoint(prior, "openai-compat:minimax", "MiniMax-M3"), input_price=0.6, output_price=2.4
    )
    assert restored.pricing == ModelPrice(0.6, 2.4)
    assert restored.spent_usd == 0.5


def test_explicit_new_prices_preserve_spent_and_interrupted_reservation(providers):
    prior = SpendLedger(5, ModelPrice(1, 2), spent_usd=0.5, reserved_usd=0.1, calls=1)
    _, restored = select(checkpoint(prior), budget=6, input_price=3, output_price=4)
    assert restored.spent_usd == 0.6
    assert restored.reserved_usd == 0
    assert restored.uncertain_calls == 1
    assert restored.limit_usd == 6
    assert restored.pricing == ModelPrice(3, 4)
    assert float(restored.reserve(1000, 1000).amount) == pytest.approx(0.007)


def test_free_resume_keeps_cumulative_spending_and_can_change_cap(providers):
    _, config_loader = providers
    prior = SpendLedger(5, ModelPrice(1, 2), spent_usd=0.5, reserved_usd=0.1)
    ai, restored = select(checkpoint(prior), name="none", budget=2)
    assert ai is None
    assert restored.spent_usd == 0.6
    assert restored.limit_usd == 2
    assert restored.pricing == prior.pricing
    config_loader.assert_not_called()


@pytest.mark.parametrize("name", ["none", "auto", "openai"])
def test_resumed_budget_cannot_fall_below_spent_plus_uncertain_requests(providers, name):
    _, config_loader = providers
    prior = SpendLedger(5, ModelPrice(1, 2), spent_usd=0.5, reserved_usd=0.1)
    with pytest.raises(ValueError, match="already spent"):
        select(checkpoint(prior), name=name, budget=0.55)
    config_loader.assert_not_called()


@pytest.mark.parametrize("name,model", [("minimax", None), ("auto", "different-model")])
def test_provider_and_model_changes_rejected_before_loading_config(providers, name, model):
    factory, config_loader = providers
    prior = SpendLedger(5, ModelPrice(1, 2))
    with pytest.raises(ValueError, match="same provider and model"):
        select(checkpoint(prior), name=name, model=model)
    config_loader.assert_not_called()
    factory.create_provider.assert_not_called()


def test_factory_identity_must_match_checkpoint(providers):
    factory, _ = providers
    factory.create_provider.return_value = FakeProvider(model="unexpected-model")
    prior = SpendLedger(5, ModelPrice(1, 2))
    with pytest.raises(ValueError, match="same provider and model"):
        select(checkpoint(prior))


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1, True])
def test_invalid_budget_rejected_even_when_no_provider_is_selected(providers, budget):
    _, config_loader = providers
    with pytest.raises(ValueError, match="finite nonnegative"):
        select(None, name="none", budget=budget)
    config_loader.assert_not_called()


def test_incomplete_price_override_rejected_instead_of_silent_fallback(providers):
    with pytest.raises(ValueError, match="both"):
        select(None, input_price=1)


def test_new_unknown_model_still_requires_explicit_prices(providers):
    with pytest.raises(ValueError, match="no verified scan price"):
        select(None, name="openai")
    _, ledger = select(None, name="openai", input_price=1, output_price=2)
    assert ledger.pricing == ModelPrice(1, 2)
    assert ledger.limit_usd == 1
