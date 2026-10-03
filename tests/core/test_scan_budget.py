"""Budget tests use fake responses only; they never call a paid service."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from enumeraite.core.scan_budget import (
    BudgetExhausted,
    ModelPrice,
    ScanPaused,
    SpendLedger,
    attach_scan_guard,
    price_for,
)
from enumeraite.providers.anthropic_provider import AnthropicProvider
from enumeraite.providers.llm_provider import ChatResult, LLMProvider
from enumeraite.providers.openai_compatible import OpenAICompatibleProvider


class FakeProvider(LLMProvider):
    def __init__(self, response=None):
        super().__init__({"model": "fake", "max_tokens": 100, "timeout": 30})
        self.response = response or ChatResult("[]", input_tokens=10, output_tokens=5)
        self.request_limits = []

    def chat(self, system, user, schema):
        self.request_limits.append((self.max_tokens, self.request_timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def get_provider_name(self):
        return "fake"


def test_unknown_prices_fail_closed_and_partial_overrides_fail():
    with pytest.raises(ValueError, match="no verified scan price"):
        price_for("openrouter", "future-model")
    with pytest.raises(ValueError, match="both"):
        price_for("openrouter", "future-model", input_price=1)
    assert price_for("openrouter", "future-model", 1, 2) == ModelPrice(1, 2)


def test_minimax_registry_has_current_dated_conservative_prices():
    m3 = price_for("openai-compat:minimax", "MiniMax-M3")
    assert m3.input_usd_per_million == 0.6
    assert m3.output_usd_per_million == 2.4
    assert m3.checked_at == "2026-10-02"
    assert m3.source == "https://platform.minimax.io/docs/pricing/overview"
    assert price_for("minimax", "MiniMax-M2.7").input_usd_per_million == 0.375


@pytest.mark.parametrize(
    "provider,model,input_price,output_price,cache_price,source",
    [
        (
            "openai",
            "gpt-4.1-mini",
            0.40,
            1.60,
            0.10,
            "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
        ),
        (
            "openai",
            "gpt-4.1",
            2.0,
            8.0,
            0.50,
            "https://developers.openai.com/api/docs/models/gpt-4.1",
        ),
        (
            "anthropic",
            "claude-sonnet-4-6",
            6.0,
            15.0,
            0.30,
            "https://platform.claude.com/docs/en/models/sonnet-4-6/overview",
        ),
    ],
)
def test_dashboard_registry_prices_have_dated_official_provenance(
    provider, model, input_price, output_price, cache_price, source
):
    price = price_for(provider, model)
    assert price.input_usd_per_million == input_price
    assert price.output_usd_per_million == output_price
    assert price.cached_input_usd_per_million == cache_price
    assert price.source == source
    assert price.checked_at == "2026-10-02"
    if provider == "openai":
        assert price_for("openai-compat:openai", model) == price


@pytest.mark.parametrize(
    "provider_name,model,endpoint",
    [
        ("openai", "gpt-4.1-mini", "https://api.openai.com/v1"),
        ("openai", "gpt-4.1", "https://api.openai.com/v1/"),
        ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com"),
        ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com/"),
    ],
)
@pytest.mark.parametrize("source", ["config", "sdk_client"])
def test_dashboard_registry_accepts_official_endpoint_from_config_or_client(
    provider_name, model, endpoint, source
):
    provider = FakeProvider()
    if source == "config":
        provider.config["base_url"] = endpoint
    else:
        provider.config["base_url"] = ""
        provider.client = SimpleNamespace(base_url=endpoint)
    ledger = SpendLedger(1, price_for(provider_name, model))
    attach_scan_guard(provider, ledger, None)
    provider.chat("s", "u", None)
    assert len(provider.request_limits) == 1
    assert ledger.calls == 1


@pytest.mark.parametrize(
    "provider_name,model,endpoint",
    [
        ("openai", "gpt-4.1-mini", "https://proxy.example.test/v1"),
        ("openai", "gpt-4.1", "https://api.openai.com.evil.test/v1"),
        ("openai", "gpt-4.1-mini", "http://api.openai.com/v1"),
        ("openai", "gpt-4.1", "https://api.openai.com/v1?route=custom"),
        ("anthropic", "claude-sonnet-4-6", "https://proxy.example.test"),
        ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com.evil.test"),
        ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com/v1"),
        ("anthropic", "claude-sonnet-4-6", "http://api.anthropic.com"),
    ],
)
@pytest.mark.parametrize("source", ["config", "sdk_client"])
def test_dashboard_registry_rejects_changed_endpoint_before_any_reservation(
    provider_name, model, endpoint, source
):
    provider = FakeProvider()
    if source == "config":
        provider.config["base_url"] = endpoint
    else:
        provider.config["base_url"] = ""
        provider.client = SimpleNamespace(base_url=endpoint)
    ledger = SpendLedger(1, price_for(provider_name, model))
    with pytest.raises(ValueError, match="endpoints require explicit"):
        attach_scan_guard(provider, ledger, None)
    assert not provider._scan_guard_attached
    assert provider.request_limits == []
    assert ledger.calls == ledger.spent_usd == ledger.reserved_usd == 0


@pytest.mark.parametrize(
    "provider_name,model", [("openai", "gpt-4.1-mini"), ("anthropic", "claude-sonnet-4-6")]
)
def test_dashboard_registry_refuses_missing_endpoint_identity(provider_name, model):
    provider = FakeProvider()
    with pytest.raises(ValueError, match="endpoints require explicit"):
        attach_scan_guard(provider, SpendLedger(1, price_for(provider_name, model)), None)
    assert provider.request_limits == []


def test_explicit_price_override_still_allows_custom_endpoint():
    provider = FakeProvider()
    provider.config["base_url"] = "https://proxy.example.test/v1"
    ledger = SpendLedger(1, price_for("openai", "gpt-4.1", 3.0, 9.0))
    attach_scan_guard(provider, ledger, None)
    provider.chat("s", "u", None)
    assert ledger.calls == 1


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_prices_and_budgets_reject_invalid_amounts(value):
    with pytest.raises(ValueError):
        ModelPrice(value, 1)
    with pytest.raises(ValueError):
        SpendLedger(value, ModelPrice(1, 2))


def test_reservation_reduces_output_limit_before_persisting_and_calling():
    provider = FakeProvider()
    # A 1038-token conservative prompt estimate at $1/M leaves 62 output tokens.
    ledger = SpendLedger(0.0011, ModelPrice(1, 1))
    snapshots = []

    def save():
        snapshots.append((ledger.to_dict(), len(provider.request_limits)))

    attach_scan_guard(provider, ledger, deadline=None, on_change=save)
    provider.chat("s", "u", None)
    assert 1 <= provider.request_limits[0][0] < 100
    assert snapshots[0][1] == 0
    assert float(snapshots[0][0]["reserved_usd"]) > 0
    assert snapshots[1][1] == 1
    assert ledger.spent_usd == pytest.approx(0.000015)
    assert ledger.reserved_usd == 0
    assert provider.max_tokens == 100


def test_unaffordable_prompt_never_starts_request():
    provider = FakeProvider()
    ledger = SpendLedger(0.00001, ModelPrice(1, 1))
    attach_scan_guard(provider, ledger, None)
    with pytest.raises(BudgetExhausted, match="prompt"):
        provider.chat("prompt", "user", None)
    assert provider.request_limits == []
    assert ledger.calls == 0


def test_pause_prevents_new_paid_call_without_reserving():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 1))
    attach_scan_guard(provider, ledger, None, stop_requested=lambda: True)
    with pytest.raises(ScanPaused):
        provider.chat("s", "u", None)
    assert provider.request_limits == []
    assert ledger.spent_usd == ledger.reserved_usd == ledger.calls == 0


def test_pause_during_checkpoint_releases_reservation_durably():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 1))
    snapshots = []
    attach_scan_guard(
        provider,
        ledger,
        None,
        lambda: snapshots.append(ledger.to_dict()),
        stop_requested=lambda: bool(snapshots),
    )
    with pytest.raises(ScanPaused):
        provider.chat("s", "u", None)
    assert provider.request_limits == []
    assert float(snapshots[0]["reserved_usd"]) > 0
    assert float(snapshots[-1]["reserved_usd"]) == 0
    restored = SpendLedger.from_dict(snapshots[-1])
    assert restored.spent_usd == restored.uncertain_calls == 0


def test_pause_between_generation_chat_calls_preserves_only_actual_spend():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 1))
    stopped = False
    attach_scan_guard(provider, ledger, None, stop_requested=lambda: stopped)
    provider.chat("s", "u", None)
    spent = ledger.spent_usd
    stopped = True
    with pytest.raises(ScanPaused):
        provider.chat("s", "u", None)
    assert len(provider.request_limits) == 1
    assert ledger.calls == 1 and ledger.spent_usd == spent


@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("network error"),
        KeyboardInterrupt(),
        ChatResult("[]"),
        ChatResult("[]", input_tokens=10, output_tokens=5, usage_complete=False),
        ChatResult("[]", input_tokens=10, output_tokens=5, reasoning_tokens=20),
        ChatResult("[]", input_tokens=10, output_tokens=0),
        ChatResult("[]", input_tokens=10, output_tokens=5, cached_input_tokens=100),
    ],
)
def test_unknown_request_outcomes_charge_full_reservation(response):
    provider = FakeProvider(response)
    ledger = SpendLedger(1, ModelPrice(1, 1))
    reserved = []
    attach_scan_guard(provider, ledger, None, lambda: reserved.append(ledger.reserved_usd))
    if isinstance(response, BaseException):
        with pytest.raises(type(response)):
            provider.chat("s", "u", None)
    else:
        provider.chat("s", "u", None)
    assert ledger.spent_usd == reserved[0]
    assert ledger.uncertain_calls == 1
    assert ledger.reserved_usd == 0


def test_cache_credit_requires_known_price_and_valid_counts():
    response = ChatResult("[]", input_tokens=100, cached_input_tokens=80, output_tokens=10)
    priced = SpendLedger(1, ModelPrice(1, 2, 0.2))
    priced.settle(priced.reserve(200, 20), response)
    assert priced.spent_usd == pytest.approx(0.000056)
    unpriced = SpendLedger(1, ModelPrice(1, 2))
    unpriced.settle(unpriced.reserve(200, 20), response)
    assert unpriced.spent_usd == pytest.approx(0.00012)


def test_reported_overspend_is_not_hidden_and_blocks_more_requests():
    provider = FakeProvider(ChatResult("[]", input_tokens=5000, output_tokens=200))
    ledger = SpendLedger(0.002, ModelPrice(1, 1))
    snapshots = []
    attach_scan_guard(provider, ledger, None, lambda: snapshots.append(ledger.to_dict()))
    with pytest.raises(BudgetExhausted, match="exceeded"):
        provider.chat("s", "u", None)
    assert ledger.spent_usd == pytest.approx(0.0052)
    assert ledger.remaining_usd == 0
    assert snapshots[-1]["blocked"] is True
    with pytest.raises(BudgetExhausted):
        provider.chat("s", "u", None)
    assert len(provider.request_limits) == 1


def test_resume_accounts_reserved_charge_once():
    ledger = SpendLedger(1, ModelPrice(1, 2))
    reservation = ledger.reserve(100, 100)
    restored = SpendLedger.from_dict(ledger.to_dict())
    assert restored.spent_usd == float(reservation.amount)
    assert restored.reserved_usd == 0
    assert restored.uncertain_calls == 1
    again = SpendLedger.from_dict(restored.to_dict())
    assert again.to_dict() == restored.to_dict()
    assert again.remaining_usd == pytest.approx(0.9997)


@pytest.mark.parametrize("field", ["limit_usd", "spent_usd", "reserved_usd"])
def test_resume_rejects_nonfinite_or_negative_amounts(field):
    data = SpendLedger(1, ModelPrice(1, 2)).to_dict()
    for invalid in ("NaN", "-1", "invalid", True):
        data[field] = invalid
        with pytest.raises(ValueError):
            SpendLedger.from_dict(data)


def test_persistence_failure_aborts_before_any_network_call():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, None, Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        provider.chat("s", "u", None)
    assert provider.request_limits == []
    assert ledger.spent_usd == ledger.reserved_usd == 0
    assert ledger.calls == 0


def test_expired_deadline_never_reserves_or_sends():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, 10)
    with patch("enumeraite.core.scan_budget.time.monotonic", return_value=11):
        with pytest.raises(TimeoutError):
            provider.chat("s", "u", None)
    assert provider.request_limits == []
    assert ledger.calls == 0


def test_deadline_rechecked_after_persisting_reservation():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, 10, lambda: None)
    with patch("enumeraite.core.scan_budget.time.monotonic", side_effect=[9, 11]):
        with pytest.raises(TimeoutError):
            provider.chat("s", "u", None)
    assert provider.request_limits == []
    assert ledger.reserved_usd == 0


def test_each_call_gets_remaining_deadline_and_late_response_is_charged():
    provider = FakeProvider()
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, 20)
    with patch("enumeraite.core.scan_budget.time.monotonic", side_effect=[12, 13, 21]):
        with pytest.raises(TimeoutError, match="during"):
            provider.chat("s", "u", None)
    assert provider.request_limits[0][1] == 7
    assert ledger.spent_usd == pytest.approx(0.00002)
    assert provider.request_timeout == 30


@patch("enumeraite.providers.openai_compatible.openai.OpenAI")
def test_scan_disables_sdk_retries_and_extension_fallback(mock_openai):
    client = mock_openai.return_value
    bounded = client.with_options.return_value
    bounded.chat.completions.create.side_effect = RuntimeError("unsupported response_format")
    provider = OpenAICompatibleProvider({"api_key": "not-a-key"})
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, None)
    with pytest.raises(RuntimeError):
        provider.chat("s", "u", {"type": "object"})
    assert client.with_options.call_args.kwargs["max_retries"] == 0
    assert bounded.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_count == 0
    assert ledger.uncertain_calls == 1


@patch("enumeraite.providers.openai_compatible.openai.OpenAI")
def test_openai_missing_usage_is_not_treated_as_free(mock_openai):
    client = mock_openai.return_value.with_options.return_value
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))], usage=None
    )
    provider = OpenAICompatibleProvider({"api_key": "not-a-key"})
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, None)
    result = provider.chat("s", "u", None)
    assert not result.usage_complete
    assert ledger.spent_usd > 0
    assert ledger.uncertain_calls == 1


@patch("enumeraite.providers.anthropic_provider.anthropic.Anthropic")
def test_anthropic_disables_retries_and_includes_all_input_categories(mock_anthropic):
    client = mock_anthropic.return_value.with_options.return_value
    client.messages.create.return_value = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(
            input_tokens=10,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=10,
            output_tokens=5,
        ),
    )
    provider = AnthropicProvider({"api_key": "not-a-key"})
    ledger = SpendLedger(1, ModelPrice(1, 2))
    attach_scan_guard(provider, ledger, None)
    result = provider.chat("s", "u", None)
    assert result.input_tokens == 100
    assert result.cached_input_tokens == 80
    assert ledger.spent_usd == pytest.approx(0.00011)
    assert mock_anthropic.return_value.with_options.call_args.kwargs["max_retries"] == 0


@patch("enumeraite.providers.openai_compatible.openai.OpenAI")
def test_registry_prices_reject_custom_endpoints(mock_openai):
    provider = OpenAICompatibleProvider(
        {"preset": "minimax", "api_key": "not-a-key", "base_url": "https://example.test/v1"}
    )
    with pytest.raises(ValueError, match="custom MiniMax"):
        attach_scan_guard(provider, SpendLedger(1, price_for("minimax", provider.model)), None)
    remote_local = OpenAICompatibleProvider(
        {"preset": "ollama", "base_url": "https://example.test/v1"}
    )
    with pytest.raises(ValueError, match="remote local-model"):
        attach_scan_guard(remote_local, SpendLedger(1, price_for("ollama", "any")), None)


def test_zero_cost_local_model_can_run_under_zero_dollar_cap():
    ledger = SpendLedger(0, ModelPrice(0, 0))
    reservation = ledger.reserve(1000, 500)
    ledger.settle(reservation, ChatResult("[]", input_tokens=10, output_tokens=5))
    assert ledger.spent_usd == 0
    assert ledger.calls == 1
