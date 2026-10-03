"""Generation request profiles must remain inside the existing spending guard."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from enumeraite.core.scan_budget import (
    BudgetExhausted,
    ScanPaused,
    SpendLedger,
    attach_scan_guard,
    price_for,
)
from enumeraite.core.strategies import SubdomainDiscoveryStrategy
from enumeraite.providers.openai_compatible import OpenAICompatibleProvider


def guarded_provider(
    monkeypatch, *, limit=1, response_error=None, on_change=None, stop_requested=None, deadline=None
):
    base_client, bounded_client = Mock(), Mock()
    base_client.with_options.return_value = bounded_client
    bounded_client.chat.completions.create.return_value = SimpleNamespace(
        model="MiniMax-M3",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"candidates":[{"value":"new.example.test"}]}'),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    if response_error:
        bounded_client.chat.completions.create.side_effect = response_error
    monkeypatch.setattr(
        "enumeraite.providers.openai_compatible.openai.OpenAI", Mock(return_value=base_client)
    )
    provider = OpenAICompatibleProvider(
        {"preset": "minimax", "api_key": "test", "max_batch_calls": 1}
    )
    ledger = SpendLedger(limit, price_for("minimax", provider.model))
    attach_scan_guard(
        provider, ledger, deadline=deadline, on_change=on_change, stop_requested=stop_requested
    )
    return provider, ledger, base_client, bounded_client


def generate(provider):
    return provider.generate_paths(
        ["known.example.test"], "example.test", 1, strategy=SubdomainDiscoveryStrategy()
    )


def test_profile_calls_guarded_public_chat_once_and_settles_spend(monkeypatch):
    snapshots = []
    provider, ledger, base, bounded = guarded_provider(
        monkeypatch,
        on_change=lambda: snapshots.append(
            (provider._runtime_generation_kind, ledger.reserved_usd)
        ),
    )
    saved = provider.max_tokens, provider.request_timeout
    result = generate(provider)
    assert result.paths == ["new.example.test"]
    assert result.metadata["requested_thinking_modes"] == ["disabled"]
    assert ledger.calls == 1 and ledger.uncertain_calls == 0
    assert ledger.spent_usd > 0 and ledger.reserved_usd == 0
    assert snapshots[0][0] == "subdomain" and snapshots[0][1] > 0
    assert provider._runtime_generation_kind is None
    assert (provider.max_tokens, provider.request_timeout) == saved
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()
    assert base.with_options.call_args.kwargs["max_retries"] == 0
    assert bounded.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True,
        "thinking": {"type": "disabled"},
    }


@pytest.mark.parametrize("failure", ["budget", "pause", "save"])
def test_pre_request_guard_failure_restores_kind_and_sends_nothing(monkeypatch, failure):
    provider, ledger, base, bounded = guarded_provider(
        monkeypatch,
        limit=0 if failure == "budget" else 1,
        stop_requested=(lambda: True) if failure == "pause" else None,
        on_change=Mock(side_effect=OSError("disk full")) if failure == "save" else None,
    )
    expected = {"budget": BudgetExhausted, "pause": ScanPaused, "save": OSError}[failure]
    with pytest.raises(expected):
        generate(provider)
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_not_called()
    assert provider._runtime_generation_kind is None
    assert ledger.spent_usd == ledger.reserved_usd == 0


@pytest.mark.parametrize(
    "error", [RuntimeError("unsupported parameter: thinking"), KeyboardInterrupt()]
)
def test_guarded_backend_failure_reserves_once_never_bypasses_or_retries(monkeypatch, error):
    provider, ledger, base, bounded = guarded_provider(monkeypatch, response_error=error)
    saved = provider.max_tokens, provider.request_timeout
    with pytest.raises(type(error)):
        generate(provider)
    assert ledger.calls == ledger.uncertain_calls == 1
    assert ledger.spent_usd > 0 and ledger.reserved_usd == 0
    assert provider._runtime_generation_kind is None
    assert (provider.max_tokens, provider.request_timeout) == saved
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()


def test_direct_chat_after_guarded_generation_has_no_leaked_profile(monkeypatch):
    provider, ledger, _, bounded = guarded_provider(monkeypatch)
    generate(provider)
    provider.chat("ordinary", "chat", None)
    assert ledger.calls == 2
    assert bounded.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True
    }
    assert provider._runtime_generation_kind is None


def test_post_request_deadline_failure_restores_kind_and_accounts_actual_spend(monkeypatch):
    now = [0]
    monkeypatch.setattr("enumeraite.core.scan_budget.time.monotonic", lambda: now[0])
    provider, ledger, base, bounded = guarded_provider(monkeypatch, deadline=1)
    response = bounded.chat.completions.create.return_value

    def complete(**kwargs):
        now[0] = 2
        return response

    bounded.chat.completions.create.side_effect = complete
    with pytest.raises(TimeoutError, match="during the model request"):
        generate(provider)
    assert provider._runtime_generation_kind is None
    assert ledger.calls == 1 and ledger.uncertain_calls == 0
    assert ledger.spent_usd > 0 and ledger.reserved_usd == 0
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()
