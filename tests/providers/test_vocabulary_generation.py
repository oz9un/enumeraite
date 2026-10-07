"""Vocabulary hypotheses use the ordinary guarded provider transport, never DNS."""

import json
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
from enumeraite.providers.anthropic_provider import AnthropicProvider
from enumeraite.providers.openai_compatible import OpenAICompatibleProvider

TARGET = "example.test"
KNOWN = ("api.dev.example.test", "api.prod.example.test")


@pytest.fixture
def layout():
    from enumeraite.core.discovery_vocabulary import build_vocabulary_layouts

    return next(
        item
        for item in build_vocabulary_layouts(KNOWN, TARGET)
        if set(item.observed_values) == {"dev", "prod"}
    )


def additions(layout, *values):
    return json.dumps({"additions": [{"plan_id": layout.plan_id, "values": list(values)}]})


def completion(raw, *, usage=True):
    return SimpleNamespace(
        model="MiniMax-M3",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=raw), finish_reason="stop", logprobs=None
            )
        ],
        usage=(
            SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            )
            if usage
            else None
        ),
    )


def make_provider(monkeypatch, raw, **config):
    base, bounded = Mock(), Mock()
    base.with_options.return_value = bounded
    base.chat.completions.create.return_value = completion(raw)
    bounded.chat.completions.create.return_value = completion(raw)
    monkeypatch.setattr(
        "enumeraite.providers.openai_compatible.openai.OpenAI", Mock(return_value=base)
    )
    provider = OpenAICompatibleProvider(
        {"preset": "minimax", "api_key": "test-only", "max_batch_calls": 9, **config}
    )
    return provider, base, bounded


def guard(provider, *, limit=1, deadline=None, on_change=None, stop_requested=None):
    ledger = SpendLedger(limit, price_for("minimax", provider.model))
    attach_scan_guard(
        provider, ledger, deadline=deadline, on_change=on_change, stop_requested=stop_requested
    )
    return ledger


def generate(provider, layout, **kwargs):
    return provider.generate_subdomain_vocabulary((layout,), TARGET, 20, **kwargs)


def test_calls_public_guarded_chat_once_with_real_additions_schema(monkeypatch, layout):
    provider, base, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    snapshots = []
    ledger = guard(provider, on_change=lambda: snapshots.append(ledger.to_dict()))
    public_chat = Mock(wraps=provider.chat)
    provider.chat = public_chat
    saved = provider.max_tokens, provider.request_timeout
    result = generate(provider, layout)

    assert result.paths == ["api.stage.example.test"]
    assert len(result.confidence_scores) == 1
    assert result.metadata["protocol"] == "grounded-one-slot-vocabulary"
    assert result.metadata["vocabulary_protocol_version"] == 1
    assert result.metadata["verified"] is False
    assert result.metadata["candidate_sources"] == {"api.stage.example.test": "model-vocabulary"}
    proof = result.metadata["provenance"]["api.stage.example.test"]
    assert proof["status"] == "hypothesis"
    assert proof["plan_id"] == layout.plan_id and proof["template"] == layout.template
    assert proof["word"] == "stage" and tuple(proof["evidence"]) == KNOWN
    assert result.metadata["batch_calls"] == 1
    assert result.metadata["token_usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 20,
        "total_tokens": 30,
    }
    public_chat.assert_called_once()
    schema = public_chat.call_args.args[2]
    assert set(schema["properties"]) == {"additions"}
    assert "plan_id" in schema["properties"]["additions"]["items"]["properties"]
    assert "values" in schema["properties"]["additions"]["items"]["properties"]
    assert "candidates" not in schema["properties"]
    assert float(snapshots[0]["reserved_usd"]) > 0
    assert float(snapshots[-1]["reserved_usd"]) == 0
    assert ledger.calls == 1 and ledger.uncertain_calls == 0 and ledger.spent_usd > 0
    assert provider._runtime_generation_kind is None
    assert (provider.max_tokens, provider.request_timeout) == saved
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()
    assert base.with_options.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("thinking", ["disabled", "default"])
def test_m3_vocabulary_uses_scoped_thinking_profile(monkeypatch, layout, thinking):
    provider, base, bounded = make_provider(
        monkeypatch, additions(layout, "stage"), subdomain_thinking=thinking
    )
    provider._runtime_generation_kind = "path"
    generate(provider, layout)
    assert bounded.chat.completions.create.call_args.kwargs["extra_body"] == (
        {"reasoning_split": True, "thinking": {"type": "disabled"}}
        if thinking == "disabled"
        else {"reasoning_split": True}
    )
    assert provider._runtime_generation_kind == "path"
    assert provider._runtime_single_request is False
    assert provider._scan_guard_attached is False
    assert base.with_options.call_args.kwargs["max_retries"] == 0
    base.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "preset,model",
    [
        ("minimax", "MiniMax-M2.7"),
        ("minimax", "MiniMax-M3-highspeed"),
        ("openai", "MiniMax-M3"),
        ("ollama", "qwen3:8b"),
    ],
)
def test_other_backends_never_receive_minimax_m3_switch(monkeypatch, layout, preset, model):
    provider, base, bounded = make_provider(
        monkeypatch, additions(layout, "stage"), preset=preset, model=model
    )
    assert generate(provider, layout).paths == ["api.stage.example.test"]
    assert "thinking" not in bounded.chat.completions.create.call_args.kwargs.get("extra_body", {})
    assert bounded.chat.completions.create.call_count == 1
    base.chat.completions.create.assert_not_called()
    assert provider._runtime_generation_kind is None


def test_native_anthropic_passes_additions_tool_schema_and_parses_tool_result(monkeypatch, layout):
    base, bounded = Mock(), Mock()
    base.with_options.return_value = bounded
    bounded.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="emit_candidates",
                input=json.loads(additions(layout, "stage")),
            )
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
    )
    monkeypatch.setattr(
        "enumeraite.providers.anthropic_provider.anthropic.Anthropic", Mock(return_value=base)
    )
    provider = AnthropicProvider({"api_key": "test-only"})
    ledger = SpendLedger(1, price_for("anthropic", provider.model))
    attach_scan_guard(provider, ledger, deadline=None)
    result = generate(provider, layout)
    assert result.paths == ["api.stage.example.test"]
    wire = bounded.messages.create.call_args.kwargs
    assert wire["tool_choice"] == {"type": "tool", "name": "emit_candidates"}
    assert set(wire["tools"][0]["input_schema"]["properties"]) == {"additions"}
    assert "thinking" not in wire
    assert result.metadata["batch_calls"] == ledger.calls == 1
    assert result.metadata["token_usage"]["output_tokens"] == 20
    assert ledger.spent_usd > 0 and ledger.reserved_usd == 0
    base.messages.create.assert_not_called()
    bounded.messages.create.assert_called_once()
    assert base.with_options.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"additions":[',
        '{"names":["stage"]}',
        '{"additions":[{"plan_id":"unknown","values":["stage"]}]}',
    ],
)
def test_invalid_vocabulary_output_keeps_paid_usage_and_never_retries(monkeypatch, layout, raw):
    provider, _, bounded = make_provider(monkeypatch, raw)
    ledger = guard(provider)
    result = generate(provider, layout)
    assert result.paths == []
    assert result.metadata["protocol"] == "grounded-one-slot-vocabulary"
    assert result.metadata["batch_calls"] == 1
    assert result.metadata["token_usage"]["input_tokens"] == 10
    assert result.metadata["token_usage"]["output_tokens"] == 20
    assert result.metadata.get("rejections") or result.metadata.get("parse_error")
    assert ledger.calls == 1 and ledger.spent_usd > 0 and ledger.reserved_usd == 0
    bounded.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("choices", [[], None, [SimpleNamespace(message=None)]])
def test_empty_completion_envelope_retains_paid_usage_without_retry(monkeypatch, layout, choices):
    provider, base, bounded = make_provider(monkeypatch, "unused")
    bounded.chat.completions.create.return_value.choices = choices
    ledger = guard(provider)
    result = generate(provider, layout)
    assert result.paths == [] and result.metadata["parse_error"]
    assert result.metadata["token_usage"]["total_tokens"] == 30
    assert result.metadata["batch_calls"] == ledger.calls == 1
    assert ledger.uncertain_calls == 0 and ledger.spent_usd > 0 and ledger.reserved_usd == 0
    assert provider._runtime_single_request is False
    assert provider._runtime_generation_kind is None
    bounded.chat.completions.create.assert_called_once()
    base.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("count", [0, 201, True, False, 1.5, "3", None])
def test_invalid_vocabulary_count_fails_before_request(monkeypatch, layout, count):
    provider, base, _ = make_provider(monkeypatch, additions(layout, "stage"))
    with pytest.raises(ValueError):
        provider.generate_subdomain_vocabulary((layout,), TARGET, count)
    base.chat.completions.create.assert_not_called()


def test_empty_layouts_make_no_call_even_when_budget_exhausted(monkeypatch):
    provider, base, bounded = make_provider(monkeypatch, "unused")
    ledger = guard(provider, limit=0)
    result = provider.generate_subdomain_vocabulary((), TARGET, 20)
    assert result.paths == [] and result.metadata["batch_calls"] == 0
    assert ledger.calls == 0 and ledger.spent_usd == ledger.reserved_usd == 0
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_not_called()


def test_cross_target_layout_is_rejected_before_request(monkeypatch, layout):
    provider, base, _ = make_provider(monkeypatch, additions(layout, "stage"))
    with pytest.raises(ValueError):
        provider.generate_subdomain_vocabulary((layout,), "other.test", 20)
    base.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("failure", ["budget", "pause", "save", "deadline"])
def test_pre_wire_guard_failures_restore_scope_and_send_nothing(monkeypatch, layout, failure):
    provider, base, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    monkeypatch.setattr("enumeraite.core.scan_budget.time.monotonic", lambda: 10)
    ledger = guard(
        provider,
        limit=0 if failure == "budget" else 1,
        deadline=9 if failure == "deadline" else None,
        stop_requested=(lambda: True) if failure == "pause" else None,
        on_change=Mock(side_effect=OSError("checkpoint failed")) if failure == "save" else None,
    )
    provider._runtime_generation_kind = "path"
    expected = {
        "budget": BudgetExhausted,
        "pause": ScanPaused,
        "save": OSError,
        "deadline": TimeoutError,
    }[failure]
    with pytest.raises(expected):
        generate(provider, layout)
    assert provider._runtime_generation_kind == "path"
    assert provider._runtime_single_request is False
    assert ledger.calls == 0 and ledger.spent_usd == ledger.reserved_usd == 0
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("unsupported parameter: thinking"),
        RuntimeError("unsupported response_format"),
        KeyboardInterrupt(),
    ],
)
def test_wire_error_is_charged_conservatively_once_no_retry(monkeypatch, layout, error):
    provider, base, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    reservations = []
    ledger = guard(provider, on_change=lambda: reservations.append(ledger.to_dict()))
    bounded.chat.completions.create.side_effect = error
    saved = provider.max_tokens, provider.request_timeout
    with pytest.raises(type(error)):
        generate(provider, layout)
    assert ledger.calls == ledger.uncertain_calls == 1
    assert ledger.spent_usd == float(reservations[0]["reserved_usd"])
    assert ledger.reserved_usd == 0
    assert provider._runtime_generation_kind is None
    assert provider._runtime_single_request is False
    assert (provider.max_tokens, provider.request_timeout) == saved
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()


def test_missing_usage_charges_full_reservation_without_extra_call(monkeypatch, layout):
    provider, _, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    bounded.chat.completions.create.return_value = completion(
        additions(layout, "stage"), usage=False
    )
    reservations = []
    ledger = guard(provider, on_change=lambda: reservations.append(ledger.to_dict()))
    result = generate(provider, layout)
    assert result.paths == ["api.stage.example.test"]
    assert result.metadata["usage_complete"] is False
    assert ledger.calls == ledger.uncertain_calls == 1
    assert ledger.spent_usd == float(reservations[0]["reserved_usd"])
    assert ledger.reserved_usd == 0
    bounded.chat.completions.create.assert_called_once()


def test_guard_blocks_excess_usage_and_prevents_later_vocabulary_request(monkeypatch, layout):
    provider, _, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    bounded.chat.completions.create.return_value.usage.completion_tokens = provider.max_tokens + 1
    ledger = guard(provider)
    with pytest.raises(BudgetExhausted, match="exceeded its reservation"):
        generate(provider, layout)
    assert ledger.blocked and ledger.calls == 1 and ledger.spent_usd > 0
    with pytest.raises(BudgetExhausted, match="inconsistent usage"):
        generate(provider, layout)
    bounded.chat.completions.create.assert_called_once()
    assert provider._runtime_generation_kind is None


def test_pause_after_reservation_checkpoint_cancels_before_wire(monkeypatch, layout):
    provider, base, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    stop = [False]

    def checkpoint():
        stop[0] = True

    ledger = guard(provider, on_change=checkpoint, stop_requested=lambda: stop[0])
    with pytest.raises(ScanPaused):
        generate(provider, layout)
    assert ledger.calls == 0 and ledger.spent_usd == ledger.reserved_usd == 0
    assert provider._runtime_generation_kind is None
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_not_called()


def test_post_wire_deadline_failure_accounts_usage_and_restores_profile(monkeypatch, layout):
    now = [0]
    monkeypatch.setattr("enumeraite.core.scan_budget.time.monotonic", lambda: now[0])
    provider, _, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    ledger = guard(provider, deadline=1)
    response = bounded.chat.completions.create.return_value

    def finish(**_kwargs):
        now[0] = 2
        return response

    bounded.chat.completions.create.side_effect = finish
    saved = provider.max_tokens, provider.request_timeout
    with pytest.raises(TimeoutError, match="during the model request"):
        generate(provider, layout)
    assert ledger.calls == 1 and ledger.uncertain_calls == 0
    assert ledger.spent_usd > 0 and ledger.reserved_usd == 0
    assert provider._runtime_generation_kind is None
    assert (provider.max_tokens, provider.request_timeout) == saved
    bounded.chat.completions.create.assert_called_once()


def test_guard_checkpoint_failure_after_request_does_not_reissue(monkeypatch, layout):
    provider, _, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    saved = Mock(side_effect=[None, OSError("settlement checkpoint failed")])
    ledger = guard(provider, on_change=saved)
    with pytest.raises(OSError, match="settlement checkpoint"):
        generate(provider, layout)
    assert ledger.calls == 1 and ledger.spent_usd > 0 and ledger.reserved_usd == 0
    assert provider._runtime_generation_kind is None
    bounded.chat.completions.create.assert_called_once()


def test_vocabulary_does_not_leak_profile_to_ordinary_chat_or_paths(monkeypatch, layout):
    provider, base, _ = make_provider(monkeypatch, additions(layout, "stage"))
    generate(provider, layout)
    base.chat.completions.create.return_value = completion('{"candidates":[{"value":"/next"}]}')
    provider.chat("ordinary", "chat", None)
    assert base.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning_split": True}
    result = provider.generate_paths(["/known"], TARGET, 1)
    assert result.paths == ["/next"]
    assert result.metadata.get("protocol") != "grounded-one-slot-vocabulary"
    assert base.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning_split": True}
    assert provider._runtime_generation_kind is None


def test_layouts_and_proposals_are_not_promoted_to_verified_evidence(monkeypatch, layout):
    provider, _, _ = make_provider(monkeypatch, additions(layout, "stage", "prod"))
    result = generate(provider, layout, exclusions=("api.stage.example.test",))
    assert not result.paths
    assert tuple(layout.evidence) == KNOWN
    assert set(layout.observed_values) == {"dev", "prod"}
    assert result.metadata["protocol"] == "grounded-one-slot-vocabulary"


def test_count_cap_and_runtime_exclusions_do_not_trigger_refill(monkeypatch, layout):
    provider, base, bounded = make_provider(
        monkeypatch, additions(layout, "stage", "qa", "sandbox")
    )
    provider.set_runtime_exclusions(("api.stage.example.test",))
    result = provider.generate_subdomain_vocabulary((layout,), TARGET, 1)
    assert result.paths == ["api.qa.example.test"]
    assert result.metadata["batch_calls"] == 1
    assert result.metadata["generated"] == 1
    assert provider._runtime_exclusions == ["api.stage.example.test"]
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()


def test_large_history_filters_old_proposals_without_expanding_paid_prompt(monkeypatch, layout):
    provider, base, bounded = make_provider(
        monkeypatch, additions(layout, "stage", "qa", "sandbox")
    )
    provider.set_runtime_exclusions(
        ["api.stage.example.test", *[f"prior{index}.example.test" for index in range(25_000)]]
    )
    excluded = ["api.qa.example.test", *[f"pending{index}.example.test" for index in range(25_001)]]
    result = generate(provider, layout, exclusions=excluded)
    assert result.paths == ["api.sandbox.example.test"]
    assert result.metadata["rejections"]["known_or_excluded"] == 2
    wire = bounded.chat.completions.create.call_args.kwargs
    prompt = json.loads(wire["messages"][1]["content"])
    assert len(prompt["already_attempted_not_evidence"]) <= 200
    base.chat.completions.create.assert_not_called()
    bounded.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(
    "error", [RuntimeError("unsupported response_format"), KeyboardInterrupt()]
)
def test_standalone_one_request_disables_sdk_and_extension_retries(monkeypatch, layout, error):
    provider, base, bounded = make_provider(
        monkeypatch, additions(layout, "stage"), preset="openai", model="gpt-4.1-mini"
    )
    bounded.chat.completions.create.side_effect = error
    with pytest.raises(type(error)):
        generate(provider, layout)
    bounded.chat.completions.create.assert_called_once()
    base.chat.completions.create.assert_not_called()
    assert base.with_options.call_args.kwargs["max_retries"] == 0
    assert provider._runtime_single_request is False
    assert provider._runtime_generation_kind is None
    assert provider._scan_guard_attached is False


@pytest.mark.parametrize("failure", [False, True])
def test_nested_single_request_scope_restores_previous_true_flag(monkeypatch, layout, failure):
    provider, _, bounded = make_provider(monkeypatch, additions(layout, "stage"))
    provider._runtime_single_request = True
    if failure:
        bounded.chat.completions.create.side_effect = RuntimeError("provider failure")
        with pytest.raises(RuntimeError):
            generate(provider, layout)
    else:
        generate(provider, layout)
    assert provider._runtime_single_request is True
    assert provider._runtime_generation_kind is None
    assert provider._scan_guard_attached is False


def test_native_anthropic_standalone_uses_no_retry_child_and_restores_flag(monkeypatch, layout):
    base, bounded = Mock(), Mock()
    base.with_options.return_value = bounded
    bounded.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="emit_candidates",
                input=json.loads(additions(layout, "stage")),
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
    )
    monkeypatch.setattr(
        "enumeraite.providers.anthropic_provider.anthropic.Anthropic", Mock(return_value=base)
    )
    provider = AnthropicProvider({"api_key": "test-only", "max_retries": 4})
    assert generate(provider, layout).paths == ["api.stage.example.test"]
    base.messages.create.assert_not_called()
    bounded.messages.create.assert_called_once()
    assert base.with_options.call_args.kwargs["max_retries"] == 0
    assert provider.max_retries == 4
    assert provider._runtime_single_request is False and provider._scan_guard_attached is False
    bounded.messages.create.side_effect = RuntimeError("tool request failure")
    with pytest.raises(RuntimeError):
        generate(provider, layout)
    assert bounded.messages.create.call_count == 2
    assert provider._runtime_single_request is False and provider._runtime_generation_kind is None
