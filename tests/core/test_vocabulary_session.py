"""Optional vocabulary shares the existing paid-pass and DNS boundaries."""

import copy
import json

import pytest

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions, validate_checkpoint
from enumeraite.core.scan_budget import ModelPrice, SpendLedger
from enumeraite.providers.llm_provider import ChatResult, LLMProvider

TARGET = "example.test"
KNOWN = ["api.dev.example.test", "api.prod.example.test"]


class VocabularyModel(LLMProvider):
    def __init__(self, answer="valid"):
        super().__init__({"model": "fixture", "hybrid": False, "max_batch_calls": 1})
        self.answer = answer
        self.requests = []

    def get_provider_name(self):
        return "vocabulary-fixture"

    def chat(self, system, user, schema):
        vocabulary = "additions" in schema.get("properties", {})
        self.requests.append(("vocabulary" if vocabulary else "direct", user))
        if vocabulary:
            if self.answer == "error":
                raise TimeoutError("fixture timed out")
            if self.answer == "malformed":
                return ChatResult("not JSON", input_tokens=100, output_tokens=20)
            layouts = json.loads(user)["layouts"]
            layout = next(item for item in layouts if item["template"] == "api.{value}")
            body = {"additions": [{"plan_id": layout["plan_id"], "values": ["stage"]}]}
        else:
            body = {
                "candidates": [
                    {
                        "value": f"direct-{len(self.requests)}.{TARGET}",
                        "confidence": 0.5,
                        "rationale": "Fixture hypothesis",
                    }
                ]
            }
        return ChatResult(json.dumps(body), input_tokens=100, output_tokens=20)


def make_session(*, enabled=True, answer="valid", known=KNOWN, checkpoint=None):
    model = VocabularyModel(answer)
    ledger = SpendLedger(1, ModelPrice(1, 2))
    session = DiscoverySession(
        TARGET,
        ScanOptions(passive=False, vocabulary_breadth=enabled),
        object(),
        provider=model,
        ledger=ledger,
        checkpoint=checkpoint,
    )
    for value in known:
        session._enqueue(value, source="seed")
        session.nodes[value].update(status="verified", attempts=1, addresses=["192.0.2.1"])
    session.state["attempted"] = len(known)
    return session, model, ledger


def test_vocabulary_replaces_third_pass_and_keeps_hypotheses_pending():
    session, model, ledger = make_session()
    assert session._model_candidates(KNOWN) == 3
    assert [kind for kind, _ in model.requests] == ["direct", "direct", "vocabulary"]
    assert ledger.calls == 3
    node = session.nodes["api.stage.example.test"]
    assert node["status"] == "pending" and node["attempts"] == 0
    assert node["source"] == "model" and node["parent"] is None
    assert node["pattern"] == "api.{value}.example.test"
    assert "api.stage.example.test" not in session.findings
    assert session.state["model_batches"] == 3
    payload = json.loads(model.requests[-1][1])
    assert payload["candidate_limit"] == 20
    witnesses = {name for layout in payload["layouts"] for name in layout["verified_examples"]}
    assert witnesses == set(KNOWN)
    assert "direct-1.example.test" in payload["already_attempted_not_evidence"]
    assert not model._runtime_exclusions
    validate_checkpoint(session.state, TARGET)


@pytest.mark.parametrize("enabled,known", [(False, KNOWN), (True, ["host-01.example.test"])])
def test_disabled_or_ineligible_vocabulary_chooses_direct_before_any_call(enabled, known):
    session, model, ledger = make_session(enabled=enabled, known=known)
    session._model_candidates(known)
    assert [kind for kind, _ in model.requests] == ["direct"] * 3
    assert ledger.calls == 3


@pytest.mark.parametrize("answer", ["malformed", "error"])
def test_bad_vocabulary_does_not_trigger_paid_fallback_or_lose_prior_batches(answer):
    saved = []
    session, model, ledger = make_session(
        answer=answer, checkpoint=lambda s: saved.append(copy.deepcopy(s))
    )
    session._model_candidates(KNOWN)
    assert [kind for kind, _ in model.requests] == ["direct", "direct", "vocabulary"]
    assert ledger.calls == 3
    assert ledger.spent_usd > 0
    assert all(f"direct-{i}.example.test" in session.nodes for i in (1, 2))
    assert any(
        state["model_batches"] == 2
        and any(node["value"] == "direct-2.example.test" for node in state["nodes"])
        for state in saved
    )
    assert "api.stage.example.test" not in session.nodes
    if answer == "error":
        assert not session.model_enabled
        assert ledger.uncertain_calls == 1
        assert saved[-1]["ledger"]["uncertain_calls"] == 1
    else:
        assert ledger.uncertain_calls == 0


def test_resume_preserves_next_vocabulary_pass_and_old_checkpoints_default_off():
    original, _, _ = make_session()
    original.state["model_batches"] = 2
    original.save()
    model = VocabularyModel()
    resumed = DiscoverySession(
        TARGET,
        ScanOptions(**original.state["options"]),
        object(),
        provider=model,
        ledger=SpendLedger(1, ModelPrice(1, 2)),
        state=copy.deepcopy(original.state),
    )
    assert resumed._model_batch(KNOWN, 1) == 1
    assert [kind for kind, _ in model.requests] == ["vocabulary"]
    assert json.loads(model.requests[0][1])["candidate_limit"] == 1
    old = copy.deepcopy(original.state)
    old["options"].pop("vocabulary_breadth")
    old.pop("vocabulary_protocol_version")
    legacy = DiscoverySession(TARGET, ScanOptions(**old["options"]), object(), state=old)
    assert legacy.options.vocabulary_breadth is False
    assert legacy.state["vocabulary_protocol_version"] == 1


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_vocabulary_option_requires_boolean(value):
    with pytest.raises(ValueError, match="vocabulary_breadth"):
        ScanOptions(vocabulary_breadth=value)


@pytest.mark.parametrize("value", [True, 0, 2, "1", None])
def test_unknown_protocol_checkpoint_rejected(value):
    session, _, _ = make_session()
    session.state["vocabulary_protocol_version"] = value
    with pytest.raises(ValueError, match="vocabulary protocol"):
        validate_checkpoint(session.state, TARGET)
