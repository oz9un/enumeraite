"""Hosted model proposals retain user-facing explanations, not hidden prompts."""

import json

from enumeraite.core.discovery_session import DiscoverySession, ScanOptions
from enumeraite.core.scan_budget import ModelPrice, SpendLedger
from enumeraite.providers.llm_provider import ChatResult, LLMProvider


class ContextProvider(LLMProvider):
    def __init__(self):
        super().__init__({"model": "fixture", "hybrid": False, "max_batch_calls": 1})
        self.prompts = []

    def get_provider_name(self):
        return "fixture"

    def chat(self, system, user, schema):
        self.prompts.append(user)
        return ChatResult(
            json.dumps(
                {
                    "candidates": [
                        {
                            "value": "ny-02-iphone.example.test",
                            "confidence": 0.7,
                            "rationale": "Observed ny-01-iphone suggests a zero-padded numeric sibling.",
                        }
                    ]
                }
            ),
            input_tokens=100,
            output_tokens=50,
        )


def test_model_uses_verified_context_feedback_and_retains_explanation():
    provider = ContextProvider()
    session = DiscoverySession(
        "example.test",
        ScanOptions(passive=False),
        object(),
        provider=provider,
        ledger=SpendLedger(1, ModelPrice(1, 1)),
    )
    known = "ny-01-iphone.example.test"
    session._enqueue(known, source="seed")
    session.nodes[known].update(status="verified", attempts=1, addresses=["192.0.2.1"])
    session.state["attempted"] = 1
    session._enqueue("wrong.example.test", source="pattern")
    session.nodes["wrong.example.test"].update(status="miss", attempts=1)
    session.state["attempted"] = 2
    assert session._model_candidates([known]) == 1
    prompt = provider.prompts[0]
    assert known in prompt
    assert "miss: wrong.example.test" in prompt
    node = session.nodes["ny-02-iphone.example.test"]
    assert node["status"] == "pending"
    assert "zero-padded numeric sibling" in node["reason"]
    assert node["source"] == "model"
    assert session.summary()["ledger"]["calls"] == 3
