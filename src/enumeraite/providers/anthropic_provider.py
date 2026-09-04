"""Native Anthropic backend with tool-forced structured candidate output."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import anthropic

from ..core.text_safety import safe_model_text
from .llm_provider import ChatResult, LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, config: dict[str, Any]):
        merged = dict(config)
        merged.setdefault("model", "claude-sonnet-4-6")
        if str(merged["model"]).startswith("anthropic/"):
            merged["model"] = str(merged["model"]).split("/", 1)[1]
        super().__init__(merged)
        self.client = anthropic.Anthropic(
            api_key=merged.get("api_key"),
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]]) -> ChatResult:
        args: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if schema is not None:
            args["tools"] = [
                {
                    "name": "emit_candidates",
                    "description": "Emit the ranked discovery candidates.",
                    "input_schema": schema,
                }
            ]
            args["tool_choice"] = {"type": "tool", "name": "emit_candidates"}
        response = self.client.messages.create(**args)

        text = ""
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use" and getattr(block, "name", "") == "emit_candidates":
                text = json.dumps(getattr(block, "input", {}))
                break
            if hasattr(block, "text"):
                text += str(block.text)
        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            input_tokens=_safe_int(getattr(usage, "input_tokens", 0)),
            cached_input_tokens=_safe_int(getattr(usage, "cache_read_input_tokens", 0)),
            output_tokens=_safe_int(getattr(usage, "output_tokens", 0)),
            model=safe_model_text(getattr(response, "model", "") or self.model, 200),
            finish_reason=safe_model_text(getattr(response, "stop_reason", "") or "", 100),
        )

    def get_provider_name(self) -> str:
        return "anthropic"


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    parsed = float(value)
    return max(0, int(parsed)) if math.isfinite(parsed) else 0
