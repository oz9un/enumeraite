"""OpenAI-compatible backend covering hosted and local chat servers."""

from __future__ import annotations

import math
from typing import Any, Optional

import openai

from ..core.text_safety import safe_model_text
from ..core.url_safety import is_loopback_host, parse_http_url
from .llm_provider import ChatResult, LLMProvider

PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "", "model": "gpt-4o-mini", "api_env": "OPENAI_API_KEY"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "api_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "api_env": "",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "local-model",
        "api_env": "",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "model": "local-model",
        "api_env": "",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_env": "GROQ_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_env": "DEEPSEEK_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "api_env": "TOGETHER_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-M3",
        "api_env": "MINIMAX_API_KEY",
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "api_env": "",
    },
}


class OpenAICompatibleProvider(LLMProvider):
    """A provider for any server implementing OpenAI chat completions."""

    def __init__(self, config: dict[str, Any]):
        self.preset = str(config.get("preset", "openai")).lower()
        preset = PRESETS.get(self.preset, PRESETS["openai"])
        merged = dict(config)
        merged.setdefault("model", preset["model"])
        merged.setdefault("base_url", preset["base_url"])
        if self.preset == "minimax":
            # M3/M2.7 are reasoning models; 4k can be exhausted before they
            # emit the requested JSON even for a modest candidate batch.
            merged.setdefault("max_tokens", 8192)
            merged.setdefault("batch_size", 20)
            merged.setdefault("max_batch_calls", 3)
        if not merged.get("api_key") and self.preset in {
            "ollama",
            "vllm",
            "llamacpp",
            "local",
        }:
            merged["api_key"] = "sk-local"
        if merged.get("base_url"):
            merged["base_url"] = _validated_base_url(
                str(merged["base_url"]),
                local=self.preset in {"ollama", "vllm", "llamacpp", "local"},
            )
        super().__init__(merged)

        client_args: dict[str, Any] = {
            "api_key": merged.get("api_key"),
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        if merged.get("base_url"):
            client_args["base_url"] = merged["base_url"]
        self.client = openai.OpenAI(**client_args)

    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]]) -> ChatResult:
        base_args: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.preset == "minimax":
            # MiniMax documents max_completion_tokens for its OpenAI-compatible
            # endpoint and can split reasoning away from the JSON response.
            base_args["max_completion_tokens"] = self.max_tokens
            base_args["extra_body"] = {"reasoning_split": True}
        else:
            base_args["max_tokens"] = self.max_tokens
        enhanced = dict(base_args)
        # MiniMax does not document response_format support. Its strict prompt
        # plus the shared tolerant parser are used instead of paying for a
        # guaranteed-to-fail extension request before every completion.
        if schema is not None and self.preset != "minimax":
            enhanced["response_format"] = {"type": "json_object"}
        if self.confidence_mode == "logprob" and self.preset != "minimax":
            enhanced["logprobs"] = True

        try:
            response = self.client.chat.completions.create(**enhanced)
        except Exception as exc:
            # Several local compatibility servers reject response_format or
            # logprobs even though they otherwise implement chat completions.
            if enhanced == base_args or not _extensions_rejected(exc):
                raise
            response = self.client.chat.completions.create(**base_args)

        choice = response.choices[0]
        content = getattr(getattr(choice, "message", None), "content", "") or ""
        usage = getattr(response, "usage", None)
        input_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
        output_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
        token_logprobs = []
        logprobs = getattr(choice, "logprobs", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        logprob_content = getattr(logprobs, "content", None)
        if not isinstance(logprob_content, (list, tuple)):
            logprob_content = []
        for item in logprob_content:
            token = getattr(item, "token", "")
            logprob = getattr(item, "logprob", None)
            if token and isinstance(logprob, (int, float)):
                token_logprobs.append((token, float(logprob)))
        return ChatResult(
            text=content,
            input_tokens=input_tokens,
            cached_input_tokens=_safe_int(getattr(prompt_details, "cached_tokens", 0)),
            output_tokens=output_tokens,
            token_logprobs=token_logprobs,
            model=safe_model_text(getattr(response, "model", "") or self.model, 200),
            finish_reason=safe_model_text(getattr(choice, "finish_reason", "") or "", 100),
            reasoning_tokens=_safe_int(getattr(completion_details, "reasoning_tokens", 0)),
        )

    def get_provider_name(self) -> str:
        return f"openai-compat:{self.preset}"


def _safe_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, int(number)) if math.isfinite(number) else 0


def _validated_base_url(value: str, *, local: bool) -> str:
    """Reject credential-bearing or cleartext remote provider endpoints."""
    text = value.strip()
    parsed = parse_http_url(text, label="provider base_url")
    hostname = parsed.hostname
    if hostname is None:  # Defensive for alternate URL parser implementations.
        raise ValueError("provider base_url contains an invalid hostname")
    if parsed.query or parsed.fragment:
        raise ValueError("provider base_url must not contain a query or fragment")
    if parsed.scheme == "http" and not local and not is_loopback_host(hostname):
        raise ValueError(
            "hosted provider base_url must use HTTPS; use a local preset for "
            "an intentional cleartext local server"
        )
    return text.rstrip("/")


def _extensions_rejected(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "response_format",
            "logprobs",
            "unsupported parameter",
            "unrecognized parameter",
            "unknown parameter",
        )
    )
