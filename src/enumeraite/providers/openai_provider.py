"""Backward-compatible OpenAI provider using the unified implementation."""

from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self, config):
        merged = dict(config)
        merged["preset"] = "openai"
        super().__init__(merged)

    def get_provider_name(self) -> str:
        return "openai"
