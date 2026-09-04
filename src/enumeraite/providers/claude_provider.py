"""Backward-compatible Claude provider using the unified implementation."""

from .anthropic_provider import AnthropicProvider


class ClaudeProvider(AnthropicProvider):
    def get_provider_name(self) -> str:
        return "claude"
