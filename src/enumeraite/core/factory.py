"""Provider factory for creating and managing AI providers."""

from typing import Any, ClassVar, Dict, List, Optional, Type

from .config import Config
from .provider import BaseProvider

HOSTED_API_KEY_HINTS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
}


class ProviderFactory:
    """Factory for creating and managing AI providers."""

    _registry: ClassVar[Dict[str, Type[BaseProvider]]] = {}

    def __init__(self, config: Config):
        """Initialize the factory with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self._register_default_providers()

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a new provider class.

        Args:
            name: Provider name identifier
            provider_class: Provider class to register
        """
        cls._registry[name] = provider_class

    def _register_default_providers(self) -> None:
        """Register built-in providers."""
        from ..providers.anthropic_provider import AnthropicProvider
        from ..providers.claude_provider import ClaudeProvider
        from ..providers.huggingface_provider import HuggingFaceProvider
        from ..providers.openai_compatible import PRESETS, OpenAICompatibleProvider
        from ..providers.openai_provider import OpenAIProvider

        self.register_provider("openai", OpenAIProvider)
        self.register_provider("claude", ClaudeProvider)
        self.register_provider("anthropic", AnthropicProvider)
        self.register_provider("openai-compat", OpenAICompatibleProvider)
        for preset in PRESETS:
            # Never let a preset alias replace the backward-compatible
            # ``openai`` registration. A regression test guards this.
            if preset not in self._registry:
                self.register_provider(preset, OpenAICompatibleProvider)
        # This module performs its own optional-runtime check and raises a
        # targeted installation hint only when the provider is selected.
        self.register_provider("huggingface", HuggingFaceProvider)

    def create_provider(
        self, provider_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            provider_name: Name of the provider to create

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is unknown or not configured
        """
        if provider_name not in self._registry:
            available = ", ".join(self._registry.keys())

            # Provide installation instructions for missing providers
            install_msg = ""
            if provider_name == "openai":
                install_msg = "\n\nReinstall provider support:\n  pip install --upgrade enumeraite"
            elif provider_name == "claude":
                install_msg = "\n\nReinstall provider support:\n  pip install --upgrade enumeraite"
            elif provider_name in ["anthropic"]:
                install_msg = "\n\nReinstall provider support:\n  pip install --upgrade enumeraite"
            else:
                install_msg = '\n\nTo install all providers:\n  pip install "enumeraite[all]"'

            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {available}{install_msg}"
            )

        provider_config = self.config.get_provider_config(provider_name)
        if not provider_config:
            raise ValueError(f"No configuration found for provider '{provider_name}'")

        provider_class = self._registry[provider_name]
        provider_config = dict(provider_config)
        provider_config.update(overrides or {})
        if provider_name in HOSTED_API_KEY_HINTS and not provider_config.get("api_key"):
            raise ValueError(
                f"Provider '{provider_name}' requires {HOSTED_API_KEY_HINTS[provider_name]} "
                "or an api_key in enumeraite.json"
            )
        if provider_class.__name__ == "OpenAICompatibleProvider":
            provider_config.setdefault(
                "preset",
                provider_name if provider_name != "openai-compat" else "openai",
            )
        return provider_class(provider_config)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names.

        Returns:
            List of provider names that are both registered and configured
        """
        available = []
        for name in self._registry.keys():
            provider_config = self.config.get_provider_config(name)
            if provider_config and (
                name not in HOSTED_API_KEY_HINTS or provider_config.get("api_key")
            ):
                available.append(name)
        return available

    def get_default_provider(self) -> BaseProvider:
        """Get the default provider instance.

        Returns:
            Default provider instance

        Raises:
            ValueError: If no providers are available
        """
        available = self.get_available_providers()
        if not available:
            raise ValueError("No providers available. Please configure at least one provider.")

        default_name = self.config.default_provider
        if default_name in available:
            return self.create_provider(default_name)

        # Intelligent fallback: prioritize quality models
        priority_order = [
            "anthropic",
            "openai",
            "openrouter",
            "deepseek",
            "minimax",
            "groq",
            "together",
            "ollama",
            "vllm",
            "llamacpp",
            "local",
            "claude",
            "huggingface",
        ]
        for provider_name in priority_order:
            if provider_name in available:
                return self.create_provider(provider_name)

        # If none of the priority providers available, use first available
        return self.create_provider(available[0])
