"""Provider factory for creating and managing AI providers."""
from typing import Dict, List, Type
from .provider import BaseProvider
from .config import Config

class ProviderFactory:
    """Factory for creating and managing AI providers."""

    _registry: Dict[str, Type[BaseProvider]] = {}

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
        try:
            from ..providers.openai_provider import OpenAIProvider
            self.register_provider("openai", OpenAIProvider)
        except ImportError:
            # OpenAI provider not available (missing dependencies)
            pass

        try:
            from ..providers.claude_provider import ClaudeProvider
            self.register_provider("claude", ClaudeProvider)
        except ImportError:
            # Claude provider not available (missing dependencies)
            pass

    def create_provider(self, provider_name: str) -> BaseProvider:
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
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

        provider_config = self.config.get_provider_config(provider_name)
        if not provider_config:
            raise ValueError(f"No configuration found for provider '{provider_name}'")

        provider_class = self._registry[provider_name]
        return provider_class(provider_config)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names.

        Returns:
            List of provider names that are both registered and configured
        """
        available = []
        for name in self._registry.keys():
            if self.config.get_provider_config(name):
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

        # Fall back to first available
        return self.create_provider(available[0])