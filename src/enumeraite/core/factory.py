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

        try:
            from ..providers.huggingface_provider import HuggingFaceProvider
            self.register_provider("huggingface", HuggingFaceProvider)
        except ImportError:
            # HuggingFace provider not available (missing dependencies)
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

            # Provide installation instructions for missing providers
            install_msg = ""
            if provider_name == "openai":
                install_msg = "\n\nTo install OpenAI support:\n  pip install \"enumeraite[openai]\""
            elif provider_name == "claude":
                install_msg = "\n\nTo install Claude support:\n  pip install \"enumeraite[claude]\""
            elif provider_name in ["anthropic"]:
                install_msg = "\n\nTo install Claude support:\n  pip install \"enumeraite[claude]\""
            else:
                install_msg = "\n\nTo install all providers:\n  pip install \"enumeraite[all]\""

            raise ValueError(f"Provider '{provider_name}' not available. Available: {available}{install_msg}")

        provider_config = self.config.get_provider_config(provider_name)
        if not provider_config:
            # Provide helpful setup instructions for each provider
            if provider_name == "claude":
                raise ValueError(
                    f"Claude provider not configured. Please set up:\n"
                    f"  export ANTHROPIC_API_KEY='your-api-key-here'\n"
                    f"  Or create enumeraite.json with your Claude API key.\n"
                    f"  Get API key: https://console.anthropic.com/"
                )
            elif provider_name == "openai":
                raise ValueError(
                    f"OpenAI provider not configured. Please set up:\n"
                    f"  export OPENAI_API_KEY='your-api-key-here'\n"
                    f"  Or create enumeraite.json with your OpenAI API key.\n"
                    f"  Get API key: https://platform.openai.com/api-keys"
                )
            else:
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

        # Intelligent fallback: prioritize quality models
        priority_order = ["claude", "openai", "huggingface"]
        for provider_name in priority_order:
            if provider_name in available:
                return self.create_provider(provider_name)

        # If none of the priority providers available, use first available
        return self.create_provider(available[0])