"""Base provider interface for AI path generation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .models import GenerationResult

if TYPE_CHECKING:
    from .strategies import PromptStrategy


class BaseProvider(ABC):
    """Abstract base class for AI providers that generate API paths."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.model: Any = config.get("model")

    @abstractmethod
    def generate_paths(
        self,
        known_paths: List[str],
        target: str,
        count: int,
        max_depth: Optional[int] = None,
        strategy: Optional["PromptStrategy"] = None,
    ) -> GenerationResult:
        """Generate new API paths based on known paths.

        Args:
            known_paths: List of known API paths to learn from
            target: Target domain name
            count: Number of paths to generate
            max_depth: Optional maximum depth for generated paths (e.g., 3 for /api/users/123)
            strategy: Optional prompt strategy for discovery type (paths, subdomains, etc.)

        Returns:
            GenerationResult containing generated paths and confidence scores
        """
        raise NotImplementedError

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Optional raw-chat hook used by the legacy analysis engines."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support raw chat calls")

    def set_runtime_context(self, value: str = "") -> None:
        """Accept ephemeral validation feedback when a provider supports it."""

    def set_runtime_exclusions(self, values: Sequence[str] = ()) -> None:
        """Accept already-spent candidates when a provider supports it."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for identification.

        Returns:
            String identifier for this provider
        """
        raise NotImplementedError
