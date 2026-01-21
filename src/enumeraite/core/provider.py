"""Base provider interface for AI path generation."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .models import GenerationResult

class BaseProvider(ABC):
    """Abstract base class for AI providers that generate API paths."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config

    @abstractmethod
    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new API paths based on known paths.

        Args:
            known_paths: List of known API paths to learn from
            target: Target domain name
            count: Number of paths to generate

        Returns:
            GenerationResult containing generated paths and confidence scores
        """
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name for identification.

        Returns:
            String identifier for this provider
        """
        raise NotImplementedError