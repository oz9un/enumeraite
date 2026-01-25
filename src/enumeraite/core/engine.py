"""Core generation engine for path generation."""
from typing import List, Optional, Set
from .provider import BaseProvider
from .models import GenerationResult
from .strategies import PromptStrategy

class GenerationEngine:
    """Core engine for orchestrating path generation."""

    def __init__(self, provider: BaseProvider):
        """Initialize the generation engine.

        Args:
            provider: AI provider for path generation
        """
        self.provider = provider
        self._seen_paths: Set[str] = set()

    def generate_paths(self, known_paths: List[str], count: int, max_depth: int = None, strategy: Optional[PromptStrategy] = None) -> GenerationResult:
        """Generate new paths using the configured provider.

        Args:
            known_paths: List of known API paths to learn from
            count: Number of paths to generate
            max_depth: Optional maximum depth for generated paths
            strategy: Optional prompt strategy for discovery type

        Returns:
            GenerationResult with filtered and deduplicated paths
        """
        result = self.provider.generate_paths(known_paths, "", count, max_depth, strategy)

        # Filter out duplicates, previously seen paths, and known paths
        filtered_paths = []
        filtered_scores = []
        known_normalized = {self._normalize_path(path) for path in known_paths}

        for path, score in zip(result.paths, result.confidence_scores):
            normalized_path = self._normalize_path(path)
            if (normalized_path not in self._seen_paths and
                normalized_path not in known_normalized):
                filtered_paths.append(path)
                filtered_scores.append(score)
                self._seen_paths.add(normalized_path)

        return GenerationResult(
            paths=filtered_paths,
            confidence_scores=filtered_scores,
            metadata=result.metadata
        )


    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison.

        Args:
            path: API path to normalize

        Returns:
            Normalized path (lowercase, trailing slash removed)
        """
        return path.lower().rstrip('/')

    def add_discovered_paths(self, paths: List[str]) -> None:
        """Add newly discovered paths to the seen set.

        Args:
            paths: List of paths to add to seen set
        """
        for path in paths:
            self._seen_paths.add(self._normalize_path(path))

    def reset_seen_paths(self) -> None:
        """Reset the seen paths set (useful for new targets)."""
        self._seen_paths.clear()