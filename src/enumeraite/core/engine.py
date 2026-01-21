"""Core generation engine for path generation and validation."""
import asyncio
from typing import List, Optional, Set, Tuple
from .provider import BaseProvider
from .validator import HTTPValidator
from .models import GenerationResult, ValidationResult

class GenerationEngine:
    """Core engine for orchestrating path generation and validation."""

    def __init__(self, provider: BaseProvider, validator: Optional[HTTPValidator] = None):
        """Initialize the generation engine.

        Args:
            provider: AI provider for path generation
            validator: Optional HTTP validator for path testing
        """
        self.provider = provider
        self.validator = validator
        self._seen_paths: Set[str] = set()

    def generate_paths(self, known_paths: List[str], target: str, count: int) -> GenerationResult:
        """Generate new paths using the configured provider.

        Args:
            known_paths: List of known API paths to learn from
            target: Target domain name
            count: Number of paths to generate

        Returns:
            GenerationResult with filtered and deduplicated paths
        """
        result = self.provider.generate_paths(known_paths, target, count)

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

    async def generate_and_validate_paths(self, known_paths: List[str], target: str,
                                        count: int, methods: Optional[List[str]] = None) -> Tuple[GenerationResult, List[ValidationResult]]:
        """Generate paths and validate them via HTTP.

        Args:
            known_paths: List of known API paths
            target: Target domain name
            count: Number of paths to generate
            methods: HTTP methods to test (default: ["GET"])

        Returns:
            Tuple of (GenerationResult, List of ValidationResult)

        Raises:
            ValueError: If no validator is configured
        """
        if not self.validator:
            raise ValueError("No validator configured for validation")

        # Generate paths
        generation_result = self.generate_paths(known_paths, target, count)

        if not generation_result.paths:
            return generation_result, []

        # Validate generated paths
        base_url = f"https://{target}"
        validation_results = await self.validator.validate_paths(
            base_url, generation_result.paths, methods
        )

        return generation_result, validation_results

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