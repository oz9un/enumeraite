"""Core generation engine for path generation."""

import asyncio
import inspect
from typing import Any, List, Optional, Sequence, Set

from .http_validator import PathHTTPResult, PathHTTPValidator, _normalize_base_url
from .models import GenerationResult, ValidationResult
from .prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC, spec_for
from .provider import BaseProvider
from .strategies import PromptStrategy


class GenerationEngine:
    """Core engine for orchestrating path generation."""

    def __init__(self, provider: BaseProvider, validator=None):
        """Initialize the generation engine.

        Args:
            provider: AI provider for path generation
        """
        self.provider = provider
        self.validator = validator
        self._seen_paths: Set[str] = set()

    def generate_paths(
        self,
        known_paths: List[str],
        target: str,
        count: int,
        max_depth: Optional[int] = None,
        strategy: Optional[PromptStrategy] = None,
    ) -> GenerationResult:
        """Generate new paths using the configured provider.

        Args:
            known_paths: List of known API paths to learn from
            count: Number of paths to generate
            max_depth: Optional maximum depth for generated paths
            strategy: Optional prompt strategy for discovery type

        Returns:
            GenerationResult with filtered and deduplicated paths
        """
        if isinstance(known_paths, (str, bytes)) or not isinstance(known_paths, Sequence):
            raise ValueError("known_paths must be a sequence of seed strings")
        if not all(isinstance(path, str) for path in known_paths):
            raise ValueError("known_paths must contain only seed strings")
        if not isinstance(target, str):
            raise ValueError("target must be a string")
        count = _bounded_int("count", count, 1, 5_000)
        if max_depth is not None:
            max_depth = _bounded_int("max_depth", max_depth, 1, 1_000)
        kind = (
            "subdomain"
            if strategy and "subdomain" in strategy.__class__.__name__.lower()
            else "path"
        )
        if kind == "subdomain":
            target = target.lower().rstrip(".")
            if not SUBDOMAIN_SPEC.validator(target):
                raise ValueError("subdomain target must be a valid DNS apex")
            supplied_count = len(known_paths)
            known_paths = [
                SUBDOMAIN_SPEC.normalizer(path)
                for path in known_paths
                if SUBDOMAIN_SPEC.validator(path)
                and _within_apex(SUBDOMAIN_SPEC.normalizer(path), target)
            ]
            known_paths = list(dict.fromkeys(known_paths))
            seed_scope_rejected = supplied_count - len(known_paths)
            if not known_paths:
                raise ValueError("no in-scope subdomain seeds were supplied")
        else:
            seed_scope_rejected = 0
        if max_depth is None and strategy is None:
            # Keep the exact three-argument call for simple/legacy providers.
            result = self.provider.generate_paths(known_paths, target, count)
        else:
            result = self.provider.generate_paths(known_paths, target, count, max_depth, strategy)

        # Filter out duplicates, previously seen paths, and known paths
        filtered_paths = []
        filtered_scores = []
        spec = spec_for(kind)
        known_normalized = {spec.identity(path) for path in known_paths}
        invalid_rejected = 0
        depth_rejected = 0
        scope_rejected = 0

        for path, score in zip(result.paths, result.confidence_scores, strict=True):
            if not spec.validator(path):
                invalid_rejected += 1
                continue
            value = spec.normalizer(path)
            if kind == "path" and max_depth is not None:
                if len([part for part in value.split("/") if part]) > max_depth:
                    depth_rejected += 1
                    continue
            if kind == "subdomain" and target and not _within_apex(value, target):
                scope_rejected += 1
                continue
            normalized_path = spec.identity(value)
            if normalized_path not in self._seen_paths and normalized_path not in known_normalized:
                filtered_paths.append(value)
                filtered_scores.append(score)
                self._seen_paths.add(normalized_path)

        metadata = dict(result.metadata)
        metadata["engine_invalid_rejected"] = invalid_rejected
        metadata["engine_depth_rejected"] = depth_rejected
        metadata["engine_scope_rejected"] = scope_rejected
        metadata["engine_seed_scope_rejected"] = seed_scope_rejected
        return GenerationResult(
            paths=filtered_paths,
            confidence_scores=filtered_scores,
            metadata=metadata,
        )

    async def generate_and_validate_paths(
        self,
        known_paths: List[str],
        target: str,
        count: int,
        methods: Optional[Sequence[str]] = None,
        max_depth: Optional[int] = None,
        strategy: Optional[PromptStrategy] = None,
    ) -> tuple[GenerationResult, List[ValidationResult]]:
        """Generate paths and validate them with a built-in or legacy validator.

        The built-in validator is synchronous, so run it in a worker thread to
        avoid blocking the event loop.  Custom validators using the original
        async ``validate_paths(base_url, paths, methods)`` protocol remain
        supported.
        """
        if self.validator is None:
            raise ValueError("No validator configured")
        generated = self.generate_paths(known_paths, target, count, max_depth, strategy)
        base_url = _normalize_base_url(target)
        selected_methods = list(methods or ["GET"])

        if isinstance(self.validator, PathHTTPValidator):
            if selected_methods != ["GET"]:
                raise ValueError("PathHTTPValidator supports the GET method only")
            raw_results = await asyncio.to_thread(
                self.validator.validate_paths,
                base_url,
                generated.paths,
            )
            validation = [_as_validation_result(result) for result in raw_results]
        else:
            outcome = self.validator.validate_paths(
                base_url,
                generated.paths,
                selected_methods,
            )
            validation = await outcome if inspect.isawaitable(outcome) else outcome
        return generated, list(validation)

    def _normalize_path(self, path: str) -> str:
        """Normalize path for comparison.

        Args:
            path: API path to normalize

        Returns:
            Normalized path (lowercase, trailing slash removed)
        """
        value = path.strip()
        return PATH_SPEC.identity(value) if value.startswith("/") else value.lower().rstrip(".")

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


def _within_apex(value: str, target: str) -> bool:
    candidate = value.lower().rstrip(".")
    apex = target.lower().rstrip(".")
    return candidate != apex and candidate.endswith("." + apex)


def _as_validation_result(result: PathHTTPResult) -> ValidationResult:
    return ValidationResult(
        path=result.path,
        status_code=result.status_code,
        exists=result.exists,
        method="GET",
        response_time=result.response_time,
    )


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
