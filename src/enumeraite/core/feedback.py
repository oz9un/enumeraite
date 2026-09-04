"""Budget-bounded generate → validate → refeed discovery loop."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import islice
from typing import Any, Callable, Iterable, List, Sequence

from .dns_validator import DNSValidator
from .engine import GenerationEngine
from .http_validator import PathHTTPValidator
from .models import GenerationResult
from .provider import BaseProvider
from .strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from .text_safety import safe_model_text


@dataclass
class ValidationOutcome:
    value: str
    exists: bool | None
    status: str = ""
    detail: str = ""


@dataclass
class FeedbackRound:
    number: int
    generated: List[str]
    hits: List[str]
    misses: List[str]
    feedback: str
    metadata: dict = field(default_factory=dict)
    inconclusive: List[str] = field(default_factory=list)


@dataclass
class FeedbackResult:
    candidates: List[str]
    confidence_scores: List[float]
    validated_hits: List[str]
    rounds: List[FeedbackRound]
    stop_reason: str
    metadata: dict = field(default_factory=dict)

    def generation_result(self) -> GenerationResult:
        return GenerationResult(
            paths=self.candidates,
            confidence_scores=self.confidence_scores,
            metadata={
                **self.metadata,
                "feedback_rounds": [asdict(item) for item in self.rounds],
                "stop_reason": self.stop_reason,
            },
        )


Validator = Callable[[Sequence[str]], Sequence[ValidationOutcome]]


class FeedbackLoop:
    """Iteratively generate candidates and feed validated evidence back in."""

    def __init__(self, provider: BaseProvider, validator: Validator):
        if not callable(validator):
            raise ValueError("validator must be callable")
        self.provider = provider
        self.validator = validator

    def run(
        self,
        known: Sequence[str],
        target: str,
        kind: str,
        batch_size: int = 50,
        max_rounds: int = 3,
        max_candidates: int = 200,
        max_empty_rounds: int = 2,
        min_new_hits: int = 1,
    ) -> FeedbackResult:
        if isinstance(known, (str, bytes)) or not isinstance(known, Sequence):
            raise ValueError("known must be a sequence of seed strings")
        if not isinstance(target, str):
            raise ValueError("target must be a string")
        if kind not in {"path", "subdomain"}:
            raise ValueError("kind must be 'path' or 'subdomain'")
        batch_size = _bounded_int("batch_size", batch_size, 1, 5_000)
        max_rounds = _bounded_int("max_rounds", max_rounds, 1, 100)
        max_candidates = _bounded_int("max_candidates", max_candidates, 1, 5_000)
        max_empty_rounds = _bounded_int("max_empty_rounds", max_empty_rounds, 1, 100)
        min_new_hits = _bounded_int("min_new_hits", min_new_hits, 1, 5_000)

        engine = GenerationEngine(self.provider)
        strategy = SubdomainDiscoveryStrategy() if kind == "subdomain" else PathDiscoveryStrategy()
        evidence = list(dict.fromkeys(known))
        candidates: List[str] = []
        scores: List[float] = []
        hits: List[str] = []
        rounds: List[FeedbackRound] = []
        empty_rounds = 0
        feedback = ""
        stop_reason = "round budget reached"
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        total_batch_calls = 0

        try:
            for number in range(1, max_rounds + 1):
                remaining = max_candidates - len(candidates)
                if remaining <= 0:
                    stop_reason = "candidate budget reached"
                    break
                self.provider.set_runtime_context(feedback)
                self.provider.set_runtime_exclusions(candidates)
                generated = engine.generate_paths(
                    evidence,
                    target,
                    min(batch_size, remaining),
                    strategy=strategy,
                )
                raw_outcomes = (
                    list(islice(self.validator(generated.paths), len(generated.paths) + 1))
                    if generated.paths
                    else []
                )
                if len(raw_outcomes) > len(generated.paths):
                    raise ValueError("validator returned more outcomes than candidates")
                if not all(isinstance(outcome, ValidationOutcome) for outcome in raw_outcomes):
                    raise ValueError("validator must return ValidationOutcome values")
                by_value = {outcome.value: outcome for outcome in raw_outcomes}
                outcomes = [
                    by_value.get(value)
                    or ValidationOutcome(
                        value,
                        None,
                        status="missing",
                        detail="validator returned no result",
                    )
                    for value in generated.paths
                ]
                round_hits = [outcome.value for outcome in outcomes if outcome.exists is True]
                round_misses = [outcome.value for outcome in outcomes if outcome.exists is False]
                round_inconclusive = [
                    outcome.value for outcome in outcomes if outcome.exists is None
                ]
                feedback = _summarize_feedback(kind, outcomes)
                usage = generated.metadata.get("token_usage", {})
                total_input_tokens += _safe_count(usage.get("input_tokens", 0))
                total_cached_input_tokens += _safe_count(usage.get("cached_input_tokens", 0))
                total_output_tokens += _safe_count(usage.get("output_tokens", 0))
                total_reasoning_tokens += _safe_count(generated.metadata.get("reasoning_tokens", 0))
                total_batch_calls += _safe_count(generated.metadata.get("batch_calls", 0))
                rounds.append(
                    FeedbackRound(
                        number,
                        generated.paths,
                        round_hits,
                        round_misses,
                        feedback,
                        metadata=generated.metadata,
                        inconclusive=round_inconclusive,
                    )
                )
                candidates.extend(generated.paths)
                scores.extend(generated.confidence_scores)
                new_hits = [
                    value for value in round_hits if value not in hits and value not in evidence
                ]
                hits.extend(value for value in new_hits if value not in hits)
                evidence.extend(value for value in new_hits if value not in evidence)

                if generated.paths and len(round_inconclusive) == len(generated.paths):
                    stop_reason = "validation inconclusive"
                    break
                if len(new_hits) < min_new_hits:
                    empty_rounds += 1
                else:
                    empty_rounds = 0
                if empty_rounds >= max_empty_rounds:
                    stop_reason = "diminishing returns"
                    break
        finally:
            self.provider.set_runtime_context("")
            self.provider.set_runtime_exclusions(())

        return FeedbackResult(
            candidates=candidates,
            confidence_scores=scores,
            validated_hits=hits,
            rounds=rounds,
            stop_reason=stop_reason,
            metadata={
                "provider": self.provider.get_provider_name(),
                "target": target,
                "kind": kind,
                "validated": True,
                "token_usage": {
                    "input_tokens": total_input_tokens,
                    "cached_input_tokens": total_cached_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens,
                },
                "reasoning_tokens": total_reasoning_tokens,
                "batch_calls": total_batch_calls,
            },
        )


def dns_validator_adapter(
    validator: DNSValidator,
    apex: str,
    detect_wildcard: bool = True,
) -> Validator:
    def validate(values: Sequence[str]) -> Sequence[ValidationOutcome]:
        results = validator.validate_subdomains(
            list(values), detect_wildcard=detect_wildcard, apex=apex
        )
        return [
            ValidationOutcome(
                value=result.subdomain,
                exists=(None if result.error else result.exists),
                status=(
                    "error"
                    if result.error
                    else "dns" if result.exists else "wildcard" if result.wildcard else "nxdomain"
                ),
                detail=safe_model_text(result.error or ",".join(result.ip_addresses), 300),
            )
            for result in results
        ]

    return validate


def http_path_validator_adapter(validator: PathHTTPValidator, target: str) -> Validator:
    def validate(values: Sequence[str]) -> Sequence[ValidationOutcome]:
        results = validator.validate_paths(target, list(values))
        return [
            ValidationOutcome(
                value=result.path,
                exists=(None if result.error or result.status_code is None else result.exists),
                status=str(result.status_code or "error"),
                detail=safe_model_text(
                    (
                        f"soft404_similarity={result.soft_404_similarity:.3f}"
                        if result.soft_404_similarity is not None
                        else result.error or ""
                    ),
                    300,
                ),
            )
            for result in results
        ]

    return validate


def _summarize_feedback(kind: str, outcomes: Iterable[ValidationOutcome]) -> str:
    values = list(outcomes)
    if not values:
        return "No candidates were validated in the previous round."
    hits = [item.value for item in values if item.exists is True]
    misses = [item.value for item in values if item.exists is False]
    inconclusive = [item.value for item in values if item.exists is None]
    lines = [
        f"Validated hits ({len(hits)}): " + (", ".join(hits[:20]) if hits else "none"),
        f"Validated misses ({len(misses)}): " + (", ".join(misses[:20]) if misses else "none"),
        f"Inconclusive ({len(inconclusive)}): "
        + (", ".join(inconclusive[:20]) if inconclusive else "none"),
    ]
    groups: Counter[str] = Counter()
    hit_groups: Counter[str] = Counter()
    for item in values:
        if item.exists is None:
            continue
        if kind == "path":
            parts = item.value.strip("/").split("/")
            group = "/" + "/".join(parts[:2]) if parts else "/"
        else:
            group = item.value.split(".")[0].split("-")[0]
        groups[group] += 1
        if item.exists is True:
            hit_groups[group] += 1
    patterns = [
        f"{group}: {hit_groups[group]}/{total} hit" for group, total in groups.most_common(8)
    ]
    if patterns:
        lines.append("Observed validation patterns: " + "; ".join(patterns))
    return "\n".join(lines)


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    parsed = float(value)
    return min(1_000_000_000, max(0, int(parsed))) if math.isfinite(parsed) else 0
