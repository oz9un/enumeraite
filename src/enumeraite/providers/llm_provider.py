"""Unified implementation shared by chat-completion LLM backends."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from math import ceil
from typing import Any, List, Optional, Sequence, Tuple

from ..core.candidate_pool import structural_candidates
from ..core.context import collect_context
from ..core.discovery_vocabulary import (
    VOCABULARY_PROTOCOL_VERSION,
    VocabularyLayout,
    expand_vocabulary_response,
    validate_vocabulary_layouts,
    vocabulary_exclusions,
    vocabulary_prompt,
)
from ..core.models import GenerationResult
from ..core.prompt_spec import Candidate, spec_for
from ..core.provider import BaseProvider
from ..core.seeds import bounded_seed_sample, normalize_seed, valid_seed
from ..core.strategies import PromptStrategy


@dataclass
class ChatResult:
    """Backend-neutral chat response."""

    text: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    token_logprobs: List[Tuple[str, float]] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    reasoning_tokens: int = 0
    usage_complete: bool = True
    requested_thinking_mode: str = ""

    def __post_init__(self) -> None:
        """Keep backend telemetry finite, nonnegative, and reasonably bounded."""
        valid_usage = all(
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000_000
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        )
        if valid_usage:
            valid_usage = self.cached_input_tokens <= self.input_tokens
        self.input_tokens = _safe_telemetry_count(self.input_tokens)
        self.cached_input_tokens = min(
            self.input_tokens, _safe_telemetry_count(self.cached_input_tokens)
        )
        self.output_tokens = _safe_telemetry_count(self.output_tokens)
        self.reasoning_tokens = _safe_telemetry_count(self.reasoning_tokens)
        self.usage_complete = self.usage_complete is True and valid_usage
        if self.requested_thinking_mode not in ("default", "disabled"):
            self.requested_thinking_mode = ""


class LLMProvider(BaseProvider):
    """Generate and score candidates once, independent of backend protocol."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model = str(config.get("model", ""))
        self.temperature = _bounded_float("temperature", config.get("temperature", 0.7), 0.0, 2.0)
        self.max_tokens = _bounded_int("max_tokens", config.get("max_tokens", 4000), 1, 131_072)
        self.request_timeout = _bounded_float("timeout", config.get("timeout", 180.0), 0.1, 3_600.0)
        self.max_retries = _bounded_int("max_retries", config.get("max_retries", 2), 0, 10)
        self.confidence_mode = str(config.get("confidence", "model")).lower()
        self.use_context = _as_bool(config.get("context", True))
        self.use_hybrid = _as_bool(config.get("hybrid", False))
        self.hybrid_model_fraction = _bounded_float(
            "hybrid_model_fraction", config.get("hybrid_model_fraction", 0.5), 0.0, 1.0
        )
        self.batch_size = _bounded_int("batch_size", config.get("batch_size", 50), 1, 500)
        self.max_batch_calls = _bounded_int(
            "max_batch_calls", config.get("max_batch_calls", 12), 1, 100
        )
        self.seed_limit = _bounded_int("seed_limit", config.get("seed_limit", 500), 1, 5_000)
        self.seed_char_limit = _bounded_int(
            "seed_char_limit", config.get("seed_char_limit", 50_000), 1_000, 1_000_000
        )
        self.subdomain_thinking = config.get("subdomain_thinking", "disabled")
        if not isinstance(self.subdomain_thinking, str) or self.subdomain_thinking not in {
            "default",
            "disabled",
        }:
            raise ValueError("subdomain_thinking must be 'default' or 'disabled'")
        self._runtime_context = ""
        self._runtime_exclusions: List[str] = []
        self._runtime_batch_limit: Optional[int] = None
        self._runtime_generation_kind: Optional[str] = None
        self._runtime_single_request = False
        self._scan_guard_attached = False

    @abstractmethod
    def chat(self, system: str, user: str, schema: Optional[dict[str, Any]]) -> ChatResult:
        """Send a chat request and return a backend-neutral result."""

    def _chat_for_discovery(
        self, kind: str, system: str, user: str, schema: Optional[dict[str, Any]]
    ) -> ChatResult:
        """Scope request profiles without bypassing an attached chat budget guard.

        Providers already carry per-scan mutable feedback and are used serially.
        Restore the prior scope for nested calls and every exceptional exit.
        """
        if kind not in {"path", "subdomain"}:
            raise ValueError("discovery kind must be 'path' or 'subdomain'")
        previous = self._runtime_generation_kind
        self._runtime_generation_kind = kind
        try:
            return self.chat(system, user, schema)
        finally:
            self._runtime_generation_kind = previous

    def set_runtime_context(self, value: str = "") -> None:
        """Set ephemeral feedback context for the next generation round."""
        self._runtime_context = value.strip()

    def set_runtime_exclusions(self, values: Sequence[str] = ()) -> None:
        """Exclude candidates already spent in earlier feedback rounds."""
        self._runtime_exclusions = list(dict.fromkeys(str(value) for value in values if value))

    def set_runtime_batch_limit(self, limit: Optional[int] = None) -> None:
        """Temporarily cap generation chat calls without changing provider configuration.

        Discovery sessions can request one batch and checkpoint its candidates
        before beginning another paid call. Clear the cap with ``None`` when
        the scoped operation finishes; existing per-request limits still apply.
        """
        self._runtime_batch_limit = (
            None if limit is None else _bounded_int("runtime_batch_limit", limit, 1, 100)
        )

    def generate_paths(
        self,
        known_paths: List[str],
        target: str,
        count: int,
        max_depth: Optional[int] = None,
        strategy: Optional[PromptStrategy] = None,
    ) -> GenerationResult:
        if isinstance(known_paths, (str, bytes)) or not isinstance(known_paths, Sequence):
            raise ValueError("known_paths must be a sequence of seed strings")
        if not all(isinstance(value, str) for value in known_paths):
            raise ValueError("known_paths must contain only seed strings")
        if not isinstance(target, str):
            raise ValueError("target must be a string")
        kind = _kind_from_strategy(strategy)
        spec = spec_for(kind)
        if kind == "subdomain":
            target = target.lower().rstrip(".")
            if not spec.validator(target):
                raise ValueError("subdomain target must be a valid DNS apex")
        if max_depth is not None:
            max_depth = _bounded_int("max_depth", max_depth, 1, 1_000)
        seeds_received = len(known_paths)
        valid_known = list(
            dict.fromkeys(
                normalize_seed(value, kind) for value in known_paths if valid_seed(value, kind)
            )
        )
        seed_scope_rejected = 0
        if kind == "subdomain":
            all_known = [value for value in valid_known if _within_apex(value, target)]
            seed_scope_rejected = len(valid_known) - len(all_known)
        else:
            all_known = valid_known
        prompt_known = bounded_seed_sample(
            all_known,
            limit=self.seed_limit,
            char_limit=self.seed_char_limit,
        )
        if not prompt_known:
            raise ValueError(f"no valid {kind} seeds were supplied")
        context_parts: List[str] = []
        observations: List[str] = []
        if self.use_context:
            bundle = collect_context(prompt_known, kind)
            rendered = bundle.render()
            observations = list(bundle.observations)
            if rendered:
                context_parts.append(rendered)
        if self._runtime_context:
            context_parts.append(
                "Validated feedback from earlier rounds:\n" + self._runtime_context
            )

        requested = _bounded_int("count", count, 1, 5_000)
        known_normalized = {spec.identity(value) for value in all_known}
        runtime_excluded = {spec.identity(value) for value in self._runtime_exclusions}
        accepted: List[tuple[Candidate, float, int, str]] = []
        seen = set(known_normalized) | runtime_excluded
        scope_rejected = 0
        depth_rejected = 0
        known_rejected = 0
        duplicate_rejected = 0
        overflow_rejected = 0
        runtime_rejected = 0
        input_tokens = 0
        cached_input_tokens = 0
        output_tokens = 0
        calls = 0
        empty_calls = 0
        response_model = self.model
        finish_reasons: List[str] = []
        reasoning_tokens = 0
        requested_thinking_modes: List[str] = []
        parse_empty_calls = 0
        adaptive_batch_size = self.batch_size
        if self.use_hybrid:
            for proposal in structural_candidates(
                prompt_known, kind, target=target, count=requested
            ):
                candidate = Candidate(proposal.value, proposal.confidence, proposal.rationale)
                if kind == "path" and max_depth is not None:
                    if _path_depth(candidate.value) > max_depth:
                        depth_rejected += 1
                        continue
                identity = spec.identity(candidate.value)
                if identity in seen:
                    continue
                seen.add(identity)
                accepted.append((candidate, candidate.confidence, len(accepted), "structural"))
        model_accepted = 0
        model_budget = requested
        model_skipped_reason = ""
        if self.use_hybrid:
            if len(accepted) >= requested:
                # The model cannot add a retained candidate unless it displaces
                # an already high-evidence structural proposal. Avoid paying
                # for a batch that normally has no effect at small budgets.
                model_budget = 0
                model_skipped_reason = "structural pool filled candidate budget"
            else:
                model_budget = max(
                    requested - len(accepted),
                    ceil(requested * self.hybrid_model_fraction),
                )
        # The computed minimum fills ordinary large budgets; the hard cap
        # prevents a badly behaved backend from causing unbounded spending.
        call_budget = (
            min(
                self.max_batch_calls,
                max(1, ceil(model_budget / self.batch_size) + 2),
            )
            if model_budget
            else 0
        )
        if self._runtime_batch_limit is not None:
            call_budget = min(call_budget, self._runtime_batch_limit)
        while model_accepted < model_budget and calls < call_budget and empty_calls < 3:
            batch_count = min(adaptive_batch_size, model_budget - model_accepted)
            response = self._chat_for_discovery(
                kind,
                spec.system(),
                spec.user(
                    prompt_known,
                    count=batch_count,
                    target=target,
                    max_depth=max_depth,
                    context="\n\n".join(context_parts),
                    exclude=[
                        *self._runtime_exclusions[-200:],
                        *[item[0].value for item in accepted[-200:]],
                    ][-200:],
                ),
                spec.json_schema(),
            )
            calls += 1
            input_tokens += response.input_tokens
            cached_input_tokens += response.cached_input_tokens
            output_tokens += response.output_tokens
            response_model = response.model or response_model
            finish_reasons.append(response.finish_reason)
            reasoning_tokens += response.reasoning_tokens
            requested_thinking_mode = getattr(response, "requested_thinking_mode", "")
            if (
                isinstance(requested_thinking_mode, str)
                and requested_thinking_mode in {"default", "disabled"}
                and requested_thinking_mode not in requested_thinking_modes
            ):
                requested_thinking_modes.append(requested_thinking_mode)
            parsed = spec.parse(response.text)
            if not parsed:
                parse_empty_calls += 1
            batch_candidates: List[Candidate] = []
            for candidate in parsed:
                if len(batch_candidates) >= model_budget - model_accepted:
                    overflow_rejected += 1
                    continue
                normalized = spec.identity(candidate.value)
                if kind == "path" and max_depth is not None:
                    if _path_depth(candidate.value) > max_depth:
                        depth_rejected += 1
                        continue
                if normalized in known_normalized:
                    known_rejected += 1
                    continue
                if normalized in runtime_excluded:
                    runtime_rejected += 1
                    continue
                if normalized in seen:
                    duplicate_rejected += 1
                    continue
                if kind == "subdomain" and target and not _within_apex(candidate.value, target):
                    scope_rejected += 1
                    continue
                seen.add(normalized)
                batch_candidates.append(candidate)
            batch_scores = self._score(batch_candidates, prompt_known, response.token_logprobs)
            start_index = len(accepted)
            accepted.extend(
                (candidate, score, start_index + index, "model")
                for index, (candidate, score) in enumerate(
                    zip(batch_candidates, batch_scores, strict=True)
                )
            )
            model_accepted += len(batch_candidates)
            if not batch_candidates:
                empty_calls += 1
                # Reasoning models can consume the completion budget before
                # reaching a large requested JSON array. A smaller retry is
                # both cheaper and more likely to produce a final answer.
                adaptive_batch_size = max(5, batch_count // 2)
            else:
                empty_calls = 0
                adaptive_batch_size = self.batch_size

        ranked = sorted(accepted, key=lambda item: (-item[1], item[2]))[:requested]
        candidates = [item[0] for item in ranked]
        scores = [item[1] for item in ranked]
        total_tokens = input_tokens + output_tokens
        return GenerationResult(
            paths=[candidate.value for candidate in candidates],
            confidence_scores=scores,
            metadata={
                "provider": self.get_provider_name(),
                "model": response_model,
                "target": target,
                "kind": kind,
                "seeds_received": seeds_received,
                "seeds_valid": len(valid_known),
                "seeds_in_scope": len(all_known),
                "seed_scope_rejected": seed_scope_rejected,
                "seeds_used": len(prompt_known),
                "seeds_truncated": len(all_known) - len(prompt_known),
                "confidence_mode": self.confidence_mode,
                "hybrid": self.use_hybrid,
                "context_observations": observations,
                "rationales": {candidate.value: candidate.rationale for candidate in candidates},
                "candidate_sources": {item[0].value: item[3] for item in ranked},
                "scope_rejected": scope_rejected,
                "depth_rejected": depth_rejected,
                "known_rejected": known_rejected,
                "duplicate_rejected": duplicate_rejected,
                "overflow_rejected": overflow_rejected,
                "runtime_rejected": runtime_rejected,
                "requested": requested,
                "generated": len(candidates),
                "batch_calls": calls,
                "model_budget": model_budget,
                "model_skipped_reason": model_skipped_reason,
                "parse_empty_calls": parse_empty_calls,
                "finish_reasons": finish_reasons,
                "reasoning_tokens": reasoning_tokens,
                **(
                    {"requested_thinking_modes": requested_thinking_modes}
                    if requested_thinking_modes
                    else {}
                ),
                "fill_rate": len(candidates) / requested,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        )

    def generate_subdomain_vocabulary(
        self,
        layouts: tuple[VocabularyLayout, ...],
        target: str,
        count: int,
        *,
        instructions: str = "",
        exclusions: Sequence[str] = (),
    ) -> GenerationResult:
        """One guarded optional vocabulary request; all output remains hypotheses.

        Selection between this method and ordinary detailed generation happens
        before calling it. Empty/malformed answers never cause a paid retry or
        fallback. In particular, this must call the possibly monkeypatched
        public chat through _chat_for_discovery, not a backend implementation.
        """
        requested = _bounded_int("count", count, 1, 200)
        layouts = validate_vocabulary_layouts(layouts, target)
        explicit = vocabulary_exclusions(exclusions, target)
        runtime = vocabulary_exclusions(self._runtime_exclusions, target)
        excluded = tuple(dict.fromkeys((*explicit, *runtime)))
        system, user, schema = vocabulary_prompt(
            layouts, target, requested, instructions=instructions, exclusions=excluded
        )
        metadata: dict[str, Any] = {
            "provider": self.get_provider_name(),
            "model": self.model,
            "target": target,
            "kind": "subdomain",
            "protocol": "grounded-one-slot-vocabulary",
            "vocabulary_protocol_version": VOCABULARY_PROTOCOL_VERSION,
            "requested": requested,
            "generated": 0,
            "batch_calls": 0,
            "layout_count": len(layouts),
            "verified": False,
            "rationales": {},
            "candidate_sources": {},
            "provenance": {},
            "rejections": {},
            "token_usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }
        if not layouts:
            metadata["model_skipped_reason"] = "no_witnessed_vocabulary_layouts"
            return GenerationResult(paths=[], confidence_scores=[], metadata=metadata)
        previous_single_request = self._runtime_single_request
        self._runtime_single_request = True
        try:
            response = self._chat_for_discovery("subdomain", system, user, schema)
        finally:
            self._runtime_single_request = previous_single_request
        # Capture actual response usage even if every output item is invalid.
        metadata.update(
            {
                "model": response.model or self.model,
                "batch_calls": 1,
                "finish_reasons": [response.finish_reason],
                "reasoning_tokens": response.reasoning_tokens,
                "usage_complete": getattr(response, "usage_complete", True),
                "token_usage": {
                    "input_tokens": response.input_tokens,
                    "cached_input_tokens": response.cached_input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens,
                },
            }
        )
        mode = getattr(response, "requested_thinking_mode", "")
        if isinstance(mode, str) and mode in {"default", "disabled"}:
            metadata["requested_thinking_modes"] = [mode]
        candidates, diagnostics = expand_vocabulary_response(
            response.text, layouts, target, requested, exclusions=excluded
        )
        metadata.update(diagnostics)
        metadata.update(
            {
                "generated": len(candidates),
                "fill_rate": len(candidates) / requested,
                "rationales": {item["value"]: item["reason"] for item in candidates},
                "candidate_sources": {item["value"]: "model-vocabulary" for item in candidates},
                "provenance": {
                    item["value"]: {key: value for key, value in item.items() if key != "value"}
                    for item in candidates
                },
                "confidence_note": "Uniform 0.5 is an uncalibrated ranking placeholder, not existence probability.",
            }
        )
        return GenerationResult(
            paths=[item["value"] for item in candidates],
            confidence_scores=[0.5] * len(candidates),
            metadata=metadata,
        )

    def _make_api_call(self, system_prompt: str, user_prompt: str) -> str:
        """Compatibility hook used by the existing analysis engines."""
        return self.chat(system_prompt, user_prompt, schema=None).text

    def _score(
        self,
        candidates: Sequence[Candidate],
        known: Sequence[str],
        token_logprobs: Sequence[Tuple[str, float]],
    ) -> List[float]:
        if self.confidence_mode == "uniform":
            return [1.0 for _ in candidates]
        if self.confidence_mode == "heuristic":
            return [self._heuristic_confidence(candidate.value, known) for candidate in candidates]
        if self.confidence_mode == "logprob":
            return [
                self._logprob_confidence(candidate.value, token_logprobs)
                for candidate in candidates
            ]
        # Unknown modes intentionally fall back to the safest no-magic option.
        return [candidate.clamp().confidence for candidate in candidates]

    @staticmethod
    def _heuristic_confidence(value: str, known: Sequence[str]) -> float:
        """Legacy similarity heuristic retained only as a benchmark control."""
        score = 0.5
        parts = _segments(value)
        for item in known:
            other = _segments(item)
            union = parts | other
            similarity = len(parts & other) / len(union) if union else 0.0
            score = max(score, 0.3 + similarity * 0.5)
        if any(token in value.lower() for token in ("api", "user", "admin", "auth")):
            score += 0.1
        return min(1.0, score)

    @staticmethod
    def _logprob_confidence(value: str, logprobs: Sequence[Tuple[str, float]]) -> float:
        if not logprobs:
            return 0.5
        rendered = "".join(str(token) for token, _ in logprobs)
        start = rendered.lower().find(value.lower())
        if start < 0:
            return 0.5
        end = start + len(value)
        offset = 0
        relevant: List[float] = []
        for token, logprob in logprobs:
            token_end = offset + len(str(token))
            if token_end > start and offset < end:
                score = float(logprob)
                if math.isfinite(score):
                    relevant.append(score)
            offset = token_end
        if not relevant:
            return 0.5
        # exp(mean(log p)) is the geometric mean probability of the aligned
        # candidate tokens. Log probabilities should be <=0; clamp malformed
        # positive backend values rather than emitting confidence above 1.
        mean_log_probability = min(0.0, sum(relevant) / len(relevant))
        return math.exp(mean_log_probability)


def _kind_from_strategy(strategy: Optional[PromptStrategy]) -> str:
    return (
        "subdomain" if strategy and "subdomain" in strategy.__class__.__name__.lower() else "path"
    )


def _segments(value: str) -> set[str]:
    separator = "/" if value.startswith("/") else "."
    return {part.lower() for part in value.strip("/.").split(separator) if part}


def _path_depth(value: str) -> int:
    return len([part for part in value.split("/") if part])


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError("boolean provider settings must use true or false")


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _safe_telemetry_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    parsed = float(value)
    if not math.isfinite(parsed):
        return 0
    return min(1_000_000_000, max(0, int(parsed)))


def _within_apex(value: str, target: str) -> bool:
    candidate = value.lower().rstrip(".")
    apex = target.lower().rstrip(".")
    return candidate != apex and candidate.endswith("." + apex)
