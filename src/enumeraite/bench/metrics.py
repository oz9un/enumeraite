"""Ground-truth scoring shared by LLM and baseline engines."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from statistics import stdev
from typing import Any, Iterable, List, Mapping, Sequence

from ..core.prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC
from .dataset import BenchCase

_TEMPLATE_SEGMENT = re.compile(r"^(?:\{[^{}]+\}|:[^/]+|<[^<>]+>)$")


@dataclass
class BenchMetrics:
    target: str
    kind: str
    generated: int
    valid: int
    recall_at_budget: dict[int, float]
    precision_at_budget: dict[int, float]
    precision: float
    average_precision: float
    reciprocal_rank: float
    brier_score: float
    fill_rate: float
    novel_valid: int
    novel_valid_items: List[str] = field(default_factory=list)
    wordlist_unique: int = 0
    wordlist_unique_items: List[str] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)
    cost_per_valid: dict[str, float | None] = field(default_factory=dict)
    ranked_candidates: List[dict[str, Any]] = field(default_factory=list)
    generation_meta: dict[str, Any] = field(default_factory=dict)
    case_meta: dict[str, Any] = field(default_factory=dict)
    valid_by_source: dict[str, int] = field(default_factory=dict)
    unique_valid_by_source: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _path_matches_template(candidate: str, truth: str) -> bool:
    if candidate.rstrip("/") == "" or truth.rstrip("/") == "":
        return candidate.rstrip("/") == truth.rstrip("/")
    candidate_parts = candidate.strip("/").split("/")
    truth_parts = truth.strip("/").split("/")
    if len(candidate_parts) != len(truth_parts):
        return False
    return all(
        _segment_matches(candidate_part, truth_part)
        for candidate_part, truth_part in zip(candidate_parts, truth_parts, strict=True)
    )


def _segment_matches(candidate: str, truth: str) -> bool:
    template = _TEMPLATE_SEGMENT.fullmatch(truth)
    if not template:
        return bool(candidate) and candidate == truth
    parameter_name = truth.strip("{}<>").lstrip(":").lower()
    if "id" in parameter_name:
        return _looks_like_identifier(candidate)
    return bool(candidate) and candidate.lower() not in _RESERVED_PARAMETER_WORDS


_RESERVED_PARAMETER_WORDS = {
    "add",
    "admin",
    "all",
    "config",
    "count",
    "create",
    "delete",
    "export",
    "get",
    "history",
    "list",
    "login",
    "logout",
    "meta",
    "register",
    "search",
    "settings",
    "signup",
    "status",
    "update",
}


def _looks_like_identifier(value: str) -> bool:
    """Conservatively reject action/resource words as `{...Id}` fills."""
    return bool(
        re.fullmatch(r"\d+", value)
        or re.fullmatch(r"[0-9a-f]{8,}", value, re.IGNORECASE)
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            value,
            re.IGNORECASE,
        )
        or (len(value) >= 6 and re.search(r"[a-z]", value, re.I) and re.search(r"\d", value))
    )


class _TruthMatcher:
    """Index truth by exact value and path depth for repeated scoring lookups."""

    def __init__(self, truth_items: Sequence[str], kind: str):
        self.kind = kind
        self.exact = set(truth_items)
        self.templates_by_depth: dict[int, list[str]] = {}
        self.literal_ids: dict[int, list[dict[str, set[int]]]] = {}
        self.general_parameter_ids: dict[int, list[set[int]]] = {}
        self.identifier_parameter_ids: dict[int, list[set[int]]] = {}
        if kind == "path":
            for truth in truth_items:
                parts = truth.strip("/").split("/") if truth.rstrip("/") else []
                if any(_TEMPLATE_SEGMENT.fullmatch(part) for part in parts):
                    self.templates_by_depth.setdefault(len(parts), []).append(truth)
            for depth, templates in self.templates_by_depth.items():
                self.literal_ids[depth] = [dict() for _ in range(depth)]
                self.general_parameter_ids[depth] = [set() for _ in range(depth)]
                self.identifier_parameter_ids[depth] = [set() for _ in range(depth)]
                for template_id, truth in enumerate(templates):
                    for index, part in enumerate(truth.strip("/").split("/")):
                        if _TEMPLATE_SEGMENT.fullmatch(part):
                            parameter_name = part.strip("{}<>").lstrip(":").lower()
                            destination = (
                                self.identifier_parameter_ids
                                if "id" in parameter_name
                                else self.general_parameter_ids
                            )
                            destination[depth][index].add(template_id)
                        else:
                            self.literal_ids[depth][index].setdefault(part, set()).add(template_id)

    def matches(self, candidate: str) -> List[str]:
        if candidate in self.exact:
            return [candidate]
        if self.kind != "path":
            return []
        parts = candidate.strip("/").split("/") if candidate.rstrip("/") else []
        depth = len(parts)
        templates = self.templates_by_depth.get(depth, ())
        if not templates:
            return []
        possible = set(range(len(templates)))
        for index, part in enumerate(parts):
            allowed = set(self.literal_ids[depth][index].get(part, ()))
            if _looks_like_identifier(part):
                allowed.update(self.identifier_parameter_ids[depth][index])
            if part and part.lower() not in _RESERVED_PARAMETER_WORDS:
                allowed.update(self.general_parameter_ids[depth][index])
            possible.intersection_update(allowed)
            if not possible:
                return []
        # Preserve the original truth ordering for equal-specificity aliases.
        best_id = max(sorted(possible), key=lambda item: _template_specificity(templates[item]))
        return [templates[best_id]]


def score_run(
    case: BenchCase,
    candidates: Sequence[str],
    confidences: Sequence[float] | None = None,
    budgets: Sequence[int] = (25, 50, 100, 250, 500),
    baseline_wordlist: Iterable[str] = (),
    cost: Mapping[str, float] | None = None,
    requested: int | None = None,
    trace_limit: int = 50,
    generation_meta: Mapping[str, Any] | None = None,
) -> BenchMetrics:
    """Score one ranked generation run against hidden ground truth."""
    normalized_budgets = tuple(
        _strict_positive_int("budget", value, maximum=5_000) for value in budgets
    )
    if not normalized_budgets:
        raise ValueError("at least one candidate budget is required")
    if requested is not None:
        requested = _strict_positive_int("requested candidate budget", requested, maximum=5_000)
    if (
        isinstance(trace_limit, bool)
        or not isinstance(trace_limit, int)
        or not 0 <= trace_limit <= 5_000
    ):
        raise ValueError("trace_limit must be an integer between 0 and 5000")
    if isinstance(candidates, (str, bytes)) or isinstance(confidences, (str, bytes)):
        raise ValueError("candidates and confidences must be arrays")
    scores = list(confidences or [])
    normalized_candidates: List[tuple[str, float, int]] = []
    known_matcher = _TruthMatcher(case.known, case.kind)
    hidden_matcher = _TruthMatcher(case.hidden, case.kind)
    universe_matcher = _TruthMatcher(case.universe, case.kind)
    hidden_matches_by_candidate: dict[str, List[str]] = {}
    valid_candidate: dict[str, bool] = {}
    spec = PATH_SPEC if case.kind == "path" else SUBDOMAIN_SPEC
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if requested is not None and index >= requested:
            break
        value = _normalize(str(candidate), case.kind)
        if not value or value in seen:
            continue
        is_valid = spec.validator(value)
        # Repeating a visible seed is not discovery and must never improve a
        # provider's precision or ranking metrics.
        if is_valid and known_matcher.matches(value):
            continue
        seen.add(value)
        score = _safe_probability(scores[index] if index < len(scores) else 0.0)
        normalized_candidates.append((value, score, index))
        valid_candidate[value] = is_valid
        hidden_matches_by_candidate[value] = hidden_matcher.matches(value) if is_valid else []
    ranked = sorted(normalized_candidates, key=lambda item: (-item[1], item[2]))

    hidden = list(case.hidden)
    matched_universe: dict[str, str] = {}
    matched_hidden: dict[str, str] = {}
    source_map_raw = (generation_meta or {}).get("candidate_sources", {})
    source_map = (
        {
            _normalize(str(candidate), case.kind): (
                str(source).lower() if str(source).lower() in {"model", "structural"} else "other"
            )
            for candidate, source in source_map_raw.items()
        }
        if isinstance(source_map_raw, dict)
        else {}
    )
    covered_by_source: dict[str, set[str]] = {}
    for candidate, _, _ in ranked:
        matches = universe_matcher.matches(candidate) if valid_candidate[candidate] else []
        if matches:
            for truth in matches:
                matched_universe.setdefault(truth, candidate)
        candidate_hidden = hidden_matches_by_candidate[candidate]
        for truth in candidate_hidden:
            matched_hidden.setdefault(truth, candidate)
        source = source_map.get(candidate, "")
        if source:
            covered_by_source.setdefault(source, set()).update(candidate_hidden)

    valid_by_source = {
        source: len(covered) for source, covered in sorted(covered_by_source.items())
    }
    unique_valid_by_source = {
        source: len(
            covered
            - set().union(*(other for name, other in covered_by_source.items() if name != source))
        )
        for source, covered in sorted(covered_by_source.items())
    }

    recall: dict[int, float] = {}
    precision_at_budget: dict[int, float] = {}
    for budget in normalized_budgets:
        covered: set[str] = set()
        for candidate, _, _ in ranked[: int(budget)]:
            covered.update(hidden_matches_by_candidate[candidate])
        recall[int(budget)] = len(covered) / len(hidden) if hidden else 0.0
        denominator = min(int(budget), len(ranked))
        precision_at_budget[int(budget)] = len(covered) / denominator if denominator else 0.0

    wordlist = {
        (
            str(item).strip().strip("/.")
            if case.kind == "path"
            else str(item).strip().lower().strip("/.")
        )
        for item in baseline_wordlist
        if str(item).strip()
    }
    known = set(case.known)
    wordlist_unique_items: List[str] = []
    novel_items: List[str] = []
    for truth, candidate in matched_universe.items():
        is_wordlist_unique = not (_wordlist_forms(case, candidate) & wordlist)
        if is_wordlist_unique:
            wordlist_unique_items.append(truth)
            if truth not in known:
                novel_items.append(truth)

    # Ground truth is a set of endpoint shapes. Two concrete IDs satisfying
    # one templated route count once, otherwise a model can inflate precision
    # by emitting `/pets/1`, `/pets/2`, ... for the same route.
    valid_count = len(matched_hidden)
    cumulative_covered: set[str] = set()
    positive_ranks: List[int] = []
    trace: List[dict[str, Any]] = []
    squared_errors: List[float] = []
    for rank, (candidate, confidence, _) in enumerate(ranked, start=1):
        matches = hidden_matches_by_candidate[candidate]
        newly_covered = [truth for truth in matches if truth not in cumulative_covered]
        is_positive = bool(newly_covered)
        cumulative_covered.update(newly_covered)
        if is_positive:
            positive_ranks.append(rank)
        probability = min(1.0, max(0.0, confidence if math.isfinite(confidence) else 0.0))
        squared_errors.append((probability - float(is_positive)) ** 2)
        if len(trace) < max(0, trace_limit):
            trace.append(
                {
                    "rank": rank,
                    "value": candidate,
                    "confidence": confidence,
                    "matches": newly_covered,
                }
            )
    precision_sum = sum(index / rank for index, rank in enumerate(positive_ranks, start=1))
    average_precision = precision_sum / len(hidden) if hidden else 0.0
    cost_values = {str(key): _safe_nonnegative(value) for key, value in (cost or {}).items()}
    per_valid = {
        key: (value / valid_count if valid_count else None) for key, value in cost_values.items()
    }
    return BenchMetrics(
        target=case.target,
        kind=case.kind,
        generated=len(ranked),
        valid=valid_count,
        recall_at_budget=recall,
        precision_at_budget=precision_at_budget,
        precision=(valid_count / len(ranked) if ranked else 0.0),
        average_precision=average_precision,
        reciprocal_rank=(1.0 / positive_ranks[0] if positive_ranks else 0.0),
        brier_score=(sum(squared_errors) / len(squared_errors) if squared_errors else 0.0),
        fill_rate=(len(ranked) / requested if requested else 1.0),
        novel_valid=len(novel_items),
        novel_valid_items=sorted(novel_items),
        wordlist_unique=len(wordlist_unique_items),
        wordlist_unique_items=sorted(wordlist_unique_items),
        cost=cost_values,
        cost_per_valid=per_valid,
        ranked_candidates=trace,
        generation_meta=dict(generation_meta or {}),
        case_meta=dict(case.meta),
        valid_by_source=valid_by_source,
        unique_valid_by_source=unique_valid_by_source,
    )


def _safe_probability(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return min(1.0, max(0.0, parsed)) if math.isfinite(parsed) else 0.0


def _strict_positive_int(name: str, value: Any, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value


def _safe_nonnegative(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, parsed) if math.isfinite(parsed) else 0.0


def aggregate(metrics: Sequence[BenchMetrics], budgets: Sequence[int]) -> dict[str, Any]:
    # Provider failures are part of benchmark quality. They contribute zero to
    # recall, ranking, and fill instead of disappearing from macro averages.
    # Calibration is computed only when a provider actually emitted scores.
    all_metrics = list(metrics)
    completed = [metric for metric in all_metrics if not metric.error]
    calibrated = [metric for metric in completed if metric.generated > 0]
    total_generated = sum(metric.generated for metric in all_metrics)
    total_valid = sum(metric.valid for metric in all_metrics)
    return {
        "cases": len(all_metrics),
        "successful_cases": len(completed),
        "calibrated_cases": len(calibrated),
        "errors": sum(1 for metric in all_metrics if metric.error),
        "macro_recall_at_budget": {
            int(budget): (
                sum(metric.recall_at_budget.get(int(budget), 0.0) for metric in all_metrics)
                / len(all_metrics)
                if all_metrics
                else 0.0
            )
            for budget in budgets
        },
        "macro_recall_ci95": {
            int(budget): _mean_ci95(_clustered_recall(all_metrics, int(budget)))
            for budget in budgets
        },
        "ci_clusters": len({_cluster_key(metric) for metric in all_metrics}),
        "micro_precision": total_valid / total_generated if total_generated else 0.0,
        "macro_average_precision": (
            sum(metric.average_precision for metric in all_metrics) / len(all_metrics)
            if all_metrics
            else 0.0
        ),
        "macro_reciprocal_rank": (
            sum(metric.reciprocal_rank for metric in all_metrics) / len(all_metrics)
            if all_metrics
            else 0.0
        ),
        "mean_brier_score": (
            sum(metric.brier_score for metric in calibrated) / len(calibrated)
            if calibrated
            else 0.0
        ),
        "mean_fill_rate": (
            sum(metric.fill_rate for metric in all_metrics) / len(all_metrics)
            if all_metrics
            else 0.0
        ),
        "generated": total_generated,
        "valid": total_valid,
        "novel_valid": sum(metric.novel_valid for metric in all_metrics),
        "wordlist_unique": sum(metric.wordlist_unique for metric in all_metrics),
        "valid_by_source": {
            source: sum(metric.valid_by_source.get(source, 0) for metric in all_metrics)
            for source in sorted(
                {source for metric in all_metrics for source in metric.valid_by_source}
            )
        },
        "unique_valid_by_source": {
            source: sum(metric.unique_valid_by_source.get(source, 0) for metric in all_metrics)
            for source in sorted(
                {source for metric in all_metrics for source in metric.unique_valid_by_source}
            )
        },
        "cost": {
            key: sum(metric.cost.get(key, 0.0) for metric in all_metrics)
            for key in sorted({key for metric in all_metrics for key in metric.cost})
        },
    }


def _cluster_key(metric: BenchMetrics) -> str:
    # Content identity prevents a copied specification under another path from
    # masquerading as an independent target in uncertainty estimates.
    return str(
        metric.case_meta.get("source_sha256") or metric.case_meta.get("source") or metric.target
    )


def _clustered_recall(metrics: Sequence[BenchMetrics], budget: int) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for metric in metrics:
        grouped.setdefault(_cluster_key(metric), []).append(
            metric.recall_at_budget.get(budget, 0.0)
        )
    return [sum(values) / len(values) for values in grouped.values() if values]


def _paired_recall_delta_ci95(
    current: Sequence[BenchMetrics], baseline: Sequence[BenchMetrics], budget: int
) -> tuple[list[float], int]:
    """Return a source-clustered paired CI for recall improvement."""
    if len(current) != len(baseline):
        raise ValueError("paired benchmark engines must have the same cases")
    grouped: dict[str, list[float]] = {}
    for current_metric, baseline_metric in zip(current, baseline, strict=True):
        current_key = _cluster_key(current_metric)
        baseline_key = _cluster_key(baseline_metric)
        if (
            current_key != baseline_key
            or current_metric.target != baseline_metric.target
            or current_metric.kind != baseline_metric.kind
            or current_metric.case_meta != baseline_metric.case_meta
        ):
            raise ValueError("paired benchmark case ordering differs between engines")
        grouped.setdefault(current_key, []).append(
            current_metric.recall_at_budget.get(budget, 0.0)
            - baseline_metric.recall_at_budget.get(budget, 0.0)
        )
    cluster_deltas = [sum(values) / len(values) for values in grouped.values() if values]
    return _mean_ci95_unbounded(cluster_deltas), len(cluster_deltas)


def _mean_ci95(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    # The benchmark suite is intentionally small and its population variance
    # is unknown. A Student-t interval avoids the anti-conservative normal
    # approximation for those small samples.
    margin = _t_critical_95(len(values) - 1) * stdev(values) / math.sqrt(len(values))
    return [max(0.0, mean - margin), min(1.0, mean + margin)]


def _mean_ci95_unbounded(values: Sequence[float]) -> list[float]:
    """Student-t interval for a signed difference, bounded to feasible deltas."""
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    margin = _t_critical_95(len(values) - 1) * stdev(values) / math.sqrt(len(values))
    return [max(-1.0, mean - margin), min(1.0, mean + margin)]


_T_CRITICAL_95 = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


def _t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < len(_T_CRITICAL_95):
        return _T_CRITICAL_95[max(1, degrees_of_freedom)]
    # Cornish-Fisher approximation around z=.975; accurate enough once df>30.
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def _matches(candidate: str, truth_items: Sequence[str], kind: str) -> List[str]:
    return _TruthMatcher(truth_items, kind).matches(candidate)


def _template_specificity(value: str) -> tuple[int, int]:
    parts = value.strip("/").split("/")
    literals = sum(not _TEMPLATE_SEGMENT.fullmatch(part) for part in parts)
    return literals, len(value)


def _normalize(value: str, kind: str) -> str:
    value = value.strip()
    return (value.rstrip("/") or "/") if kind == "path" else value.lower().rstrip(".")


def _wordlist_forms(case: BenchCase, candidate: str) -> set[str]:
    if case.kind == "path":
        relative = candidate.strip("/")
        last = relative.rsplit("/", 1)[-1]
        return {relative, last}
    value = candidate.lower().rstrip(".")
    apex = case.target.lower().rstrip(".")
    if apex and value.endswith("." + apex):
        relative = value[: -(len(apex) + 1)]
    else:
        relative = value.split(".")[0]
    return {relative, relative.split(".")[0]}
