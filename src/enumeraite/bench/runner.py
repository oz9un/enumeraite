"""Benchmark execution and report generation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import subprocess  # nosec B404
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Union, cast

from .. import __version__
from ..core.bounded_io import read_bounded_regular_file
from ..core.provider import BaseProvider
from ..core.strategies import PromptStrategy
from ..core.text_safety import safe_exception_text, safe_model_text
from .dataset import BenchCase
from .metrics import BenchMetrics, _paired_recall_delta_ci95, aggregate, score_run

Engine = Callable[
    [Sequence[str], int, str],
    Union[
        tuple[Sequence[str], Sequence[float], Mapping[str, float]],
        tuple[
            Sequence[str],
            Sequence[float],
            Mapping[str, float],
            Mapping[str, Any],
        ],
    ],
]

_GENERATION_META_KEYS = {
    "provider",
    "model",
    "kind",
    "seeds_received",
    "seeds_valid",
    "seeds_in_scope",
    "seeds_used",
    "seeds_truncated",
    "seed_scope_rejected",
    "confidence_mode",
    "hybrid",
    "context_observations",
    "candidate_sources",
    "scope_rejected",
    "depth_rejected",
    "known_rejected",
    "duplicate_rejected",
    "overflow_rejected",
    "runtime_rejected",
    "requested",
    "generated",
    "batch_calls",
    "model_budget",
    "model_skipped_reason",
    "parse_empty_calls",
    "finish_reasons",
    "reasoning_tokens",
    "fill_rate",
}

_MAX_REPLAY_BYTES = 256 * 1024 * 1024


class _ReplayProviderFailure(Exception):
    def __init__(self, message: str, cost: Mapping[str, Any]):
        super().__init__(message)
        self.original_error = safe_model_text(message, 2_000)
        self.cost = {str(key): _safe_nonnegative(value) for key, value in cost.items()}


DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # Input/output USD per 1M tokens. This is editable benchmark metadata, not
    # a billing source of truth. Unknown and local models default to zero.
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-6": (3.00, 15.00),
    "deepseek-chat": (0.27, 1.10),
    "MiniMax-M3": (0.30, 1.20),
    "MiniMax-M2.7": (0.30, 1.20),
    "MiniMax-M2.7-highspeed": (0.60, 2.40),
}
DEFAULT_CACHE_READ_PRICING: dict[str, float] = {
    "MiniMax-M3": 0.06,
    "MiniMax-M2.7": 0.06,
    "MiniMax-M2.7-highspeed": 0.06,
}
PRICING_SOURCE_URL = "https://platform.minimax.io/subscribe/token-plan?tab=api-enterprise"
PRICING_CHECKED_AT = "2026-09-02"
METRIC_VERSION = "route-shape-v10"


@dataclass
class RunConfig:
    count: int = 200
    budgets: tuple[int, ...] = (25, 50, 100, 200)
    baseline_wordlist: Sequence[str] = field(default_factory=tuple)
    pricing: Mapping[str, tuple[float, float]] = field(default_factory=lambda: DEFAULT_PRICING)
    cache_read_pricing: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_CACHE_READ_PRICING
    )
    out_dir: str | Path = "bench-results"
    verbose: bool = False
    experiment: Mapping[str, Any] = field(default_factory=dict)
    engine_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def make_llm_engine(
    provider: BaseProvider,
    strategy: PromptStrategy,
    pricing: Mapping[str, tuple[float, float]] | None = None,
    cache_read_pricing: Mapping[str, float] | None = None,
) -> Engine:
    price_table = pricing or DEFAULT_PRICING
    cache_price_table = cache_read_pricing or DEFAULT_CACHE_READ_PRICING
    _validate_pricing(price_table, cache_price_table)

    def run(known: Sequence[str], count: int, target: str):
        start = time.perf_counter()
        result = provider.generate_paths(list(known), target, count, strategy=strategy)
        elapsed = time.perf_counter() - start
        usage_value = result.metadata.get("token_usage", {})
        usage = usage_value if isinstance(usage_value, Mapping) else {}
        input_tokens = _safe_count(usage.get("input_tokens", 0))
        cached_input_tokens = _safe_count(usage.get("cached_input_tokens", 0))
        output_tokens = _safe_count(usage.get("output_tokens", 0))
        reasoning_tokens = _safe_count(result.metadata.get("reasoning_tokens", 0))
        batch_calls = _safe_count(result.metadata.get("batch_calls", 0))
        parse_empty_calls = _safe_count(result.metadata.get("parse_empty_calls", 0))
        candidate_sources = result.metadata.get("candidate_sources", {})
        source_values = (
            list(candidate_sources.values()) if isinstance(candidate_sources, dict) else []
        )
        model_candidates = sum(source == "model" for source in source_values)
        structural_candidates = sum(source == "structural" for source in source_values)
        model = str(result.metadata.get("model", ""))
        input_price, output_price = price_table.get(model, (0.0, 0.0))
        cached_input_tokens = min(input_tokens, max(0, cached_input_tokens))
        new_input_tokens = input_tokens - cached_input_tokens
        cache_read_price = cache_price_table.get(model, input_price)
        usd = (
            new_input_tokens / 1_000_000 * input_price
            + cached_input_tokens / 1_000_000 * cache_read_price
            + output_tokens / 1_000_000 * output_price
        )
        generation_meta = {
            key: value for key, value in result.metadata.items() if key in _GENERATION_META_KEYS
        }
        return (
            result.paths,
            result.confidence_scores,
            {
                "input_tokens": float(input_tokens),
                "cached_input_tokens": float(cached_input_tokens),
                "new_input_tokens": float(new_input_tokens),
                "output_tokens": float(output_tokens),
                "reasoning_tokens": float(reasoning_tokens),
                "batch_calls": float(batch_calls),
                "parse_empty_calls": float(parse_empty_calls),
                "model_candidates": float(model_candidates),
                "structural_candidates": float(structural_candidates),
                "usd": usd,
                "seconds": elapsed,
            },
            generation_meta,
        )

    return run


def make_baseline_engine(
    gen_fn: Callable[[Sequence[str], int, str], Sequence[str]],
) -> Engine:
    def run(known: Sequence[str], count: int, target: str):
        start = time.perf_counter()
        candidates = list(gen_fn(known, count, target))
        return (
            candidates,
            [1.0] * len(candidates),
            {"seconds": time.perf_counter() - start},
        )

    return run


class BenchRunner:
    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        if (
            isinstance(self.config.count, bool)
            or not isinstance(self.config.count, int)
            or not 1 <= self.config.count <= 5_000
        ):
            raise ValueError("benchmark count must be between 1 and 5000")
        if not self.config.budgets or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5_000
            for value in self.config.budgets
        ):
            raise ValueError("benchmark budgets must be integers between 1 and 5000")
        if len(set(self.config.budgets)) != len(self.config.budgets):
            raise ValueError("benchmark budgets must not contain duplicates")
        _validate_pricing(self.config.pricing, self.config.cache_read_pricing)

    def run(self, dataset: Iterable[BenchCase], engines: Mapping[str, Engine]) -> dict[str, Any]:
        cases = list(dataset)
        per_engine: dict[str, dict[str, Any]] = {}
        metrics_by_engine: dict[str, list[BenchMetrics]] = {}
        for engine_name, engine in engines.items():
            metrics: list[BenchMetrics] = []
            for case in cases:
                case_started = time.perf_counter()
                try:
                    engine_result = engine(case.known, self.config.count, case.target)
                    if len(engine_result) == 4:
                        candidates, confidences, cost, generation_meta = engine_result
                    else:
                        candidates, confidences, cost = engine_result
                        generation_meta = {}
                    if isinstance(candidates, (str, bytes)) or isinstance(
                        confidences, (str, bytes)
                    ):
                        raise ValueError(
                            "benchmark engine candidates and confidences must be arrays"
                        )
                    # Apply the experiment budget at the raw emission boundary.
                    # Otherwise an engine could return an unbounded tail and let
                    # those extra candidates compete during confidence ranking.
                    bounded_candidates = list(islice(candidates, self.config.count))
                    bounded_confidences = list(islice(confidences, self.config.count))
                    measured_cost = dict(cost)
                    measured_cost.setdefault("seconds", time.perf_counter() - case_started)
                    metric = score_run(
                        case,
                        bounded_candidates,
                        bounded_confidences,
                        budgets=self.config.budgets,
                        baseline_wordlist=self.config.baseline_wordlist,
                        cost=measured_cost,
                        requested=self.config.count,
                        trace_limit=self.config.count,
                        generation_meta=generation_meta,
                    )
                except _ReplayProviderFailure as exc:
                    metric = _failed_metric(
                        case,
                        self.config.budgets,
                        exc.original_error,
                        exc.cost,
                    )
                except Exception as exc:  # One provider/case must not abort the run.
                    metric = BenchMetrics(
                        target=case.target,
                        kind=case.kind,
                        generated=0,
                        valid=0,
                        recall_at_budget={budget: 0.0 for budget in self.config.budgets},
                        precision_at_budget={budget: 0.0 for budget in self.config.budgets},
                        precision=0.0,
                        average_precision=0.0,
                        reciprocal_rank=0.0,
                        brier_score=0.0,
                        fill_rate=0.0,
                        novel_valid=0,
                        cost={"seconds": time.perf_counter() - case_started},
                        cost_per_valid={"seconds": None},
                        case_meta=dict(case.meta),
                        error=safe_exception_text(f"{type(exc).__name__}: {exc}", 2_000),
                    )
                metrics.append(metric)
            per_engine[engine_name] = {
                "metadata": dict(self.config.engine_metadata.get(engine_name, {})),
                "aggregate": aggregate(metrics, self.config.budgets),
                "cases": [metric.to_dict() for metric in metrics],
            }
            metrics_by_engine[engine_name] = metrics

        mutation = per_engine.get("mutation", {}).get("aggregate")
        if mutation:
            largest = max(self.config.budgets) if self.config.budgets else 0
            baseline_recall = mutation["macro_recall_at_budget"].get(largest, 0.0)
            mutation_metrics = metrics_by_engine["mutation"]
            for name, payload in per_engine.items():
                if name == "mutation":
                    continue
                current = payload["aggregate"]
                delta_ci, delta_clusters = _paired_recall_delta_ci95(
                    metrics_by_engine[name], mutation_metrics, largest
                )
                current["comparison_to_mutation"] = {
                    "valid_delta": current["valid"] - mutation["valid"],
                    "precision_delta": current["micro_precision"] - mutation["micro_precision"],
                    "recall_delta": current["macro_recall_at_budget"].get(largest, 0.0)
                    - baseline_recall,
                    "recall_delta_ci95": delta_ci,
                    "recall_budget": largest,
                    "paired_clusters": delta_clusters,
                }

        result = {
            "schema_version": 2,
            "metric_version": METRIC_VERSION,
            "config": {
                **dict(self.config.experiment),
                "count": self.config.count,
                "budgets": list(self.config.budgets),
                "pricing_usd_per_million_tokens": {
                    model: {
                        "input": prices[0],
                        "output": prices[1],
                        "cache_read": self.config.cache_read_pricing.get(model, prices[0]),
                    }
                    for model, prices in sorted(self.config.pricing.items())
                },
                "pricing_source": PRICING_SOURCE_URL,
                "pricing_checked_at": PRICING_CHECKED_AT,
            },
            "runtime": _runtime_metadata(),
            "dataset_cases": len(cases),
            "dataset": [
                {
                    "target": case.target,
                    "kind": case.kind,
                    "known": case.known,
                    "hidden": case.hidden,
                    "universe": case.universe,
                    "meta": case.meta,
                }
                for case in cases
            ],
            "engines": per_engine,
        }
        self._write(result)
        return result

    def _write(self, result: dict[str, Any]) -> None:
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")


def replay_benchmark(
    source_path: str | Path,
    out_dir: str | Path,
    baseline_wordlist: Sequence[str] = (),
    wordlist_sha256: str | None = None,
    *,
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Rescore a complete saved trace without invoking a generation backend."""
    if source_bytes is not None and not isinstance(source_bytes, bytes):
        raise ValueError("source_bytes must be bytes")
    if source_bytes is not None and len(source_bytes) > _MAX_REPLAY_BYTES:
        raise ValueError("benchmark result exceeds the 256 MiB replay limit")
    raw = (
        source_bytes
        if source_bytes is not None
        else read_bounded_regular_file(
            source_path,
            _MAX_REPLAY_BYTES,
            not_regular_message="benchmark result must be a regular file",
            too_large_message="benchmark result exceeds the 256 MiB replay limit",
        )
    )
    try:
        source = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark JSON: {exc}") from exc
    if not isinstance(source, dict):
        raise ValueError("benchmark result must be a JSON object")
    dataset_payload = source.get("dataset")
    engines_payload = source.get("engines")
    config_payload = source.get("config")
    if not isinstance(dataset_payload, list) or not isinstance(engines_payload, dict):
        raise ValueError("benchmark result is missing its dataset or engine traces")
    if not isinstance(config_payload, dict):
        raise ValueError("benchmark result is missing its config")
    dataset_cases = source.get("dataset_cases")
    if (
        isinstance(dataset_cases, bool)
        or not isinstance(dataset_cases, int)
        or dataset_cases != len(dataset_payload)
    ):
        raise ValueError("benchmark result dataset case count differs from its manifest")

    expected_wordlist_hash = config_payload.get("wordlist_sha256")
    if expected_wordlist_hash is not None:
        if not isinstance(expected_wordlist_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_wordlist_hash, re.IGNORECASE
        ):
            raise ValueError("benchmark result has an invalid wordlist SHA-256")
        if wordlist_sha256 is None:
            raise ValueError("the source run requires its original wordlist and SHA-256")
        if wordlist_sha256.lower() != expected_wordlist_hash.lower():
            raise ValueError("the supplied wordlist SHA-256 does not match the source run")

    cases: list[BenchCase] = []
    for item in dataset_payload:
        if not isinstance(item, dict):
            raise ValueError("benchmark dataset contains a malformed case")
        try:
            meta = item.get("meta", {})
            if not isinstance(meta, dict):
                raise TypeError("meta must be an object")
            cases.append(
                BenchCase(
                    target=str(item["target"]),
                    kind=cast(Any, str(item["kind"])),
                    known=_string_list(item["known"]),
                    hidden=_string_list(item["hidden"]),
                    universe=_string_list(item["universe"]),
                    meta=dict(meta),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"benchmark dataset contains a malformed case: {exc}") from exc

    engines: dict[str, Engine] = {}
    engine_metadata: dict[str, Mapping[str, Any]] = {}
    for engine_name, engine_payload in engines_payload.items():
        if not isinstance(engine_payload, dict) or not isinstance(
            engine_payload.get("cases"), list
        ):
            raise ValueError(f"engine {engine_name!r} is missing case traces")
        records = cast(list[Mapping[str, Any]], engine_payload["cases"])
        if len(records) != len(cases):
            raise ValueError(f"engine {engine_name!r} case count differs from the dataset")
        _validate_replay_records(str(engine_name), records, cases)
        engines[str(engine_name)] = _replay_engine(str(engine_name), records)
        metadata = engine_payload.get("metadata", {})
        engine_metadata[str(engine_name)] = metadata if isinstance(metadata, dict) else {}

    raw_budgets = config_payload.get("budgets", [])
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("benchmark result has invalid candidate budgets")
    budgets = tuple(
        _strict_replay_int(value, "candidate budget", minimum=1) for value in raw_budgets
    )
    count = _strict_replay_int(
        config_payload.get("count", 0), "candidate count", minimum=1, maximum=5_000
    )
    if not 1 <= count <= 5_000:  # Defensive if the helper contract changes.
        raise ValueError("benchmark result has an invalid candidate count")
    experiment = dict(config_payload)
    experiment.pop("count", None)
    experiment.pop("budgets", None)
    experiment.update(
        {
            "replayed_from": str(Path(source_path).resolve()),
            "source_results_sha256": hashlib.sha256(raw).hexdigest(),
            "source_metric_version": source.get("metric_version", "unknown"),
        }
    )
    if wordlist_sha256 is not None:
        experiment["wordlist_sha256"] = wordlist_sha256
    return BenchRunner(
        RunConfig(
            count=count,
            budgets=budgets,
            baseline_wordlist=baseline_wordlist,
            out_dir=out_dir,
            experiment=experiment,
            engine_metadata=engine_metadata,
        )
    ).run(cases, engines)


def _replay_engine(engine_name: str, records: Sequence[Mapping[str, Any]]) -> Engine:
    index = 0

    def run(known: Sequence[str], count: int, target: str):
        del known, count
        nonlocal index
        if index >= len(records):
            raise ValueError(f"engine {engine_name!r} replay exhausted its traces")
        record = records[index]
        index += 1
        if str(record.get("target", "")) != target:
            raise ValueError(f"engine {engine_name!r} replay case ordering differs")
        if record.get("error"):
            cost = record.get("cost", {})
            raise _ReplayProviderFailure(
                str(record["error"]), cost if isinstance(cost, dict) else {}
            )
        trace = record.get("ranked_candidates", [])
        if not isinstance(trace, list):
            raise ValueError(f"engine {engine_name!r} has an invalid candidate trace")
        generated = int(record.get("generated", len(trace)))
        if len(trace) < generated:
            raise ValueError(
                f"engine {engine_name!r} trace is incomplete ({len(trace)}/{generated}); "
                "the source run cannot be replayed exactly"
            )
        candidate_records = [item for item in trace if isinstance(item, dict)]
        candidates = [str(item["value"]) for item in candidate_records]
        confidences = [float(item.get("confidence", 0.0)) for item in candidate_records]
        cost = record.get("cost", {})
        generation_meta = record.get("generation_meta", {})
        return (
            candidates,
            confidences,
            cost if isinstance(cost, dict) else {},
            generation_meta if isinstance(generation_meta, dict) else {},
        )

    return run


def _validate_replay_records(
    engine_name: str,
    records: Sequence[Mapping[str, Any]],
    cases: Sequence[BenchCase],
) -> None:
    for index, (record, case) in enumerate(zip(records, cases, strict=True), start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"engine {engine_name!r} case {index} is malformed")
        if str(record.get("target", "")) != case.target or str(record.get("kind", "")) != case.kind:
            raise ValueError(
                f"engine {engine_name!r} case {index} ordering differs from the dataset"
            )
        case_meta = record.get("case_meta", {})
        if not isinstance(case_meta, dict) or case_meta != case.meta:
            raise ValueError(
                f"engine {engine_name!r} case {index} metadata differs from the dataset"
            )
        if record.get("error"):
            continue
        trace = record.get("ranked_candidates")
        if not isinstance(trace, list):
            raise ValueError(f"engine {engine_name!r} case {index} has an invalid candidate trace")
        try:
            generated = _strict_replay_int(
                record.get("generated", -1), "generated count", minimum=0
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"engine {engine_name!r} case {index} has an invalid generated count"
            ) from exc
        if generated < 0 or len(trace) != generated:
            raise ValueError(
                f"engine {engine_name!r} trace is incomplete ({len(trace)}/{generated}); "
                "the source run cannot be replayed exactly"
            )
        for rank, item in enumerate(trace, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("value"), str):
                raise ValueError(
                    f"engine {engine_name!r} case {index} trace item {rank} is malformed"
                )
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"engine {engine_name!r} case {index} trace item {rank} has invalid confidence"
                ) from exc
            if not math.isfinite(confidence):
                raise ValueError(
                    f"engine {engine_name!r} case {index} trace item {rank} has invalid confidence"
                )
        generation_meta = record.get("generation_meta", {})
        if not isinstance(generation_meta, dict):
            raise ValueError(f"engine {engine_name!r} case {index} has invalid generation metadata")
        cost = record.get("cost", {})
        if not isinstance(cost, dict):
            raise ValueError(f"engine {engine_name!r} case {index} has invalid cost metadata")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("known, hidden, and universe must be string arrays")
    return list(value)


def _strict_replay_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"benchmark result has an invalid {label}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"benchmark result has an invalid {label}")
    return value


def _validate_pricing(
    pricing: Mapping[str, tuple[float, float]],
    cache_read_pricing: Mapping[str, float],
) -> None:
    if not isinstance(pricing, Mapping) or not isinstance(cache_read_pricing, Mapping):
        raise ValueError("benchmark pricing must be mappings")
    for model, prices in pricing.items():
        if not isinstance(model, str) or not isinstance(prices, (tuple, list)) or len(prices) != 2:
            raise ValueError("benchmark model pricing must contain input/output pairs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in prices
        ):
            raise ValueError("benchmark model prices must be finite and non-negative")
    for model, value in cache_read_pricing.items():
        if (
            not isinstance(model, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("benchmark cache-read prices must be finite and non-negative")


def _failed_metric(
    case: BenchCase,
    budgets: Sequence[int],
    error: str,
    cost: Mapping[str, Any],
) -> BenchMetrics:
    cost_values = {str(key): _safe_nonnegative(value) for key, value in cost.items()}
    return BenchMetrics(
        target=case.target,
        kind=case.kind,
        generated=0,
        valid=0,
        recall_at_budget={budget: 0.0 for budget in budgets},
        precision_at_budget={budget: 0.0 for budget in budgets},
        precision=0.0,
        average_precision=0.0,
        reciprocal_rank=0.0,
        brier_score=0.0,
        fill_rate=0.0,
        novel_valid=0,
        cost=cost_values,
        cost_per_valid={key: None for key in cost_values},
        case_meta=dict(case.meta),
        error=safe_exception_text(error, 2_000),
    )


def _safe_count(value: Any) -> int:
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, int(parsed)) if math.isfinite(parsed) else 0


def _safe_nonnegative(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, parsed) if math.isfinite(parsed) else 0.0


def _markdown_report(result: dict[str, Any]) -> str:
    budgets = result["config"]["budgets"]
    largest = int(max(budgets)) if budgets else 0
    rows = []
    for name, payload in result["engines"].items():
        aggregate_result = payload["aggregate"]
        recall = aggregate_result["macro_recall_at_budget"].get(largest)
        if recall is None:
            recall = aggregate_result["macro_recall_at_budget"].get(str(largest), 0.0)
        ci = aggregate_result.get("macro_recall_ci95", {}).get(largest)
        if ci is None:
            ci = aggregate_result.get("macro_recall_ci95", {}).get(str(largest), [recall, recall])
        rows.append((name, float(recall), [float(ci[0]), float(ci[1])], aggregate_result))
    rows.sort(key=lambda row: row[1], reverse=True)
    lines = [
        "# Enumeraite Benchmark Report",
        "",
        f"Cases: {result['dataset_cases']} · Candidate count: {result['config']['count']}",
        "",
        f"| Engine | Model | Recall@{largest} | Δ recall vs mutation | Precision | Avg precision | Fill | Δ hits | Model-only hits | USD | Seconds | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, recall, ci, payload in rows:
        comparison = payload.get("comparison_to_mutation", {})
        delta = comparison.get("valid_delta")
        delta_text = f"{int(delta):+d}" if delta is not None else "—"
        recall_delta = comparison.get("recall_delta")
        recall_delta_ci = comparison.get("recall_delta_ci95")
        recall_delta_text = "—"
        if recall_delta is not None and recall_delta_ci:
            recall_delta_text = (
                f"{float(recall_delta):+.3f} "
                f"({float(recall_delta_ci[0]):+.3f}–{float(recall_delta_ci[1]):+.3f})"
            )
        cost = payload.get("cost", {})
        model_only = payload.get("unique_valid_by_source", {}).get("model")
        model_only_text = str(int(model_only)) if model_only is not None else "—"
        display_name = _markdown_cell(name)
        model = _markdown_cell(result["engines"][name].get("metadata", {}).get("model", "—"))
        lines.append(
            f"| {display_name} | {model} | {recall:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | "
            f"{recall_delta_text} | "
            f"{payload['micro_precision']:.3f} | "
            f"{payload['macro_average_precision']:.3f} | {payload['mean_fill_rate']:.2f} | "
            f"{delta_text} | {model_only_text} | {float(cost.get('usd', 0.0)):.4f} | "
            f"{float(cost.get('seconds', 0.0)):.1f} | {payload['errors']} |"
        )
    lines.extend(
        [
            "",
            "> Recall is macro-averaged per target. Precision is micro-averaged across generated candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    return safe_model_text(value, 200).replace("|", "\\|") or "—"


def _runtime_metadata() -> dict[str, Any]:
    revision = "unknown"
    dirty: bool | None = None
    try:
        git_executable = shutil.which("git")
        if not git_executable:
            raise OSError("git is not installed")
        # The executable is resolved once and every argument is constant; no shell is used.
        completed = subprocess.run(  # nosec B603
            [git_executable, "rev-parse", "--short=12", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            revision = completed.stdout.strip()
        status = subprocess.run(  # nosec B603
            [git_executable, "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "enumeraite_version": __version__,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
