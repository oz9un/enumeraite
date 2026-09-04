"""CLI commands for reproducible ground-truth benchmarks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import click

from ..bench.baselines import (
    mutation_baseline,
    mutation_only_baseline,
    structural_baseline,
    wordlist_baseline,
)
from ..bench.dataset import load_openapi_dataset, load_subdomain_dataset
from ..bench.runner import (
    DEFAULT_CACHE_READ_PRICING,
    DEFAULT_PRICING,
    BenchRunner,
    RunConfig,
    make_baseline_engine,
    make_llm_engine,
    replay_benchmark,
)
from ..core.bounded_io import read_bounded_regular_file
from ..core.config import load_config
from ..core.factory import ProviderFactory
from ..core.strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from ..core.text_safety import safe_exception_text

_MAX_REPLAY_BYTES = 256 * 1024 * 1024
_MAX_WORDLIST_BYTES = 64 * 1024 * 1024


@click.group("bench")
def bench_command() -> None:
    """Compare LLMs with identical ground truth and baseline controls."""


def _common_options(function):
    options = [
        click.option("--providers", default="", help="Comma-separated provider names."),
        click.option("--model", help="Override the model for every selected provider."),
        click.option("--temperature", type=click.FloatRange(min=0.0, max=2.0)),
        click.option(
            "--hybrid/--no-hybrid",
            default=None,
            help="Enable/disable structural+model candidate pooling.",
        ),
        click.option("--batch-size", type=click.IntRange(min=1, max=500)),
        click.option("--max-batch-calls", type=click.IntRange(min=1, max=100)),
        click.option("--max-tokens", type=click.IntRange(min=1, max=131_072)),
        click.option("--provider-timeout", type=click.FloatRange(min=0.1, max=3_600.0)),
        click.option("--hybrid-model-fraction", type=click.FloatRange(min=0.0, max=1.0)),
        click.option(
            "--count",
            default=200,
            show_default=True,
            type=click.IntRange(min=1, max=5000),
        ),
        click.option("--budgets", default="25,50,100,200", show_default=True),
        click.option("--seed-frac", default=None, type=click.FloatRange(min=0.01, max=0.99)),
        click.option(
            "--seed-limit",
            type=click.IntRange(min=1, max=5_000),
            help="Cap visible seeds per case.",
        ),
        click.option(
            "--repeats",
            default=1,
            show_default=True,
            type=click.IntRange(min=1, max=50),
        ),
        click.option("--salt", default="enumeraite-v1", show_default=True),
        click.option("--limit", type=click.IntRange(min=1)),
        click.option("--wordlist", type=click.Path(exists=True, dir_okay=False, path_type=Path)),
        click.option("--no-baselines", is_flag=True),
        click.option(
            "-o",
            "--out-dir",
            default="bench-results",
            type=click.Path(file_okay=False, path_type=Path),
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@bench_command.command("openapi")
@click.option(
    "--specs",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@_common_options
def openapi_bench(specs: Path, **kwargs) -> None:
    """Benchmark paths from complete OpenAPI specifications."""
    seed_frac = kwargs.pop("seed_frac") or 0.2
    repeats = kwargs.pop("repeats")
    seed_limit = kwargs.pop("seed_limit")
    dataset = []
    for repeat in range(repeats):
        repeat_salt = kwargs["salt"] if repeats == 1 else f"{kwargs['salt']}:repeat-{repeat + 1}"
        dataset.extend(
            load_openapi_dataset(
                specs,
                seed_frac=seed_frac,
                salt=repeat_salt,
                limit=kwargs["limit"],
                max_seed=seed_limit,
            )
        )
    _run(
        dataset,
        "path",
        repeats=repeats,
        seed_frac=seed_frac,
        seed_limit=seed_limit,
        **kwargs,
    )


@bench_command.command("subdomain")
@click.option(
    "--truth",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@_common_options
def subdomain_bench(truth: Path, **kwargs) -> None:
    """Benchmark one complete <apex>.txt truth set per target."""
    seed_frac = kwargs.pop("seed_frac") or 0.5
    repeats = kwargs.pop("repeats")
    seed_limit = kwargs.pop("seed_limit")
    dataset = []
    for repeat in range(repeats):
        repeat_salt = kwargs["salt"] if repeats == 1 else f"{kwargs['salt']}:repeat-{repeat + 1}"
        dataset.extend(
            load_subdomain_dataset(
                truth,
                seed_frac=seed_frac,
                salt=repeat_salt,
                limit=kwargs["limit"],
                max_seed=seed_limit,
            )
        )
    _run(
        dataset,
        "subdomain",
        repeats=repeats,
        seed_frac=seed_frac,
        seed_limit=seed_limit,
        **kwargs,
    )


@bench_command.command("replay")
@click.argument(
    "results",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--wordlist",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Original wordlist; required when the source run recorded one.",
)
@click.option(
    "-o",
    "--out-dir",
    default="bench-results/replay",
    type=click.Path(file_okay=False, path_type=Path),
)
def replay_bench(results: Path, wordlist: Path | None, out_dir: Path) -> None:
    """Rescore saved candidate traces with the current metrics; no model calls."""
    try:
        raw = read_bounded_regular_file(
            results,
            _MAX_REPLAY_BYTES,
            not_regular_message="benchmark result must be a regular file",
            too_large_message="benchmark result exceeds the 256 MiB replay limit",
        )
        source = json.loads(raw.decode("utf-8"))
        expected_hash = source.get("config", {}).get("wordlist_sha256")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError) as exc:
        raise click.ClickException(
            f"Could not read benchmark results: {safe_exception_text(exc)}"
        ) from exc
    if expected_hash and not wordlist:
        raise click.ClickException(
            "The source run used a wordlist; pass the same file with --wordlist."
        )
    words, actual_hash = _read_wordlist_payload(wordlist)
    if expected_hash and actual_hash != expected_hash:
        raise click.ClickException("The supplied wordlist SHA-256 does not match the source run.")
    try:
        result = replay_benchmark(
            results,
            out_dir,
            baseline_wordlist=words,
            wordlist_sha256=actual_hash,
            source_bytes=raw,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(safe_exception_text(exc)) from exc
    click.echo(
        f"Replayed {result['dataset_cases']} case(s) with {result['metric_version']}; no model calls."
    )
    click.echo(str(out_dir / "report.md"))


def _run(
    dataset,
    kind: str,
    providers: str,
    model: str | None,
    temperature: float | None,
    hybrid: bool | None,
    batch_size: int | None,
    max_batch_calls: int | None,
    max_tokens: int | None,
    provider_timeout: float | None,
    hybrid_model_fraction: float | None,
    repeats: int,
    seed_frac: float,
    seed_limit: int | None,
    count: int,
    budgets: str,
    salt: str,
    limit: int | None,
    wordlist: Path | None,
    no_baselines: bool,
    out_dir: Path,
) -> None:
    del limit
    if not dataset:
        raise click.ClickException("No eligible benchmark cases were found.")
    try:
        parsed_budgets = tuple(
            sorted({int(value) for value in budgets.split(",") if int(value) > 0})
        )
    except ValueError as exc:
        raise click.ClickException("--budgets must be comma-separated positive integers") from exc
    if not parsed_budgets:
        raise click.ClickException("--budgets must contain at least one positive integer")
    if any(value > 5_000 for value in parsed_budgets):
        raise click.ClickException("--budgets values must not exceed 5000")
    words, wordlist_hash = _read_wordlist_payload(wordlist)
    engines = {}
    engine_metadata: dict[str, dict[str, Any]] = {}
    if not no_baselines:
        engines["structural"] = make_baseline_engine(
            lambda known, size, target: structural_baseline(known, size, kind, target)
        )
        engine_metadata["structural"] = {
            "type": "baseline",
            "algorithm": "seed-derived structural gap completion only",
        }
        engines["mutation-only"] = make_baseline_engine(
            lambda known, size, target: mutation_only_baseline(known, size, kind, target)
        )
        engine_metadata["mutation-only"] = {
            "type": "baseline",
            "algorithm": "deterministic lexical mutations only",
        }
        engines["mutation"] = make_baseline_engine(
            lambda known, size, target: mutation_baseline(known, size, kind, target)
        )
        engine_metadata["mutation"] = {
            "type": "baseline",
            "algorithm": "seed-derived structural pool plus deterministic mutations",
        }
        if words:
            engines["wordlist"] = make_baseline_engine(
                lambda known, size, target: wordlist_baseline(words, target, kind, size)
            )
            engine_metadata["wordlist"] = {
                "type": "baseline",
                "algorithm": "supplied wordlist in file order",
                "sha256": wordlist_hash or "",
            }

    strategy = SubdomainDiscoveryStrategy() if kind == "subdomain" else PathDiscoveryStrategy()
    config = load_config()
    factory = ProviderFactory(config)
    for provider_name in [name.strip() for name in providers.split(",") if name.strip()]:
        try:
            overrides: dict[str, Any] = {}
            if model:
                overrides["model"] = model
            elif provider_name == "huggingface" and not config.get_provider_config(
                "huggingface"
            ).get("model"):
                suffix = "Subdomain" if kind == "subdomain" else "Path"
                overrides["model"] = f"enumeraite/Enumeraite-x-Qwen3-4B-{suffix}"
            if temperature is not None:
                overrides["temperature"] = temperature
            if hybrid is not None:
                overrides["hybrid"] = hybrid
            for key, value in {
                "batch_size": batch_size,
                "max_batch_calls": max_batch_calls,
                "max_tokens": max_tokens,
                "timeout": provider_timeout,
                "hybrid_model_fraction": hybrid_model_fraction,
            }.items():
                if value is not None:
                    overrides[key] = value
            provider = factory.create_provider(provider_name, overrides)
        except Exception as exc:
            raise click.ClickException(f"{provider_name}: {safe_exception_text(exc)}") from exc
        engine_name = provider_name + ("+hybrid" if getattr(provider, "use_hybrid", False) else "")
        engines[engine_name] = make_llm_engine(provider, strategy)
        resolved_model = str(getattr(provider, "model", ""))
        prices = DEFAULT_PRICING.get(resolved_model)
        engine_metadata[engine_name] = {
            "type": "llm",
            "provider": provider_name,
            "implementation": provider.get_provider_name(),
            "model": resolved_model,
            "hybrid": bool(getattr(provider, "use_hybrid", False)),
            "hybrid_model_fraction": getattr(provider, "hybrid_model_fraction", None),
            "confidence": getattr(provider, "confidence_mode", None),
            "context": getattr(provider, "use_context", None),
            "temperature": getattr(provider, "temperature", None),
            "max_tokens": getattr(provider, "max_tokens", None),
            "batch_size": getattr(provider, "batch_size", None),
            "max_batch_calls": getattr(provider, "max_batch_calls", None),
            "pricing_usd_per_million_tokens": (
                {
                    "input": prices[0],
                    "output": prices[1],
                    "cache_read": DEFAULT_CACHE_READ_PRICING.get(resolved_model, prices[0]),
                }
                if prices
                else None
            ),
        }
    if not engines:
        raise click.ClickException("Select --providers or keep baseline controls enabled.")

    runner = BenchRunner(
        RunConfig(
            count=count,
            budgets=parsed_budgets,
            baseline_wordlist=words,
            out_dir=out_dir,
            engine_metadata=engine_metadata,
            experiment={
                "salt": salt,
                "repeats": repeats,
                "seed_fraction": seed_frac,
                "seed_limit": seed_limit,
                "wordlist_sha256": wordlist_hash,
            },
        )
    )
    result = runner.run(dataset, engines)
    click.echo(f"Benchmarked {result['dataset_cases']} case(s).")
    click.echo(str(out_dir / "report.md"))


def _read_wordlist(path: Path | None) -> Sequence[str]:
    return _read_wordlist_payload(path)[0]


def _read_wordlist_payload(path: Path | None) -> tuple[Sequence[str], str | None]:
    if not path:
        return (), None
    try:
        raw = read_bounded_regular_file(
            path,
            _MAX_WORDLIST_BYTES,
            not_regular_message="wordlist must be a regular file",
            too_large_message="Wordlist exceeds the 64 MiB input limit.",
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not read wordlist: {safe_exception_text(exc)}") from exc
    return (
        tuple(
            line.strip()
            for line in raw.decode("utf-8", errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        hashlib.sha256(raw).hexdigest(),
    )
