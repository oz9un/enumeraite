"""The simple, pipe-first ``enumeraite run`` workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Sequence

import click

from ..core.config import Config, load_config
from ..core.dns_validator import DNSValidator
from ..core.engine import GenerationEngine
from ..core.factory import ProviderFactory
from ..core.feedback import FeedbackLoop, dns_validator_adapter, http_path_validator_adapter
from ..core.http_validator import PathHTTPValidator, _normalize_base_url
from ..core.prompt_spec import SUBDOMAIN_SPEC
from ..core.seeds import read_seed_file
from ..core.strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from ..core.text_safety import json_safe_value, safe_exception_text
from .detect import detect_kind, detect_provider, infer_apex
from .ui import Colors, info, warning


@click.command("run")
@click.argument("seeds", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-c",
    "--count",
    default=100,
    show_default=True,
    type=click.IntRange(min=1, max=5000),
    help="Candidate budget.",
)
@click.option("--provider", help="Provider name; auto-detected when omitted.")
@click.option("--model", help="Override the configured model.")
@click.option(
    "--hybrid/--no-hybrid",
    default=None,
    help="Combine seed-derived structural proposals with model candidates.",
)
@click.option(
    "--kind",
    type=click.Choice(["auto", "path", "subdomain"]),
    default="auto",
    show_default=True,
)
@click.option("--target", help="Target host/base URL; required for path validation.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write results to a file.",
)
@click.option("-q", "--quiet", is_flag=True, help="Emit raw results without status output.")
@click.option("--json", "json_output", is_flag=True, help="Emit structured JSON.")
@click.option("--validate", is_flag=True, help="Return only DNS/HTTP-validated findings.")
@click.option("--loop", is_flag=True, help="Iteratively generate, validate, and refeed findings.")
@click.option(
    "--rounds",
    default=3,
    show_default=True,
    type=click.IntRange(min=1, max=20),
    help="Maximum feedback rounds.",
)
def run_command(
    seeds: Path,
    count: int,
    provider: str | None,
    model: str | None,
    hybrid: bool | None,
    kind: str,
    target: str | None,
    output: Path | None,
    quiet: bool,
    json_output: bool,
    validate: bool,
    loop: bool,
    rounds: int,
) -> None:
    """Generate target-specific candidates from a seed file.

    Example: ``enumeraite run subs.txt | httpx``
    """
    items = _load_items(seeds)
    if not items:
        raise click.ClickException("The seed file contains no usable items.")
    detected_kind, confidence = detect_kind(items)
    resolved_kind = detected_kind if kind == "auto" else kind
    if kind == "auto" and confidence < 0.6 and not quiet:
        warning(f"Mixed seed file; assuming {resolved_kind} ({confidence:.0%} agreement)")

    resolved_target = target or (infer_apex(items) if resolved_kind == "subdomain" else "")
    if resolved_kind == "subdomain":
        resolved_target = resolved_target.lower().rstrip(".")
        if not resolved_target:
            raise click.ClickException(
                "Could not safely infer an apex; pass --target (for example, example.com)."
            )
        if not SUBDOMAIN_SPEC.validator(resolved_target):
            raise click.ClickException("Subdomain --target must be a valid DNS apex.")
    elif resolved_target:
        try:
            resolved_target = _normalize_base_url(resolved_target)
        except ValueError as exc:
            raise click.ClickException(f"Invalid path --target: {exc}") from exc

    config = load_config()
    provider_name = provider or detect_provider(config)
    if not provider_name:
        raise click.ClickException(
            "No usable AI provider detected. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "OPENROUTER_API_KEY, MINIMAX_API_KEY, or start Ollama; "
            "run 'enumeraite doctor' for details."
        )
    try:
        overrides: dict[str, Any] = {}
        if model:
            overrides["model"] = model
        elif provider_name == "huggingface" and not config.get_provider_config("huggingface").get(
            "model"
        ):
            overrides["model"] = _huggingface_model(resolved_kind)
        if hybrid is not None:
            overrides["hybrid"] = hybrid
        ai_provider = ProviderFactory(config).create_provider(provider_name, overrides)
    except Exception as exc:
        raise click.ClickException(safe_exception_text(exc)) from exc

    strategy = (
        SubdomainDiscoveryStrategy() if resolved_kind == "subdomain" else PathDiscoveryStrategy()
    )
    if not quiet:
        info(f"{resolved_kind} discovery via {ai_provider.get_provider_name()}")

    if loop:
        validate = True
        validator = _validator_for(resolved_kind, resolved_target, config)
        batch_size = max(1, min(count, (count + rounds - 1) // rounds))
        feedback_result = FeedbackLoop(ai_provider, validator).run(
            items,
            resolved_target,
            resolved_kind,
            batch_size=batch_size,
            max_rounds=rounds,
            max_candidates=count,
        )
        values = feedback_result.validated_hits
        score_by_value = dict(
            zip(feedback_result.candidates, feedback_result.confidence_scores, strict=True)
        )
        scores = [score_by_value.get(value, 0.0) for value in values]
        metadata = {
            **feedback_result.metadata,
            "stop_reason": feedback_result.stop_reason,
            "rounds": len(feedback_result.rounds),
            "generated": len(feedback_result.candidates),
        }
    else:
        generation = GenerationEngine(ai_provider).generate_paths(
            items, resolved_target, count, strategy=strategy
        )
        values = generation.paths
        scores = generation.confidence_scores
        metadata = generation.metadata
        if validate:
            outcomes = list(_validator_for(resolved_kind, resolved_target, config)(values))
            valid = {outcome.value for outcome in outcomes if outcome.exists}
            pairs = [
                (value, score)
                for value, score in zip(values, scores, strict=True)
                if value in valid
            ]
            values = [value for value, _ in pairs]
            scores = [score for _, score in pairs]
            metadata = {**metadata, "validated": True}

    _emit(values, scores, metadata, output, quiet, json_output)


def _validator_for(kind: str, target: str, config: Config | None = None):
    settings = config or Config()
    if kind == "subdomain":
        if not target:
            raise click.ClickException("Could not infer an apex for DNS validation; pass --target.")
        return dns_validator_adapter(
            DNSValidator(
                timeout=settings.request_timeout,
                max_concurrent=settings.max_concurrent_requests,
            ),
            target,
            detect_wildcard=True,
        )
    if not target:
        raise click.ClickException(
            "Path validation and --loop require --target (host or base URL)."
        )
    return http_path_validator_adapter(
        PathHTTPValidator(
            timeout=settings.request_timeout,
            max_concurrent=settings.max_concurrent_requests,
            verify_ssl=settings.http_verify_ssl,
            soft_404_threshold=settings.http_soft_404_threshold,
            baseline_samples=settings.http_baseline_samples,
            baseline_depth=settings.http_baseline_depth,
            max_baseline_groups=settings.http_max_baseline_groups,
        ),
        target,
    )


def _load_items(path: Path) -> List[str]:
    try:
        return read_seed_file(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise click.ClickException(f"Could not read seeds: {safe_exception_text(exc)}") from exc


def _huggingface_model(kind: str) -> str:
    suffix = "Subdomain" if kind == "subdomain" else "Path"
    return f"enumeraite/Enumeraite-x-Qwen3-4B-{suffix}"


def _emit(
    values: Sequence[str],
    scores: Sequence[float],
    metadata: dict,
    output: Path | None,
    quiet: bool,
    json_output: bool,
) -> None:
    if json_output:
        rendered = (
            json.dumps(
                {
                    "candidates": [
                        {"value": value, "confidence": score}
                        for value, score in zip(values, scores, strict=True)
                    ],
                    "metadata": json_safe_value(metadata),
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    else:
        rendered = "".join(f"{value}\n" for value in values)
    if output:
        output.write_text(rendered, encoding="utf-8")
        if not quiet:
            info(f"Wrote {len(values)} result(s) to {output}")
        return
    if json_output or quiet or not sys.stdout.isatty():
        click.echo(rendered, nl=False)
        return
    for value, score in zip(values, scores, strict=True):
        if score >= 0.7:
            color = Colors.BRIGHT_GREEN
        elif score >= 0.4:
            color = Colors.BRIGHT_YELLOW
        else:
            color = Colors.WHITE
        click.echo(f"{click.style(f'[{score:.2f}]', fg=color)} {value}")
