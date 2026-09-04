"""Opt-in Recon Royale automation with local validation gates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from ..core.config import load_config
from ..core.dns_validator import DNSValidator
from ..core.factory import ProviderFactory
from ..core.feedback import FeedbackLoop, dns_validator_adapter
from ..core.recon_royale import ReconRoyaleClient, filter_in_scope, prepare_submission
from ..core.seeds import read_seed_file
from ..core.text_safety import json_safe_value, safe_exception_text
from .detect import detect_provider


@click.command("royale")
@click.option(
    "--seeds",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Passive baseline results.",
)
@click.option("--provider", help="AI provider; auto-detected when omitted.")
@click.option("--model", help="Override configured model.")
@click.option("--count", default=200, show_default=True, type=click.IntRange(min=1, max=5000))
@click.option("--rounds", default=3, show_default=True, type=click.IntRange(min=1, max=10))
@click.option(
    "--min-valid-rate",
    default=0.20,
    show_default=True,
    type=click.FloatRange(min=0.10, max=1.0),
)
@click.option(
    "--submit",
    is_flag=True,
    help="Submit after all safety gates pass; default is dry-run.",
)
def royale_command(
    seeds: Path,
    baseline: Path | None,
    provider: str | None,
    model: str | None,
    count: int,
    rounds: int,
    min_valid_rate: float,
    submit: bool,
) -> None:
    """Generate and locally validate a Recon Royale round.

    This command is a dry-run unless ``--submit`` is explicitly supplied.
    """
    api_key = os.getenv("RECON_ROYALE_API_KEY", "")
    if not api_key:
        raise click.ClickException("Set RECON_ROYALE_API_KEY first.")
    client = ReconRoyaleClient(api_key)
    try:
        target = client.current_target()
    except Exception as exc:
        raise click.ClickException(
            f"Could not fetch the current target: {safe_exception_text(exc)}"
        ) from exc

    seed_values = _read(seeds)
    seed_values = filter_in_scope(seed_values, target)
    if not seed_values:
        raise click.ClickException("The seed file has no subdomains in the current target scope.")
    passive_values = _read(baseline) if baseline else []
    config = load_config()
    provider_name = provider or detect_provider(config)
    if not provider_name:
        raise click.ClickException("No usable AI provider; run 'enumeraite doctor'.")
    try:
        overrides = {"model": model} if model else None
        if (
            provider_name == "huggingface"
            and not model
            and not config.get_provider_config("huggingface").get("model")
        ):
            overrides = {"model": "enumeraite/Enumeraite-x-Qwen3-4B-Subdomain"}
        ai_provider = ProviderFactory(config).create_provider(provider_name, overrides)
    except Exception as exc:
        raise click.ClickException(safe_exception_text(exc)) from exc

    dns = DNSValidator(
        timeout=config.request_timeout,
        max_concurrent=config.max_concurrent_requests,
    )
    generated = FeedbackLoop(ai_provider, dns_validator_adapter(dns, target)).run(
        seed_values,
        target,
        "subdomain",
        batch_size=max(1, (count + rounds - 1) // rounds),
        max_rounds=rounds,
        max_candidates=count,
    )
    report = prepare_submission(
        target,
        passive_values,
        generated.candidates,
        validator=dns,
        min_valid_rate=min_valid_rate,
    )
    output = {
        **report.to_dict(),
        "dry_run": not submit,
        "loop_stop_reason": generated.stop_reason,
    }
    if submit:
        if not report.eligible:
            raise click.ClickException("Submission safety gate failed: " + report.reason)
        output["api_response"] = client.submit(report.valid_subdomains)
        output["dry_run"] = False
    click.echo(json.dumps(json_safe_value(output), indent=2, allow_nan=False))


def _read(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        return read_seed_file(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise click.ClickException(f"Could not read seeds: {safe_exception_text(exc)}") from exc
