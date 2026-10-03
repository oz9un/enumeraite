"""One-domain discovery with passive seeds, recursive expansion, and checkpoints."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from ..core.config import load_config
from ..core.discovery_session import (
    DiscoverySession,
    ScanOptions,
    normalize_target,
    validate_checkpoint,
)
from ..core.factory import ProviderFactory
from ..core.passive import collect_passive
from ..core.scan_budget import SpendLedger, price_for
from ..core.scan_dns import ScanDNS
from ..core.scan_input import resolve_scan_input, validate_seed_depth
from ..core.scan_state import ScanStore, default_state_path
from ..core.seeds import read_seed_file
from ..core.text_safety import safe_exception_text
from ..providers.llm_provider import LLMProvider
from .detect import detect_provider
from .scan_tree import text_tree, write_tree


def _duration(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h)?", value.strip().lower())
    if not match:
        raise ValueError("--time accepts seconds or a duration such as 30s, 5m, 1h")
    seconds = float(match[1]) * {None: 1, "s": 1, "m": 60, "h": 3600}[match[2]]
    if not 0 < seconds <= 86400:
        raise ValueError("--time must be positive and no more than 24h")
    return seconds


@click.command("scan")
@click.argument("target")
@click.option("--seeds", help="Optional known-subdomain file; use - for stdin.")
@click.option(
    "--seed",
    "inline_seeds",
    multiple=True,
    help="Known hostname to learn from; repeat for multiple seeds.",
)
@click.option(
    "--scope", help="Pin an authorized DNS zone instead of inferring the registrable domain."
)
@click.option(
    "--find",
    "find_count",
    type=click.IntRange(1, 50000),
    help="Stop at N unique verified subdomains, including known names.",
)
@click.option(
    "--budget-usd",
    type=click.FloatRange(min=0),
    help="Cumulative generation-token budget (default: $1 with AI); excludes optional Jev prediction.",
)
@click.option(
    "--predictor",
    type=click.Choice(["none", "jev"]),
    default=None,
    help="Optional numeric plan ranking with Jev; disabled by default, preserved on resume.",
)
@click.option(
    "--prediction-budget-usd",
    type=click.FloatRange(min=0, max=1000),
    help="Additional cumulative Jev prediction budget, separate from --budget-usd (default: $0.05 when enabled).",
)
@click.option(
    "--time", "time_limit", help="Time per invocation, including passive discovery (default: 5m)."
)
@click.option(
    "--max-candidates",
    type=click.IntRange(1, 50000),
    help="Total unique DNS candidates, including passive names (default: 1000).",
)
@click.option(
    "--rounds",
    type=click.IntRange(1, 100),
    help="Maximum recursive expansion rounds (default: 10).",
)
@click.option(
    "--until-stale",
    is_flag=True,
    help="Stop after two rounds without findings; allow up to 100 rounds unless --rounds is set.",
)
@click.option(
    "--workers", type=click.IntRange(1, 100), help="Concurrent DNS workers (default: 20)."
)
@click.option(
    "--rate",
    type=click.FloatRange(min=0.1, max=1000),
    help="Maximum DNS queries per second, including wildcard probes (default: 50).",
)
@click.option(
    "--known-limit",
    type=click.IntRange(1, 5000),
    help="Maximum passive names retained (default: 200).",
)
@click.option(
    "--max-depth",
    type=click.IntRange(1, 10),
    help="Maximum expansion depth and DNS labels below target (default: 4).",
)
@click.option(
    "--max-number",
    type=click.IntRange(1, 9999),
    help=(
        "Inclusive ceiling for deterministic numeric-slot alternatives, not fixed literals "
        "or supplied/AI names (default: 99)."
    ),
)
@click.option(
    "--passive/--no-passive",
    default=None,
    help="Collect public-source names (default: enabled for roots, disabled with known seeds).",
)
@click.option(
    "--provider",
    default="none",
    show_default=True,
    help="AI provider; a dollar budget auto-detects one. Default discovery is free.",
)
@click.option("--model", help="Model override.")
@click.option(
    "--vocabulary-breadth/--no-vocabulary-breadth",
    default=None,
    help="Experimental: use some existing AI passes for grounded naming vocabulary (default: off).",
)
@click.option(
    "--input-price",
    type=click.FloatRange(min=0),
    help="Custom input price in USD per million tokens.",
)
@click.option(
    "--output-price",
    type=click.FloatRange(min=0),
    help="Custom output price in USD per million tokens.",
)
@click.option(
    "--resume", is_flag=True, help="Continue the latest saved scan, or the selected --state file."
)
@click.option(
    "--state",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Checkpoint file (default: private application state directory).",
)
@click.option("--tree", is_flag=True, help="Show the verified discovery tree on stderr.")
@click.option(
    "--tree-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save an interactive HTML discovery tree.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit a structured report instead of one finding per line.",
)
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress; findings still go to stdout.")
def scan_command(
    target: str,
    seeds: str | None,
    inline_seeds: tuple[str, ...],
    scope: str | None,
    find_count: int | None,
    budget_usd: float | None,
    predictor: str | None,
    prediction_budget_usd: float | None,
    time_limit: str | None,
    max_candidates: int | None,
    rounds: int | None,
    until_stale: bool,
    workers: int | None,
    rate: float | None,
    known_limit: int | None,
    max_depth: int | None,
    max_number: int | None,
    passive: bool | None,
    provider: str,
    model: str | None,
    vocabulary_breadth: bool | None,
    input_price: float | None,
    output_price: float | None,
    resume: bool,
    state: Path | None,
    tree: bool,
    tree_file: Path | None,
    json_output: bool,
    output: Path | None,
    quiet: bool,
) -> None:
    """Discover from a root domain or learn sibling patterns from a known hostname.

    Example: enumeraite scan example.com --find 100 --tree

    A hostname such as ny-01-iphone.example.com seeds a scan within example.com.
    Use --scope to restrict discovery to a narrower authorized DNS zone.

    Only scan domains you are authorized to assess. A finding goal is best effort;
    count, time, and spending limits can stop a scan before it reaches that goal.
    """

    def status(message: str) -> None:
        if not quiet:
            click.echo(message, err=True)

    prediction_client = None
    prediction_key = ""
    try:
        target = normalize_target(target)
        for name, value in (
            ("budget-usd", budget_usd),
            ("input-price", input_price),
            ("output-price", output_price),
            ("prediction-budget-usd", prediction_budget_usd),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"--{name} must be finite")
        supplied_seeds = [*inline_seeds, *_read_seeds(seeds)]
        scan_input = resolve_scan_input(
            target, scope=scope, seeds=None if resume else supplied_seeds
        )
        if state is not None:
            state_path = state
        elif resume and scope is None and scan_input.input != scan_input.target:
            # Older versions treated every positional hostname as an exact zone.
            # Prefer that existing checkpoint instead of silently broadening it.
            try:
                state_path = default_state_path(scan_input.input, resume=True)
            except ValueError:
                state_path = default_state_path(scan_input.target, resume=True)
        else:
            state_path = default_state_path(scan_input.target, resume)
        protected = [state_path.resolve()]
        if seeds and seeds != "-":
            protected.append(Path(seeds).resolve())
        if output and output.resolve() in protected:
            raise ValueError("Output must not overwrite the checkpoint or seeds")
        if tree_file and tree_file.resolve() in protected + ([output.resolve()] if output else []):
            raise ValueError("Tree output must have a separate path")
        store = ScanStore(state_path)
        with store.locked(resume=resume):
            saved = store.load() if resume else None
            if saved:
                saved_target = saved.get("target")
                if not isinstance(saved_target, str):
                    raise ValueError("Checkpoint target does not match this scan")
                validate_checkpoint(saved, saved_target)
                if scope is not None and normalize_target(scope) != saved_target:
                    raise ValueError("Checkpoint target does not match the requested scope")
                if target != saved_target and not target.endswith("." + saved_target):
                    raise ValueError("Checkpoint target does not match this scan")
                # Saved scope is authoritative; resume must never infer a wider
                # target or silently discard the user's newly supplied seeds.
                scan_input = resolve_scan_input(target, scope=saved_target, seeds=supplied_seeds)
            target = scan_input.target
            options_data = dict(saved["options"]) if saved else asdict(ScanOptions())
            if not saved:
                options_data["passive"] = scan_input.passive_default
            overrides = {
                "find": find_count,
                "time_limit": _duration(time_limit) if time_limit else None,
                "max_candidates": max_candidates,
                "max_rounds": rounds,
                "workers": workers,
                "rate": rate,
                "known_limit": known_limit,
                "max_depth": max_depth,
                "max_number": max_number,
                "passive": passive,
                "vocabulary_breadth": vocabulary_breadth,
            }
            options_data.update(
                {key: value for key, value in overrides.items() if value is not None}
            )
            if until_stale:
                options_data["stale_rounds"] = 2
                if rounds is None:
                    options_data["max_rounds"] = 100
            options = ScanOptions(**options_data)
            validate_seed_depth(target, scan_input.seeds, options.max_depth)
            prediction_client, prediction_ledger = _predictor_and_ledger(
                predictor, prediction_budget_usd, saved
            )
            if prediction_client is not None:
                prediction_key = os.environ.get("TYPESAFE_API_KEY", "")
            provider_source = click.get_current_context().get_parameter_source("provider")
            if provider_source == click.core.ParameterSource.DEFAULT and (
                budget_usd is not None or (saved and saved.get("provider"))
            ):
                provider = "auto"
            ai, ledger = _provider_and_ledger(
                provider, model, budget_usd, input_price, output_price, saved, status
            )
            status(
                f"Scanning {target}: up to {options.max_candidates} candidates, {options.time_limit:g}s, {options.workers} DNS workers"
            )
            if scan_input.seeds:
                status(
                    f"Learning from {len(scan_input.seeds)} supplied seed hostname(s) within {target}"
                )
            if ai is not None and ledger is not None:
                status(
                    f"AI: {ai.get_provider_name()} / {ai.model}; token-spend cap ${ledger.limit_usd:g}"
                )
            elif prediction_client is None:
                status(
                    "Using free DNS validation and deterministic pattern expansion; no paid AI calls"
                )
            else:
                status(
                    "Using deterministic candidate generation; optional Jev prediction is separately billed"
                )
            if prediction_client is not None and prediction_ledger is not None:
                status(
                    f"Jev prediction: additional cumulative cap ${prediction_ledger.limit_usd:g}; "
                    "separate from the generation budget"
                )
            status(f"Progress saved to {state_path}")
            emitted: set[str] = set()

            def event(kind: str, payload: dict) -> None:
                payload = _redact_prediction_key(payload, prediction_key)
                if kind == "finding":
                    value = payload["value"]
                    if value not in emitted and not json_output and output is None:
                        click.echo(value)
                    emitted.add(value)
                elif kind in {"phase", "warning"}:
                    status(payload["message"])
                elif kind == "passive":
                    status(f"Collected {payload['count']} known names; verifying DNS")
                elif kind == "progress":
                    spend = float((payload.get("ledger") or {}).get("spent_usd", 0))
                    prediction_spend = float(
                        (payload.get("prediction_ledger") or {}).get("spent_usd", 0)
                    )
                    status(
                        f"{payload['verified']} verified · {payload['attempted']} attempted · {payload['queued']} queued · ${spend:.4f} generation"
                        + (
                            f" + ${prediction_spend:.4f} additional Jev prediction"
                            if prediction_client is not None
                            else ""
                        )
                    )

            prediction_args = (
                {"predictor": prediction_client, "prediction_ledger": prediction_ledger}
                if prediction_client is not None
                else {}
            )
            session = DiscoverySession(
                target,
                options,
                ScanDNS(
                    target, workers=options.workers, rate=options.rate, timeout=options.dns_timeout
                ),
                passive_fetch=collect_passive,
                provider=ai,
                ledger=ledger,
                checkpoint=(
                    lambda value: store.save(_redact_prediction_key(value, prediction_key))
                ),
                event=event,
                state=saved,
                **prediction_args,
            )
            session.run(scan_input.seeds)
            report = _redact_prediction_key(session.report(), prediction_key)
            report["state_file"] = str(state_path)
            report["scan_input"] = asdict(scan_input)
            if tree:
                click.echo(text_tree(report), err=True)
            if tree_file:
                write_tree(report, tree_file)
                status(f"Discovery tree saved to {tree_file}")
            if json_output:
                rendered = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
            else:
                rendered = "".join(value + "\n" for value in report["findings"])
            if output:
                output.write_text(rendered, encoding="utf-8")
            elif json_output:
                click.echo(rendered, nl=False)
            status(
                f"Finished: {report['verified']} verified ({report['known_verified']} previously known); "
                f"{report['inconclusive']} inconclusive, {report['wildcard']} wildcard; {report['stop_reason']}"
            )
            if report["inconclusive"]:
                status(
                    f"Warning: DNS checks were inconclusive for {report['inconclusive']} names; "
                    "results are incomplete, not confirmed absences. Check your network/DNS "
                    "and use --resume to retry eligible names."
                )
            if report["stop_reason"].startswith("interrupted"):
                raise click.exceptions.Exit(130)
    except (OSError, ValueError, TypeError) as exc:
        raise click.ClickException(
            _redact_prediction_key(safe_exception_text(exc), prediction_key)
        ) from exc
    finally:
        if prediction_client is not None:
            try:
                prediction_client.close()
            except Exception:
                status("Warning: prediction connection cleanup failed")


def _redact_prediction_key(value: Any, key: str) -> Any:
    if not key:
        return value
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, dict):
        return {name: _redact_prediction_key(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_prediction_key(item, key) for item in value]
    return value


def _predictor_and_ledger(
    name: str | None, budget: float | None, saved: dict | None
) -> tuple[Any, SpendLedger | None]:
    previous = saved.get("predictor") if saved else None
    name = name if name is not None else ("jev" if previous else "none")
    if name not in {"none", "jev"}:
        raise ValueError("Choose no predictor or Jev")
    if saved is not None and name != ("jev" if previous else "none"):
        raise ValueError(
            "Resume requires the same prediction provider; start a new scan to change it"
        )
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or not 0 <= budget <= 1000
    ):
        raise ValueError("Additional prediction budget must be between 0 and 1000 USD")
    if name == "none":
        if budget is not None:
            raise ValueError("Use --predictor jev to set an additional prediction budget")
        return None, None
    from ..providers.jev import JEV_MODEL, JEV_PRICE, JevPredictor

    if previous is not None and previous != {"name": "jev", "model": JEV_MODEL}:
        raise ValueError("Resume requires the saved Jev model")
    if saved is not None:
        if saved.get("prediction_ledger") is None:
            raise ValueError("Saved Jev scan is missing its prediction budget")
        ledger = SpendLedger.from_dict(saved["prediction_ledger"])
        if budget is not None:
            restored = ledger.to_dict()
            restored["limit_usd"] = str(budget)
            ledger = SpendLedger.from_dict(restored)
            if Decimal(ledger.to_dict()["spent_usd"]) > Decimal(ledger.to_dict()["limit_usd"]):
                raise ValueError(
                    "Prediction budget cannot be below previously spent or reserved charges"
                )
    else:
        ledger = SpendLedger(0.05 if budget is None else budget, JEV_PRICE)
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise ValueError("Set TYPESAFE_API_KEY in your environment to enable Jev prediction")
    try:
        client = JevPredictor(api_key=key, model=JEV_MODEL)
    except Exception as exc:
        raise ValueError("Could not initialize Jev prediction; check TYPESAFE_API_KEY") from exc
    return client, ledger


def _read_seeds(seeds: str | None) -> list[str]:
    if not seeds:
        return []
    if seeds != "-":
        return read_seed_file(Path(seeds))
    data = sys.stdin.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("Seed stdin exceeds 2 MB")
    return [
        value.strip()
        for value in data.splitlines()
        if value.strip() and not value.lstrip().startswith("#")
    ]


def _provider_and_ledger(
    name: str,
    model: str | None,
    budget: float | None,
    input_price: float | None,
    output_price: float | None,
    saved: dict | None,
    status: Any,
) -> tuple[LLMProvider | None, SpendLedger | None]:
    saved_provider = saved.get("provider") if saved else None
    previous_ledger = saved.get("ledger") if saved else None
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not 0 <= budget < float("inf")
    ):
        raise ValueError("Budget must be a finite nonnegative USD amount")
    explicit_price = (
        price_for("", "", input_price, output_price)
        if input_price is not None or output_price is not None
        else None
    )
    ledger = None
    if previous_ledger is not None:
        previous = dict(previous_ledger)
        if budget is not None:
            previous["limit_usd"] = budget
        # Restore unsettled reservations at their original charge before
        # changing future prices or the cap. Historical spend is cumulative.
        ledger = SpendLedger.from_dict(previous)
        if ledger.spent_usd > ledger.limit_usd:
            raise ValueError("Budget cannot be less than the amount already spent")
    if name == "none":
        return None, ledger
    if saved_provider is not None:
        saved_name = saved_provider["name"].removeprefix("openai-compat:")
        if name != "auto" and name.removeprefix("openai-compat:") != saved_name:
            raise ValueError("Resume requires the same provider and model as the saved scan")
        if model is not None and model != saved_provider["model"]:
            raise ValueError("Resume requires the same provider and model as the saved scan")
        model = saved_provider["model"]
    config = load_config()
    if name == "auto" and saved_provider:
        selected = saved_provider["name"].removeprefix("openai-compat:")
    else:
        selected = detect_provider(config) if name == "auto" else name
    if not selected:
        return None, ledger
    overrides: dict[str, Any] = {
        "max_retries": 0,
        "max_batch_calls": 1,
        "batch_size": 20,
        "timeout": 30,
        "hybrid": False,
    }
    if model:
        overrides["model"] = model
    try:
        ai = ProviderFactory(config).create_provider(selected, overrides)
        if not isinstance(ai, LLMProvider):
            raise ValueError("scan AI requires a chat provider; use run for Hugging Face models")
        identity = {"name": ai.get_provider_name(), "model": str(ai.model)}
        if saved_provider is not None and identity != saved_provider:
            raise ValueError("Resume requires the same provider and model as the saved scan")
        if ledger is not None:
            # Saved prices are part of the resume contract, including unknown
            # custom models and the original dated registry snapshot. Explicit
            # overrides deliberately update future estimates, not past charges.
            if explicit_price is not None:
                ledger.pricing = explicit_price
                status(
                    "Using updated token prices for future requests; saved spending is unchanged."
                )
        else:
            price = explicit_price or price_for(ai.get_provider_name(), str(ai.model))
            ledger = SpendLedger(1.0 if budget is None else budget, price)
        return ai, ledger
    except (ValueError, TypeError) as exc:
        if name != "auto" or saved_provider:
            raise
        status("AI unavailable; continuing deterministic discovery: " + safe_exception_text(exc))
        return None, None
