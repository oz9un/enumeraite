"""Professional CLI commands for Enumeraite."""

from pathlib import Path
from typing import List

import click

from ..core.config import load_config
from ..core.dns_validator import DNSValidator
from ..core.engine import GenerationEngine
from ..core.factory import ProviderFactory
from ..core.http_validator import HTTPValidator
from ..core.path_function_analysis import PathFunctionAnalysisEngine
from ..core.pattern_analysis import PatternAnalysisEngine
from ..core.prompt_spec import PATH_SPEC, SUBDOMAIN_SPEC
from ..core.seeds import read_seed_file
from ..core.strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from ..core.text_safety import safe_exception_text
from .detect import detect_provider, infer_apex
from .ui import (
    error,
    info,
    print_debug_info,
    print_generation_start,
    print_path_analysis,
    print_pattern_analysis,
    print_results,
    print_startup_banner,
    print_validation_stats,
    section_header,
    status,
    success,
    warning,
)


# Generate command group
@click.group()
def generate():
    """Bulk generation from wordlists with AI pattern recognition."""
    pass


@generate.command("subdomain")
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input file with known subdomains",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    help="Output file (default: stdout)",
)
@click.option(
    "-c",
    "--count",
    default=50,
    show_default=True,
    type=click.IntRange(min=1, max=5000),
    help="Number to generate",
)
@click.option("--provider", help="AI provider name; auto-detected when omitted")
@click.option("--model", help="Specific model to use")
@click.option("--target", help="DNS apex; required when it cannot be inferred safely")
@click.option("--validate", is_flag=True, help="Enable DNS validation")
@click.option("--check-http", is_flag=True, help="Check HTTP (requires --validate)")
@click.option("--debug", is_flag=True, help="Show debug info")
@click.pass_context
def generate_subdomain(
    ctx,
    input_file,
    output_file,
    count,
    provider,
    model,
    target,
    validate,
    check_http,
    debug,
):
    """Generate subdomains from existing patterns.

    \b
    Examples:
      enumeraite generate subdomain -i known.txt -o new.txt
      enumeraite generate subdomain -i subs.txt --validate --check-http
      enumeraite generate subdomain -i subs.txt --provider huggingface
    """
    # Load known subdomains
    known_subdomains = _load_items_from_file(input_file)
    if not known_subdomains:
        error("No subdomains found in input file")
        ctx.exit(1)

    # Validate input format - should be full subdomain names
    for subdomain in known_subdomains[:3]:  # Check first few
        if "." not in subdomain or subdomain.startswith("/"):
            warning(
                f"Input should contain full subdomains (e.g., 'api.example.com'), got: {subdomain}"
            )
            break

    if check_http and not validate:
        error("--check-http requires --validate")
        ctx.exit(1)
    target = (target or infer_apex(known_subdomains)).lower().rstrip(".")
    if not target:
        error("Could not safely infer an apex; pass --target (for example, example.com)")
        ctx.exit(1)
    if not SUBDOMAIN_SPEC.validator(target):
        error("--target must be a valid DNS apex")
        ctx.exit(1)

    # Load configuration and setup provider
    config, ai_provider = _setup_provider(ctx, provider, model, "subdomain")

    # Setup engine
    engine = GenerationEngine(ai_provider)

    # Print startup banner on first use
    print_startup_banner()

    # Print generation start info
    print_generation_start(count, ai_provider.get_provider_name(), "subdomain")

    if validate:
        info("DNS validation enabled")
    if check_http and validate:
        info("HTTP validation enabled")

    existing_subdomains: List[str] = []
    try:
        # Generate subdomains
        subdomain_strategy = SubdomainDiscoveryStrategy()
        generation_result = engine.generate_paths(
            known_subdomains, target, count, strategy=subdomain_strategy
        )

        # Output results. With validation, defer the file write until only
        # validated names remain; do not claim the file is already saved.
        if output_file and not validate:
            _output_items_to_file(generation_result.paths, output_file)
        print_results(
            generation_result.paths,
            output_file,
            saved=bool(output_file and not validate),
        )

        # DNS validation
        if validate:
            section_header("DNS Validation")
            status(f"Validating {len(generation_result.paths)} subdomains")

            validator = DNSValidator(
                timeout=config.request_timeout,
                max_concurrent=config.max_concurrent_requests,
            )
            dns_results = validator.validate_subdomains(
                generation_result.paths,
                detect_wildcard=True,
                apex=target,
            )

            # Filter to only existing subdomains
            existing_subdomains = [result.subdomain for result in dns_results if result.exists]

            # Show validation statistics
            stats = validator.get_validation_stats(dns_results)
            print_validation_stats(stats)

            if output_file:
                _output_items_to_file(existing_subdomains, output_file)
                success(f"Saved {len(existing_subdomains)} validated names to: {output_file}")

        if check_http and validate and existing_subdomains:
            section_header("HTTP Validation")
            status(f"Validating {len(existing_subdomains)} subdomains via HTTP")

            http_validator = HTTPValidator(
                timeout=config.request_timeout,
                max_concurrent=config.max_concurrent_requests,
                verify_ssl=config.http_verify_ssl,
            )
            http_results = http_validator.validate_subdomains(
                existing_subdomains, check_both_protocols=True
            )

            # Show HTTP validation statistics
            http_stats = http_validator.get_validation_stats(http_results)
            print_validation_stats(http_stats)

        # Show debug information if requested
        if debug:
            print_debug_info(generation_result.metadata)

    except Exception as e:
        error(f"Generation failed: {safe_exception_text(e)}")
        ctx.exit(1)


@generate.command("path")
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Input file with known paths",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    help="Output file (default: stdout)",
)
@click.option(
    "-c",
    "--count",
    default=50,
    show_default=True,
    type=click.IntRange(min=1, max=5000),
    help="Number to generate",
)
@click.option("--provider", help="AI provider name; auto-detected when omitted")
@click.option("--model", help="Specific model to use")
@click.option("--debug", is_flag=True, help="Show debug info")
@click.pass_context
def generate_path(ctx, input_file, output_file, count, provider, model, debug):
    """Generate API paths from existing patterns.

    \b
    Examples:
      enumeraite generate path -i known.txt -o fuzz.txt -c 100
      enumeraite generate path -i paths.txt | ffuf -w - -u https://target/FUZZ
      enumeraite generate path -i paths.txt --provider huggingface
    """
    # Load known paths
    known_paths = _load_items_from_file(input_file)
    if not known_paths:
        error("No paths found in input file")
        ctx.exit(1)

    # Load configuration and setup provider
    _, ai_provider = _setup_provider(ctx, provider, model, "path")

    # Setup engine
    engine = GenerationEngine(ai_provider)

    # Print startup banner on first use
    print_startup_banner()

    # Print generation start info
    print_generation_start(count, ai_provider.get_provider_name(), "path")

    try:
        # Generate paths
        path_strategy = PathDiscoveryStrategy()
        generation_result = engine.generate_paths(known_paths, "", count, strategy=path_strategy)

        # Persist first so the UI never reports success for a failed write.
        if output_file:
            _output_items_to_file(generation_result.paths, output_file)
        print_results(generation_result.paths, output_file, saved=bool(output_file))

        # Show debug information if requested
        if debug:
            print_debug_info(generation_result.metadata)

    except Exception as e:
        error(f"Generation failed: {safe_exception_text(e)}")
        ctx.exit(1)


# Analyze command group
@click.group()
def analyze():
    """Deep AI analysis of individual targets for pattern discovery."""
    pass


@analyze.command("subdomain")
@click.argument("subdomain_input", required=True, metavar="SUBDOMAIN")
@click.option(
    "-c",
    "--count",
    default=20,
    show_default=True,
    type=click.IntRange(min=1, max=5000),
    help="Number of variants",
)
@click.option("--provider", help="AI provider name; auto-detected when omitted")
@click.option("--model", help="Specific model to use")
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    help="Output file (default: stdout)",
)
@click.option("--debug", is_flag=True, help="Show debug info")
@click.pass_context
def analyze_subdomain(ctx, subdomain_input, count, provider, model, output_file, debug):
    """Analyze subdomain patterns and generate variants.

    Decomposes complex naming patterns and generates realistic variants.

    \b
    Examples:
      enumeraite analyze subdomain api-prod-us1.example.com
      enumeraite analyze subdomain api-us1.example.com -c 30 -o out.txt
    """
    # Validate subdomain format
    if not SUBDOMAIN_SPEC.validator(subdomain_input):
        error("Please provide a valid full subdomain (e.g., 'api-us1.example.com')")
        ctx.exit(1)

    # Load configuration and setup provider
    _, ai_provider = _setup_provider(ctx, provider, model, "subdomain")

    # Setup pattern analysis engine
    pattern_engine = PatternAnalysisEngine(ai_provider)

    # Print startup banner
    print_startup_banner()

    try:
        # Perform pattern analysis
        result = pattern_engine.analyze_pattern(subdomain_input, variant_count=count)

        # Display the analysis results
        print_pattern_analysis(result, debug)

        # Handle output
        if result.generated_variants:
            if output_file:
                _output_items_to_file(result.generated_variants, output_file)
                success(f"Generated variants saved to: {output_file}")
            click.echo()
            success(
                f"Analysis complete! Generated {len(result.generated_variants)} variants from pattern analysis"
            )
        else:
            warning(
                "No valid variants generated. Try adjusting parameters or checking the input format"
            )

    except Exception as e:
        error(f"Pattern analysis failed: {safe_exception_text(e)}")
        ctx.exit(1)


@analyze.command("path")
@click.argument("path_input", required=True, metavar="PATH")
@click.option(
    "-f",
    "--function",
    "function_context",
    required=True,
    help='Functionality to find (e.g., "admin ops", "user deletion")',
)
@click.option(
    "-c",
    "--count",
    default=20,
    show_default=True,
    type=click.IntRange(min=1, max=5000),
    help="Number of variants",
)
@click.option("--provider", help="AI provider name; auto-detected when omitted")
@click.option("--model", help="Specific model to use")
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    help="Output file (default: stdout)",
)
@click.option("--debug", is_flag=True, help="Show debug info")
@click.pass_context
def analyze_path(ctx, path_input, function_context, count, provider, model, output_file, debug):
    """Analyze paths and generate function-specific endpoints.

    Finds related endpoints implementing specific functionality.

    \b
    Examples:
      enumeraite analyze path /api/v1/users -f "user deletion"
      enumeraite analyze path /api/v1/posts -f "admin ops" -c 25
    """
    # Validate path format
    if not PATH_SPEC.validator(path_input):
        error("Please provide a valid API path starting with '/' (e.g., '/api/v1/usr_crt')")
        ctx.exit(1)

    # Load configuration and setup provider
    _, ai_provider = _setup_provider(ctx, provider, model, "path")

    # Setup path function analysis engine
    path_engine = PathFunctionAnalysisEngine(ai_provider)

    # Print startup banner
    print_startup_banner()

    try:
        # Perform path function analysis
        result = path_engine.analyze_path_function(
            path_input, function_context, variant_count=count
        )

        # Display the analysis results
        print_path_analysis(result, debug)

        # Handle output
        if result.generated_paths:
            if output_file:
                _output_items_to_file(result.generated_paths, output_file)
                success(f"Generated paths saved to: {output_file}")
            click.echo()
            success(
                f"Analysis complete! Generated {len(result.generated_paths)} function-specific paths"
            )
        else:
            warning(
                "No valid paths generated. Try adjusting parameters or checking the input format"
            )

    except Exception as e:
        error(f"Path analysis failed: {safe_exception_text(e)}")
        ctx.exit(1)


# Utility functions
def _load_items_from_file(file_path: str) -> List[str]:
    """Load items from input file (paths or subdomains)."""
    try:
        return read_seed_file(file_path)
    except Exception as e:
        error(f"Failed to read input file: {safe_exception_text(e)}")
        return []


def _setup_provider(ctx, provider, model, generation_type=None):
    """Setup AI provider with configuration."""
    try:
        config = load_config()
        factory = ProviderFactory(config)
    except Exception as e:
        error(f"Configuration failed: {safe_exception_text(e)}")
        ctx.exit(1)
        raise RuntimeError("Click context exit unexpectedly returned")

    try:
        # Handle HuggingFace model selection first, then use normal factory
        if provider == "huggingface":
            # Override config for automatic model selection or explicit model
            provider_config = config.get_provider_config(provider)
            if provider_config:
                provider_config = provider_config.copy()

                if model:
                    provider_config["model"] = model
                elif generation_type and not provider_config.get("model"):
                    if generation_type == "path":
                        provider_config["model"] = "enumeraite/Enumeraite-x-Qwen3-4B-Path"
                    elif generation_type == "subdomain":
                        provider_config["model"] = "enumeraite/Enumeraite-x-Qwen3-4B-Subdomain"

                provider_class = factory._registry[provider]
                ai_provider = provider_class(provider_config)
            else:
                raise ValueError(f"No configuration found for provider '{provider}'")
        else:
            # Use normal factory logic for all other cases
            provider_name = provider or detect_provider(config)
            if not provider_name:
                raise ValueError("No usable provider detected; run 'enumeraite doctor'")
            ai_provider = factory.create_provider(provider_name)

            # Override model if specified via CLI (skip for HuggingFace as it uses object)
            if model and ai_provider.get_provider_name() != "huggingface":
                ai_provider.model = model

        return config, ai_provider
    except Exception as e:
        error(f"Provider setup failed: {safe_exception_text(e)}")
        ctx.exit(1)


def _output_simple(result, output_file):
    """Output simple item list."""
    if output_file:
        _output_items_to_file(result.paths, output_file)
    else:
        for path in result.paths:
            click.echo(path)


def _output_items_to_file(items, output_file):
    """Save items to a file."""
    Path(output_file).write_text(
        "".join(f"{item}\n" for item in items),
        encoding="utf-8",
    )


# Debug information is now handled by print_debug_info from UI module


# Pattern and path analysis display functions are now handled by the UI module
