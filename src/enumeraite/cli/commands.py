"""Professional CLI commands for Enumeraite."""
import click
import json
from pathlib import Path
from typing import List

from ..core.config import load_config
from ..core.factory import ProviderFactory
from ..core.engine import GenerationEngine
from ..core.strategies import PathDiscoveryStrategy, SubdomainDiscoveryStrategy
from ..core.dns_validator import DNSValidator
from ..core.http_validator import HTTPValidator
from ..core.pattern_analysis import PatternAnalysisEngine
from ..core.path_function_analysis import PathFunctionAnalysisEngine
from .ui import (
    print_startup_banner, success, info, warning, error, status,
    section_header, print_generation_start, print_results,
    print_validation_stats, print_pattern_analysis,
    print_path_analysis, print_debug_info
)


# Generate command group
@click.group()
def generate():
    """Bulk generation from wordlists with AI pattern recognition."""
    pass


@generate.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file with known subdomains')
@click.option('-o', '--output', 'output_file', type=click.Path(),
              help='Output file (default: stdout)')
@click.option('-c', '--count', default=50, show_default=True, help='Number to generate')
@click.option('--provider', help='AI provider (claude, openai, huggingface)')
@click.option('--model', help='Specific model to use')
@click.option('--validate', is_flag=True, help='Enable DNS validation')
@click.option('--check-http', is_flag=True, help='Check HTTP (requires --validate)')
@click.option('--debug', is_flag=True, help='Show debug info')
@click.pass_context
def subdomain(ctx, input_file, output_file, count, provider, model, validate, check_http, debug):
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
        if not '.' in subdomain or subdomain.startswith('/'):
            warning(f"Input should contain full subdomains (e.g., 'api.example.com'), got: {subdomain}")
            break

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

    try:
        # Generate subdomains
        subdomain_strategy = SubdomainDiscoveryStrategy()
        generation_result = engine.generate_paths(known_subdomains, count, strategy=subdomain_strategy)

        # Output results
        print_results(generation_result.paths, output_file)

        # DNS validation
        if validate:
            section_header("DNS Validation")
            status(f"Validating {len(generation_result.paths)} subdomains")

            validator = DNSValidator(timeout=5.0, max_concurrent=20)
            dns_results = validator.validate_subdomains(generation_result.paths)

            # Filter to only existing subdomains
            existing_subdomains = [result.subdomain for result in dns_results if result.exists]

            # Show validation statistics
            stats = validator.get_validation_stats(dns_results)
            print_validation_stats(stats)

            if existing_subdomains and output_file:
                _output_items_to_file(existing_subdomains, output_file)

        if check_http and validate and existing_subdomains:
            section_header("HTTP Validation")
            status(f"Validating {len(existing_subdomains)} subdomains via HTTP")

            http_validator = HTTPValidator(timeout=10.0, max_concurrent=15)
            http_results = http_validator.validate_subdomains(existing_subdomains, check_both_protocols=True)

            # Show HTTP validation statistics
            http_stats = http_validator.get_validation_stats(http_results)
            print_validation_stats(http_stats)

        # Show debug information if requested
        if debug:
            print_debug_info(generation_result.metadata)

    except Exception as e:
        error(f"Generation failed: {e}")
        ctx.exit(1)


@generate.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file with known paths')
@click.option('-o', '--output', 'output_file', type=click.Path(),
              help='Output file (default: stdout)')
@click.option('-c', '--count', default=50, show_default=True, help='Number to generate')
@click.option('--provider', help='AI provider (claude, openai, huggingface)')
@click.option('--model', help='Specific model to use')
@click.option('--debug', is_flag=True, help='Show debug info')
@click.pass_context
def path(ctx, input_file, output_file, count, provider, model, debug):
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
    config, ai_provider = _setup_provider(ctx, provider, model, "path")

    # Setup engine
    engine = GenerationEngine(ai_provider)

    # Print startup banner on first use
    print_startup_banner()

    # Print generation start info
    print_generation_start(count, ai_provider.get_provider_name(), "path")

    try:
        # Generate paths
        path_strategy = PathDiscoveryStrategy()
        generation_result = engine.generate_paths(known_paths, count, strategy=path_strategy)

        # Output results
        print_results(generation_result.paths, output_file)

        # Show debug information if requested
        if debug:
            print_debug_info(generation_result.metadata)

    except Exception as e:
        error(f"Generation failed: {e}")
        ctx.exit(1)


# Analyze command group
@click.group()
def analyze():
    """Deep AI analysis of individual targets for pattern discovery."""
    pass


@analyze.command()
@click.argument('subdomain_input', required=True, metavar='SUBDOMAIN')
@click.option('-c', '--count', default=20, show_default=True, help='Number of variants')
@click.option('--provider', help='AI provider (claude, openai, huggingface)')
@click.option('--model', help='Specific model to use')
@click.option('-o', '--output', 'output_file', type=click.Path(),
              help='Output file (default: stdout)')
@click.option('--debug', is_flag=True, help='Show debug info')
@click.pass_context
def subdomain(ctx, subdomain_input, count, provider, model, output_file, debug):
    """Analyze subdomain patterns and generate variants.

    Decomposes complex naming patterns and generates realistic variants.

    \b
    Examples:
      enumeraite analyze subdomain api-prod-us1.example.com
      enumeraite analyze subdomain api-us1.example.com -c 30 -o out.txt
    """
    # Validate subdomain format
    if not '.' in subdomain_input:
        error("Please provide a full subdomain (e.g., 'activateiphone-use1-cx02.example.com')")
        ctx.exit(1)

    # Load configuration and setup provider
    config, ai_provider = _setup_provider(ctx, provider, model, "subdomain")

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
            success(f"Analysis complete! Generated {len(result.generated_variants)} variants from pattern analysis")
        else:
            warning("No valid variants generated. Try adjusting parameters or checking the input format")

    except Exception as e:
        error(f"Pattern analysis failed: {e}")
        ctx.exit(1)


@analyze.command()
@click.argument('path_input', required=True, metavar='PATH')
@click.option('-f', '--function', 'function_context', required=True,
              help='Functionality to find (e.g., "admin ops", "user deletion")')
@click.option('-c', '--count', default=20, show_default=True, help='Number of variants')
@click.option('--provider', help='AI provider (claude, openai, huggingface)')
@click.option('--model', help='Specific model to use')
@click.option('-o', '--output', 'output_file', type=click.Path(),
              help='Output file (default: stdout)')
@click.option('--debug', is_flag=True, help='Show debug info')
@click.pass_context
def path(ctx, path_input, function_context, count, provider, model, output_file, debug):
    """Analyze paths and generate function-specific endpoints.

    Finds related endpoints implementing specific functionality.

    \b
    Examples:
      enumeraite analyze path /api/v1/users -f "user deletion"
      enumeraite analyze path /api/v1/posts -f "admin ops" -c 25
    """
    # Validate path format
    if not path_input.startswith('/'):
        error("Please provide a valid API path starting with '/' (e.g., '/api/v1/usr_crt')")
        ctx.exit(1)

    # Load configuration and setup provider
    config, ai_provider = _setup_provider(ctx, provider, model, "path")

    # Setup path function analysis engine
    path_engine = PathFunctionAnalysisEngine(ai_provider)

    # Print startup banner
    print_startup_banner()

    try:
        # Perform path function analysis
        result = path_engine.analyze_path_function(path_input, function_context, variant_count=count)

        # Display the analysis results
        print_path_analysis(result, debug)

        # Handle output
        if result.generated_paths:
            if output_file:
                _output_items_to_file(result.generated_paths, output_file)
                success(f"Generated paths saved to: {output_file}")
            click.echo()
            success(f"Analysis complete! Generated {len(result.generated_paths)} function-specific paths")
        else:
            warning("No valid paths generated. Try adjusting parameters or checking the input format")

    except Exception as e:
        error(f"Path analysis failed: {e}")
        ctx.exit(1)


# Utility functions
def _load_items_from_file(file_path: str) -> List[str]:
    """Load items from input file (paths or subdomains)."""
    items = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                item = line.strip()
                if item and not item.startswith('#'):
                    items.append(item)
    except Exception as e:
        error(f"Failed to read input file: {e}")
    return items


def _setup_provider(ctx, provider, model, generation_type=None):
    """Setup AI provider with configuration."""
    try:
        config = load_config()
        factory = ProviderFactory(config)
    except Exception as e:
        error(f"Configuration failed: {e}")
        ctx.exit(1)

    try:
        # Handle HuggingFace model selection first, then use normal factory
        if provider == "huggingface":
            # Override config for automatic model selection or explicit model
            provider_config = config.get_provider_config(provider)
            if provider_config:
                provider_config = provider_config.copy()
                
                if model:
                    provider_config["model"] = model
                elif generation_type:
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
            if provider:
                ai_provider = factory.create_provider(provider)
            else:
                ai_provider = factory.get_default_provider()

            # Override model if specified via CLI (skip for HuggingFace as it uses object)
            if model and ai_provider.get_provider_name() != "huggingface":
                ai_provider.model = model

        return config, ai_provider
    except Exception as e:
        error(f"Provider setup failed: {e}")
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
    try:
        with open(output_file, 'w') as f:
            for item in items:
                f.write(f"{item}\n")
    except Exception as e:
        error(f"Failed to write output file: {e}")


# Debug information is now handled by print_debug_info from UI module


# Pattern and path analysis display functions are now handled by the UI module