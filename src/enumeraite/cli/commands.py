"""CLI commands for Enumeraite."""
import click
import json
import asyncio
from pathlib import Path
from typing import List

from ..core.config import load_config
from ..core.factory import ProviderFactory
from ..core.engine import GenerationEngine
from ..core.validator import HTTPValidator

@click.command()
@click.option('--target', required=True, help='Target domain (e.g., example.com)')
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file containing known paths')
@click.option('--output', 'output_file', type=click.Path(),
              help='Output file for generated paths')
@click.option('--count', default=50, help='Number of paths to generate')
@click.option('--provider', help='AI provider to use (openai, claude)')
@click.option('--advanced', is_flag=True, help='Output detailed JSON format')
@click.option('--validate', is_flag=True, help='Validate generated paths via HTTP')
@click.option('--methods', default='GET', help='HTTP methods to test (comma-separated)')
def batch(target, input_file, output_file, count, provider, advanced, validate, methods):
    """Generate paths in batch mode."""
    # Load known paths
    known_paths = _load_paths_from_file(input_file)
    if not known_paths:
        click.echo("No paths found in input file", err=True)
        return

    # Load configuration
    try:
        config = load_config()
        factory = ProviderFactory(config)
    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        return

    # Get provider
    try:
        if provider:
            ai_provider = factory.create_provider(provider)
        else:
            ai_provider = factory.get_default_provider()
    except Exception as e:
        click.echo(f"Provider error: {e}", err=True)
        return

    # Setup engine
    validator = HTTPValidator() if validate else None
    engine = GenerationEngine(ai_provider, validator)

    click.echo(f"Generating {count} paths for {target} using {ai_provider.get_provider_name()}")

    try:
        if validate:
            # Async validation
            http_methods = [m.strip().upper() for m in methods.split(',')]
            generation_result, validation_results = asyncio.run(
                engine.generate_and_validate_paths(known_paths, target, count, http_methods)
            )
        else:
            # Generation only
            generation_result = engine.generate_paths(known_paths, target, count)
            validation_results = []

        # Output results
        if advanced:
            _output_advanced(generation_result, validation_results, output_file)
        else:
            _output_simple(generation_result, output_file)

        click.echo(f"Generated {len(generation_result.paths)} paths")
        if validation_results:
            valid_count = sum(1 for r in validation_results if r.exists)
            click.echo(f"Validated {len(validation_results)} paths, {valid_count} appear to exist")

    except Exception as e:
        click.echo(f"Generation error: {e}", err=True)

@click.command()
@click.option('--target', required=True, help='Target domain')
@click.option('--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file containing known paths (will be updated)')
@click.option('--duration', default='30m', help='How long to run (e.g., 30m, 2h)')
@click.option('--batch-size', default=20, help='Paths to generate per round')
@click.option('--max-empty-rounds', default=5, help='Stop after N rounds with no discoveries')
@click.option('--provider', help='AI provider to use')
@click.option('--methods', default='GET', help='HTTP methods to test')
def continuous(target, input_file, duration, batch_size, max_empty_rounds, provider, methods):
    """Run continuous discovery mode with live path updates."""
    click.echo("Continuous mode will be implemented in the next task...")
    click.echo(f"Would run discovery for {target} for {duration}")

def _load_paths_from_file(file_path: str) -> List[str]:
    """Load paths from input file.

    Args:
        file_path: Path to file containing known paths

    Returns:
        List of paths (comments and empty lines filtered out)
    """
    paths = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path and not path.startswith('#'):
                    paths.append(path)
    except Exception as e:
        click.echo(f"Error reading input file: {e}", err=True)
    return paths

def _output_simple(result, output_file):
    """Output simple path list.

    Args:
        result: GenerationResult object
        output_file: Output file path (None for stdout)
    """
    if output_file:
        try:
            with open(output_file, 'w') as f:
                for path in result.paths:
                    f.write(f"{path}\n")
        except Exception as e:
            click.echo(f"Error writing output file: {e}", err=True)
    else:
        for path in result.paths:
            click.echo(path)

def _output_advanced(generation_result, validation_results, output_file):
    """Output detailed JSON format.

    Args:
        generation_result: GenerationResult object
        validation_results: List of ValidationResult objects
        output_file: Output file path (None for stdout)
    """
    # Create validation lookup by extracting path from URL
    validation_lookup = {}
    for val_result in validation_results:
        # Extract path from full URL for matching
        if val_result.path.startswith('http'):
            path = '/' + val_result.path.split('/', 3)[3] if len(val_result.path.split('/', 3)) > 3 else '/'
        else:
            path = val_result.path
        validation_lookup[path] = val_result

    output_data = {
        "generated_count": len(generation_result.paths),
        "metadata": generation_result.metadata,
        "generated_paths": []
    }

    for path, score in zip(generation_result.paths, generation_result.confidence_scores):
        path_data = {
            "path": path,
            "confidence": round(score, 3)
        }

        # Add validation data if available
        if path in validation_lookup:
            val = validation_lookup[path]
            path_data.update({
                "verified": val.exists,
                "status": val.status_code,
                "method": val.method,
                "response_time": round(val.response_time, 3) if val.response_time else None
            })

        output_data["generated_paths"].append(path_data)

    json_output = json.dumps(output_data, indent=2)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(json_output)
        except Exception as e:
            click.echo(f"Error writing output file: {e}", err=True)
    else:
        click.echo(json_output)