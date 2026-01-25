"""Professional terminal UI utilities for enumeraite."""
import click
from typing import List, Dict, Any, Optional


# Color scheme - red-focused theme
class Colors:
    """Terminal color constants."""
    # Primary colors
    RED = 'red'
    BRIGHT_RED = 'bright_red'
    
    # Secondary colors
    WHITE = 'white'
    YELLOW = 'yellow'
    GREEN = 'green'
    CYAN = 'cyan'
    
    # Accent colors
    BRIGHT_GREEN = 'bright_green'
    BRIGHT_YELLOW = 'bright_yellow'
    BRIGHT_CYAN = 'bright_cyan'


def print_logo():
    """Display the enumeraite ASCII logo."""
    logo = """
                    +*
                    ++
                =++ +* +++
              =++   ++   ++*=
          =++       ++       +*+   +
         =+=                  *++
        ++       =++++++*+=     **
       ++  +   =++++    ***      **
            +  +++      ++=
   =========   ++*++++++++=   =====+++=+  +
       +=      *+*               =+
        ++      =***+*++++      +*=
        =+       =+++*+++       +=
          ===                +++
            ==+=    ++    =+=
               ===  =+  =++   +
               +    ++
                    ++
                    ++
"""
    click.echo(click.style(logo, fg=Colors.BRIGHT_RED), err=True)
    click.echo(click.style("  enumeraite ", fg=Colors.BRIGHT_RED, bold=True) + 
               click.style("v0.1.0", fg=Colors.RED), err=True)
    click.echo(err=True)


# ─────────────────────────────────────────────────────────────────
# Message helpers (all output to stderr for pipe compatibility)
# ─────────────────────────────────────────────────────────────────

def success(message: str) -> None:
    """Print success message to stderr."""
    click.echo(f"  {click.style('✓', fg=Colors.BRIGHT_GREEN, bold=True)} {message}", err=True)


def info(message: str) -> None:
    """Print info message to stderr."""
    click.echo(f"  {click.style('›', fg=Colors.RED)} {message}", err=True)


def warning(message: str) -> None:
    """Print warning message to stderr."""
    click.echo(f"  {click.style('!', fg=Colors.BRIGHT_YELLOW, bold=True)} {message}", err=True)


def error(message: str) -> None:
    """Print error message to stderr."""
    click.echo(f"  {click.style('✗', fg=Colors.BRIGHT_RED, bold=True)} {message}", err=True)


def status(message: str) -> None:
    """Print status message to stderr."""
    click.echo(f"  {click.style('→', fg=Colors.RED)} {message}", err=True)


# ─────────────────────────────────────────────────────────────────
# Section headers (all output to stderr for pipe compatibility)
# ─────────────────────────────────────────────────────────────────

def section_header(title: str) -> None:
    """Print section header with separator to stderr."""
    click.echo(err=True)
    click.echo(click.style(f"  ┌{'─'*56}┐", fg=Colors.RED), err=True)
    click.echo(click.style(f"  │ {title.upper():54} │", fg=Colors.BRIGHT_RED, bold=True), err=True)
    click.echo(click.style(f"  └{'─'*56}┘", fg=Colors.RED), err=True)


def subsection_header(title: str) -> None:
    """Print subsection header to stderr."""
    click.echo(err=True)
    click.echo(f"  {click.style('──', fg=Colors.RED)} {click.style(title, fg=Colors.BRIGHT_RED, bold=True)}", err=True)


# ─────────────────────────────────────────────────────────────────
# Generation output (UI to stderr, data to stdout for piping)
# ─────────────────────────────────────────────────────────────────

def print_generation_start(count: int, provider: str, mode: str) -> None:
    """Print generation start banner to stderr."""
    section_header(f"{mode} Generation")
    info(f"Provider   {click.style(provider, fg=Colors.WHITE, bold=True)}")
    info(f"Target     {click.style(str(count), fg=Colors.WHITE, bold=True)} items")
    click.echo(err=True)


def print_results(items: List[str], output_file: Optional[str] = None) -> None:
    """Print generated results - raw items to stdout, UI to stderr."""
    if not items:
        warning("No valid items generated")
        return

    subsection_header(f"Results ({len(items)} items)")
    click.echo(err=True)

    # Print raw items to stdout (for piping to ffuf, etc.)
    # Print formatted items to stderr (for user visibility)
    for i, item in enumerate(items, 1):
        # Raw item to stdout for piping
        click.echo(item)
        # Formatted item to stderr for user
        num = click.style(f"{i:3d}", fg=Colors.RED)
        click.echo(f"    {num}  {click.style(item, fg=Colors.WHITE)}", err=True)

    click.echo(err=True)
    
    if output_file:
        success(f"Saved to {click.style(output_file, fg=Colors.WHITE, bold=True)}")


# ─────────────────────────────────────────────────────────────────
# Validation output (all to stderr)
# ─────────────────────────────────────────────────────────────────

def print_validation_stats(stats: Dict[str, Any]) -> None:
    """Print validation statistics to stderr."""
    subsection_header("Validation")
    click.echo(err=True)

    # DNS Stats
    if 'existing' in stats:
        found = stats.get('existing', 0)
        not_found = stats.get('not_found', 0)
        errors = stats.get('errors', 0)
        rate = stats.get('success_rate', 0)
        avg_time = stats.get('avg_response_time', 0)
        
        click.echo(f"    {click.style('Found', fg=Colors.WHITE):20} {click.style(str(found), fg=Colors.BRIGHT_GREEN, bold=True)}", err=True)
        click.echo(f"    {click.style('Not found', fg=Colors.WHITE):20} {click.style(str(not_found), fg=Colors.YELLOW)}", err=True)
        if errors > 0:
            click.echo(f"    {click.style('Errors', fg=Colors.WHITE):20} {click.style(str(errors), fg=Colors.BRIGHT_RED)}", err=True)
        click.echo(f"    {click.style('Success rate', fg=Colors.WHITE):20} {click.style(f'{rate:.1f}%', fg=Colors.WHITE, bold=True)}", err=True)
        click.echo(f"    {click.style('Avg response', fg=Colors.WHITE):20} {avg_time:.3f}s", err=True)

    # HTTP Stats
    if 'accessible' in stats:
        accessible = stats.get('accessible', 0)
        https = stats.get('https_available', 0)
        rate = stats.get('success_rate', 0)
        
        click.echo(f"    {click.style('Accessible', fg=Colors.WHITE):20} {click.style(str(accessible), fg=Colors.BRIGHT_GREEN, bold=True)}", err=True)
        click.echo(f"    {click.style('HTTPS available', fg=Colors.WHITE):20} {click.style(str(https), fg=Colors.WHITE)}", err=True)
        click.echo(f"    {click.style('Success rate', fg=Colors.WHITE):20} {click.style(f'{rate:.1f}%', fg=Colors.WHITE, bold=True)}", err=True)


# ─────────────────────────────────────────────────────────────────
# Analysis output (UI to stderr, data to stdout for piping)
# ─────────────────────────────────────────────────────────────────

def print_analysis_header(target: str, analysis_type: str) -> None:
    """Print analysis header to stderr."""
    section_header(f"{analysis_type} Analysis")
    info(f"Target     {click.style(target, fg=Colors.WHITE, bold=True)}")


def _format_confidence(confidence: float) -> str:
    """Format confidence score with color coding."""
    if confidence >= 0.8:
        color = Colors.BRIGHT_GREEN
        label = "HIGH"
    elif confidence >= 0.5:
        color = Colors.BRIGHT_YELLOW
        label = "MEDIUM"
    else:
        color = Colors.BRIGHT_RED
        label = "LOW"
    
    bar_filled = int(confidence * 10)
    bar_empty = 10 - bar_filled
    bar = click.style('█' * bar_filled, fg=color) + click.style('░' * bar_empty, dim=True)
    
    return f"{bar} {click.style(f'{confidence:.0%}', fg=color, bold=True)} {click.style(label, fg=color)}"


def print_pattern_analysis(result, show_details: bool = False) -> None:
    """Print pattern analysis results - raw items to stdout, UI to stderr."""
    print_analysis_header(result.original_subdomain, "Pattern")
    
    click.echo(err=True)
    info(f"Confidence {_format_confidence(result.confidence_score)}")
    
    if result.pattern_template:
        info(f"Pattern    {click.style(result.pattern_template, fg=Colors.WHITE, bold=True)}")

    # AI Reasoning
    if result.reasoning:
        subsection_header("AI Analysis")
        click.echo(err=True)
        # Wrap text nicely
        for line in result.reasoning.split('\n'):
            click.echo(f"    {line}", err=True)

    # Component breakdown (if requested)
    if show_details and result.decomposition:
        subsection_header("Components")
        click.echo(err=True)
        for i, comp in enumerate(result.decomposition, 1):
            num = click.style(f"{i}.", fg=Colors.RED)
            value = click.style(comp.value, fg=Colors.WHITE, bold=True)
            comp_type = click.style(f"({comp.type})", dim=True)
            click.echo(f"    {num} {value} {comp_type}", err=True)
            click.echo(f"       {comp.description}", err=True)
            if comp.alternatives:
                alts = ", ".join(comp.alternatives[:5])
                if len(comp.alternatives) > 5:
                    alts += "..."
                click.echo(f"       {click.style('Alternatives:', fg=Colors.RED)} {alts}", err=True)
            click.echo(err=True)

    # Generated variants - raw to stdout, formatted to stderr
    if result.generated_variants:
        subsection_header(f"Generated Variants ({len(result.generated_variants)})")
        click.echo(err=True)
        for i, variant in enumerate(result.generated_variants, 1):
            # Raw variant to stdout for piping
            click.echo(variant)
            # Formatted variant to stderr for user
            num = click.style(f"{i:3d}", fg=Colors.RED)
            click.echo(f"    {num}  {click.style(variant, fg=Colors.WHITE)}", err=True)


def print_path_analysis(result, show_details: bool = False) -> None:
    """Print path analysis results - raw items to stdout, UI to stderr."""
    print_analysis_header(result.original_path, "Path Function")
    
    click.echo(err=True)
    info(f"Function   {click.style(result.function_context, fg=Colors.WHITE, bold=True)}")
    info(f"Confidence {_format_confidence(result.confidence_score)}")

    # Function analysis
    if result.function_analysis:
        subsection_header("Function Analysis")
        click.echo(err=True)
        for line in result.function_analysis.split('\n'):
            click.echo(f"    {line}", err=True)

    # AI Reasoning
    if result.reasoning:
        subsection_header("AI Reasoning")
        click.echo(err=True)
        for line in result.reasoning.split('\n'):
            click.echo(f"    {line}", err=True)

    # Path breakdown (if requested)
    if show_details and result.path_breakdown:
        subsection_header("Path Components")
        click.echo(err=True)
        for comp in result.path_breakdown:
            num = click.style(f"{comp.position + 1}.", fg=Colors.RED)
            value = click.style(comp.value, fg=Colors.WHITE, bold=True)
            comp_type = click.style(f"({comp.type})", dim=True)
            click.echo(f"    {num} {value} {comp_type}", err=True)
            click.echo(f"       {comp.description}", err=True)
            click.echo(err=True)

    # Generated paths - raw to stdout, formatted to stderr
    if result.generated_paths:
        subsection_header(f"Generated Paths ({len(result.generated_paths)})")
        click.echo(err=True)
        for i, path in enumerate(result.generated_paths, 1):
            # Raw path to stdout for piping
            click.echo(path)
            # Formatted path to stderr for user
            num = click.style(f"{i:3d}", fg=Colors.RED)
            click.echo(f"    {num}  {click.style(path, fg=Colors.WHITE)}", err=True)


# ─────────────────────────────────────────────────────────────────
# Debug output (all to stderr)
# ─────────────────────────────────────────────────────────────────

def print_debug_info(metadata: Dict[str, Any]) -> None:
    """Print debug information to stderr."""
    subsection_header("Debug Info")
    click.echo(err=True)

    provider = metadata.get('provider', 'unknown')
    model = metadata.get('model', 'unknown')
    
    click.echo(f"    {click.style('Provider', fg=Colors.WHITE):20} {provider}", err=True)
    click.echo(f"    {click.style('Model', fg=Colors.WHITE):20} {model}", err=True)

    # Token usage if available
    if "token_usage" in metadata:
        usage = metadata["token_usage"]
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        click.echo(err=True)
        click.echo(f"    {click.style('Input tokens', fg=Colors.WHITE):20} {input_tokens:,}", err=True)
        click.echo(f"    {click.style('Output tokens', fg=Colors.WHITE):20} {output_tokens:,}", err=True)
        click.echo(f"    {click.style('Total tokens', fg=Colors.WHITE):20} {click.style(f'{total_tokens:,}', bold=True)}", err=True)

        # Cost estimation
        cost = None
        if provider.lower() == 'claude' and 'sonnet' in model.lower():
            input_cost = (input_tokens / 1_000_000) * 3
            output_cost = (output_tokens / 1_000_000) * 15
            cost = input_cost + output_cost
        elif provider.lower() == 'openai' and 'gpt-4' in model.lower():
            input_cost = (input_tokens / 1_000_000) * 30
            output_cost = (output_tokens / 1_000_000) * 60
            cost = input_cost + output_cost
        
        if cost is not None:
            click.echo(f"    {click.style('Est. cost', fg=Colors.WHITE):20} {click.style(f'${cost:.6f}', fg=Colors.BRIGHT_GREEN)}", err=True)


# ─────────────────────────────────────────────────────────────────
# Startup (all to stderr)
# ─────────────────────────────────────────────────────────────────

def print_startup_banner() -> None:
    """Print startup banner to stderr."""
    print_logo()
    click.echo(click.style("  AI-powered enumeration for security professionals", dim=True), err=True)
    click.echo(err=True)
