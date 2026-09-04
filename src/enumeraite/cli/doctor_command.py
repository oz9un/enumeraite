"""Local configuration diagnostics for Enumeraite."""

from __future__ import annotations

import click

from ..core.config import load_config
from .detect import PROVIDER_PREFERENCE, _provider_usable, detect_provider
from .ui import Colors

ENV_HINTS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "ollama": "start Ollama on :11434",
    "vllm": "start vLLM on :8000",
    "llamacpp": "start llama.cpp on :8080",
    "local": "start Ollama on :11434",
    "huggingface": "pip install 'enumeraite[huggingface]'",
}


@click.command("doctor")
def doctor_command() -> None:
    """Check providers, local runtimes, confidence, and context settings."""
    config = load_config()
    click.echo("Enumeraite doctor")
    click.echo()
    for name in PROVIDER_PREFERENCE:
        usable = _provider_usable(config, name)
        mark = click.style("✓", fg=Colors.BRIGHT_GREEN) if usable else click.style("·", dim=True)
        suffix = "ready" if usable else ENV_HINTS.get(name, "not configured")
        click.echo(f"  {mark} {name:12} {suffix}")

    selected = detect_provider(config)
    click.echo()
    click.echo(f"Auto-selected provider: {selected or 'none'}")
    selected_config = config.get_provider_config(selected) if selected else {}
    click.echo(f"Confidence mode: {selected_config.get('confidence', 'model')}")
    click.echo(f"Context inference: {'on' if selected_config.get('context', True) else 'off'}")
    click.echo(f"Hybrid candidate pool: {'on' if selected_config.get('hybrid', True) else 'off'}")
    if selected:
        click.echo(click.style("Ready.", fg=Colors.BRIGHT_GREEN))
    else:
        click.echo(click.style("No usable provider found.", fg=Colors.BRIGHT_YELLOW))
        click.echo(
            "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, "
            "MINIMAX_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY, or TOGETHER_API_KEY"
        )
