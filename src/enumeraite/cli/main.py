"""Main CLI entry point for Enumeraite."""
import click
from .commands import batch, continuous

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Enumeraite: AI-Assisted Web Attack Surface Enumeration

    Generate new API paths from known endpoints using AI pattern recognition.
    """
    pass

cli.add_command(batch)
cli.add_command(continuous)

if __name__ == '__main__':
    cli()