"""Main CLI entry point for Enumeraite."""

import sys

import click

from .. import __version__
from .bench_commands import bench_command
from .commands import analyze, generate
from .doctor_command import doctor_command
from .royale_command import royale_command
from .run_command import run_command
from .ui import Colors, print_startup_banner


class EnumeraiteGroup(click.Group):
    """Custom group class with professional help formatting."""

    def format_help(self, ctx, _formatter):
        """Custom help formatting with sections."""
        # Show banner first
        if "--help" in sys.argv or (len(sys.argv) == 1 and sys.argv[0].endswith("enumeraite")):
            print_startup_banner()
        click.echo(
            click.style(
                "Enumeraite — AI-assisted web attack-surface enumeration",
                fg=Colors.BRIGHT_RED,
                bold=True,
            )
        )
        click.echo()

        # Commands section
        self._print_commands(ctx)

        # Custom sections
        self._print_examples()
        self._print_providers()
        self._print_footer()

        # Options at the end
        click.echo(click.style("Options:", fg=Colors.BRIGHT_RED, bold=True))
        click.echo()
        click.echo(f"  {click.style('--version', fg=Colors.RED)}  Show version and exit")
        click.echo(f"  {click.style('--help', fg=Colors.RED)}     Show this message and exit")
        click.echo()

    def _print_commands(self, ctx):
        """Custom command formatting."""
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands.append((subcommand, cmd))

        if commands:
            click.echo(click.style("Commands:", fg=Colors.BRIGHT_RED, bold=True))
            click.echo()

            for name, cmd in commands:
                help_text = cmd.get_short_help_str(limit=55)
                click.echo(f"  {click.style(name, fg=Colors.BRIGHT_RED, bold=True):12} {help_text}")
            click.echo()

    def _print_examples(self):
        """Print quick start examples."""
        click.echo(click.style("Examples:", fg=Colors.BRIGHT_RED, bold=True))
        click.echo()

        examples = [
            ("enumeraite", "run", "seeds.txt", "Auto-detect kind and provider"),
            (
                "enumeraite",
                "run",
                "subs.txt | httpx",
                "Pipe clean candidates to another tool",
            ),
            ("enumeraite", "doctor", "", "Check provider setup"),
            (
                "enumeraite",
                "generate subdomain",
                "-i subs.txt --validate",
                "Generate subdomains with DNS validation",
            ),
            (
                "enumeraite",
                "generate path",
                "-i paths.txt -o fuzz.txt",
                "Generate paths for fuzzing",
            ),
            (
                "enumeraite",
                "analyze subdomain",
                "api-us1.example.com",
                "Pattern analysis of a subdomain",
            ),
            (
                "enumeraite",
                "analyze path",
                "/api/v1/users -f 'admin'",
                "Find admin endpoints",
            ),
        ]

        for tool, cmd, args, desc in examples:
            click.echo(
                f"  {click.style('$', fg=Colors.RED)} {click.style(tool, fg=Colors.BRIGHT_RED)} {click.style(cmd, fg=Colors.RED)} {args}"
            )
            click.echo(f"    {click.style(desc, dim=True)}")
        click.echo()

    def _print_providers(self):
        """Print provider info."""
        click.echo(click.style("Providers:", fg=Colors.BRIGHT_RED, bold=True))
        click.echo()

        click.echo(
            f"  {click.style('anthropic', fg=Colors.BRIGHT_RED):14} requires {click.style('ANTHROPIC_API_KEY', fg=Colors.YELLOW)}"
        )
        click.echo(
            f"  {click.style('openai', fg=Colors.BRIGHT_RED):14} requires {click.style('OPENAI_API_KEY', fg=Colors.YELLOW)}"
        )
        click.echo(
            f"  {click.style('minimax', fg=Colors.BRIGHT_RED):14} requires {click.style('MINIMAX_API_KEY', fg=Colors.YELLOW)}"
        )
        click.echo(f"  {click.style('ollama', fg=Colors.BRIGHT_RED):14} auto-detected on localhost")
        click.echo(
            f"  {click.style('openrouter', fg=Colors.BRIGHT_RED):14} requires {click.style('OPENROUTER_API_KEY', fg=Colors.YELLOW)}"
        )
        click.echo(
            f"  {click.style('huggingface', fg=Colors.BRIGHT_RED):14} optional heavy local benchmark control"
        )
        click.echo()

    def _print_footer(self):
        """Print footer."""
        click.echo(click.style("─" * 65, fg=Colors.RED, dim=True))
        click.echo(
            f"  Compatible with {click.style('ffuf', fg=Colors.RED)}, {click.style('dirb', fg=Colors.RED)}, {click.style('gobuster', fg=Colors.RED)}, {click.style('nuclei', fg=Colors.RED)}"
        )
        click.echo(f"  Config: {click.style('enumeraite.json', fg=Colors.RED)} or env vars")
        click.echo()


@click.group(cls=EnumeraiteGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="enumeraite")
@click.pass_context
def cli(ctx):
    """Enumeraite — AI-assisted web attack-surface enumeration.

    Intelligently generate attack surfaces by learning patterns from your data.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Add the command groups
cli.add_command(generate)
cli.add_command(analyze)
cli.add_command(run_command)
cli.add_command(doctor_command)
cli.add_command(bench_command)
cli.add_command(royale_command)


if __name__ == "__main__":
    cli()
