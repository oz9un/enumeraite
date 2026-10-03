"""Launch the local browser control room."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import click


@click.command("ui")
@click.option(
    "--port",
    type=click.IntRange(0, 65535),
    default=8765,
    show_default=True,
    help="Local port; 0 chooses an available port.",
)
@click.option(
    "--open/--no-open", "open_browser", default=True, help="Open the dashboard in your browser."
)
@click.option(
    "--state-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Private directory for dashboard checkpoints (never API keys).",
)
def ui_command(port: int, open_browser: bool, state_dir: Path | None) -> None:
    """Start scans and watch workers and discoveries in a local dashboard.

    Local use only: do not publish or tunnel this server. No dashboard login
    is required. Provider keys entered in the dashboard stay in process memory.
    """
    from ..web.manager import ScanManager
    from ..web.server import DashboardServer

    manager = ScanManager(storage_dir=state_dir)
    try:
        server = DashboardServer(manager, port=port)
    except OSError as exc:
        manager.close()
        raise click.ClickException("Could not open local port; try enumeraite ui --port 0") from exc
    try:
        click.echo("Enumeraite control room — local access only")
        click.echo(server.url)
        click.echo("No dashboard login needed. Ctrl-C pauses work and stops the dashboard.")
        if open_browser:
            try:
                webbrowser.open(server.url)
            except Exception:
                click.echo("Open the printed URL in your browser.", err=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        click.echo("\nPausing active work and closing the dashboard…", err=True)
    finally:
        server.server_close()
        manager.close()
