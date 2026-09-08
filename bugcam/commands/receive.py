"""CLI command for starting the DOT receiver server."""

import typer
import logging
from rich.console import Console

from ..receiver import create_app, start_tracker_finalization
from ..receiver.config import RECEIVER_DEFAULT_PORT, RECEIVER_DEFAULT_HOST

app = typer.Typer(help="Manage DOT data receiver server")
console = Console()

logger = logging.getLogger(__name__)


@app.command("start")
def start_receiver(
    port: int = typer.Option(RECEIVER_DEFAULT_PORT, "--port", "-p", help="HTTP server port"),
    host: str = typer.Option(RECEIVER_DEFAULT_HOST, "--host", "-h", help="Bind address"),
    debug: bool = typer.Option(
        False, "--debug",
        help="Enable Flask/Werkzeug debug mode (interactive tracebacks + "
             "auto-reload on file changes). Safe here since this command "
             "runs only the receiver, standalone; `bugcam run`'s embedded "
             "receiver never exposes this, since a reload would restart "
             "the whole process mid-recording.",
    ),
) -> None:
    """Start the DOT data receiver server."""
    console.print("[cyan]Starting DOT receiver server...[/cyan]")
    console.print(f"[dim]Host: {host}, Port: {port}[/dim]")

    try:
        flask_app = create_app(config={"port": port, "host": host})

        tracker = flask_app.config.get("TRACKER")
        if tracker:
            start_tracker_finalization(tracker)

        console.print("[green]✓ DOT receiver started[/green]")
        console.print(f"[dim]Endpoints available at http://{host}:{port}[/dim]")

        flask_app.run(host=host, port=port, threaded=True, debug=debug)

    except Exception as e:
        console.print(f"[red]Error starting receiver: {e}[/red]")
        raise typer.Exit(1)
