"""CLI command for starting the DOT receiver server."""

import typer
import threading
import logging
from rich.console import Console

from ..receiver import create_app
from ..receiver.config import RECEIVER_DEFAULT_PORT, RECEIVER_DEFAULT_HOST
from ..receiver.tracker import finalization_loop

app = typer.Typer(help="Manage DOT data receiver server")
console = Console()

logger = logging.getLogger(__name__)


@app.command("start")
def start_receiver(
    port: int = typer.Option(RECEIVER_DEFAULT_PORT, "--port", "-p", help="HTTP server port"),
    host: str = typer.Option(RECEIVER_DEFAULT_HOST, "--host", "-h", help="Bind address"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """Start the DOT data receiver server."""
    console.print("[cyan]Starting DOT receiver server...[/cyan]")
    console.print(f"[dim]Host: {host}, Port: {port}[/dim]")

    try:
        flask_app = create_app(config={"port": port, "host": host})

        tracker = flask_app.config.get("TRACKER")
        if tracker:
            logger.info("Scanning for orphaned tracks...")
            tracker.recover_orphaned_tracks()

            stop_event = threading.Event()
            finalization_thread = threading.Thread(
                target=finalization_loop,
                args=(tracker, stop_event),
                daemon=True
            )
            finalization_thread.start()
            logger.info("Track finalization thread started")

        console.print("[green]✓ DOT receiver started[/green]")
        console.print(f"[dim]Endpoints available at http://{host}:{port}[/dim]")

        flask_app.run(host=host, port=port, threaded=True, debug=debug)

    except Exception as e:
        console.print(f"[red]Error starting receiver: {e}[/red]")
        raise typer.Exit(1)
