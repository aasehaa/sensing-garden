"""Show DOT sensor setup details for operators."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from bugcam.settings import get_configured_flick_id, get_input_storage_dir, get_output_storage_dir, load_config, parse_dot_ids

app = typer.Typer(help="Show DOT sensor setup info", invoke_without_command=True, no_args_is_help=False)
console = Console()


@dataclass(frozen=True)
class DotInfoSettings:
    flick_id: str
    dot_ids: list[str]
    input_dir: Path
    output_dir: Path


def _load_dot_info_settings() -> DotInfoSettings:
    config = load_config()
    return DotInfoSettings(
        flick_id=get_configured_flick_id(),
        dot_ids=parse_dot_ids(config.get("dot_ids")),
        input_dir=get_input_storage_dir(),
        output_dir=get_output_storage_dir(),
    )


def _example_header(example_dot_id: str, today: date) -> str:
    return f"Folder structure (example for {example_dot_id}, today {today.isoformat()}):"


def _print_dot_ids(dot_ids: list[str]) -> None:
    console.print("Your DOT IDs:")
    for dot_id in dot_ids:
        console.print(f"  {dot_id}")


def _print_input_structure(example_dot_id: str, input_dir: Path, today_stamp: str) -> None:
    console.print("\nInput folder:")
    console.print(f"  {input_dir}")
    console.print(f"  {example_dot_id}_{today_stamp}/")
    console.print("  ├── crops/")
    console.print("  │   └── {track_id}_{HHMMSS}/")
    console.print("  │       ├── frame_000001.jpg")
    console.print("  │       ├── frame_000002.jpg")
    console.print("  │       └── done.txt              <- add this when track is complete")
    console.print("  ├── labels/")
    console.print("  │   └── {track_id}.json           <- bounding box data per frame")
    console.print("  ├── videos/")
    console.print(f"  │   └── {example_dot_id}_{today_stamp}" + "_{HHMMSS}.mp4")
    console.print("  └── {HHMMSS}_background.jpg       <- reference frame without insects")


def _print_output_structure(example_dot_id: str, output_dir: Path) -> None:
    console.print("\nOutput folder:")
    console.print(f"  {output_dir}")
    console.print(f"  {example_dot_id}/")
    console.print("  ├── heartbeats/")
    console.print("  │   └── {YYYYMMDD_HHMMSS}.json")
    console.print("  └── environment/")
    console.print("      └── {YYYYMMDD_HHMMSS}.json")


@app.callback()
def dot_info() -> None:
    """Print the DOT sensor setup instructions for operators."""
    settings = _load_dot_info_settings()
    console.print("DOT Sensor Setup\n")

    if not settings.dot_ids:
        console.print("No DOT sensors configured. Run bugcam setup with dot count > 0.")
        return
    if not settings.flick_id:
        raise typer.BadParameter("Missing required config value: flick_id. Run `bugcam setup`.")

    today = date.today()
    today_stamp = today.strftime("%Y%m%d")
    example_dot_id = settings.dot_ids[0]

    console.print(f"Device: {settings.flick_id}")
    console.print(f"Input folder: {settings.input_dir}")
    console.print(f"Output folder: {settings.output_dir}\n")
    _print_dot_ids(settings.dot_ids)
    console.print()
    console.print(_example_header(example_dot_id, today))
    console.print()
    _print_input_structure(example_dot_id, settings.input_dir, today_stamp)
    _print_output_structure(example_dot_id, settings.output_dir)
