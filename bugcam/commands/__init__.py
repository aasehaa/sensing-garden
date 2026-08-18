"""Shared helpers for bugcam CLI commands."""
from __future__ import annotations

import typer

from bugcam.edge26.config import parse_capture_resolution


def parse_resolution_option(value: str) -> tuple[int, int]:
    """Parse a --resolution CLI option, converting bad input to a Typer error."""
    try:
        return parse_capture_resolution(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
