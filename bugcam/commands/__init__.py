"""Shared helpers for bugcam CLI commands."""
from __future__ import annotations

import typer

from bugcam.edge26.config import parse_capture_resolution
from bugcam.settings import get_configured_flick_id, resolve_flick_id


def parse_resolution_option(value: str) -> tuple[int, int]:
    """Parse a --resolution CLI option, converting bad input to a Typer error."""
    try:
        return parse_capture_resolution(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def require_configured_flick_id(flick_id: str | None) -> str:
    """Resolve flick_id, refusing to silently fall back to the hardcoded
    default ("bugcam-rpi") -- run/process/record/environment must not
    operate as an unconfigured device on a fresh install. A CLI override,
    persisted config, or BUGCAM_FLICK_ID all count as "configured"; only
    the hardcoded literal fallback doesn't."""
    if not (flick_id or get_configured_flick_id()):
        raise typer.BadParameter(
            "No device id configured. Run `bugcam setup`, pass --flick-id, "
            "or set BUGCAM_FLICK_ID."
        )
    return resolve_flick_id(flick_id)
