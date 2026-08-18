"""Heartbeat snapshot command."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from bugcam.settings import get_input_storage_dir, get_output_storage_dir, load_config, parse_dot_ids

app = typer.Typer(help="Write a heartbeat snapshot", invoke_without_command=True, no_args_is_help=False)
console = Console()

_PROC_NET_DEV_PATH = Path("/proc/net/dev")


def _read_cpu_temperature_celsius() -> float:
    raw_value = Path("/sys/class/thermal/thermal_zone0/temp").read_text(encoding="utf-8").strip()
    return int(raw_value) / 1000.0


def _read_uptime_seconds() -> float:
    raw_value = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
    return float(raw_value)


def _read_network_usage() -> list[dict[str, object]]:
    """Cumulative rx/tx byte counters per network interface, excluding loopback.
    Returns [] if /proc/net/dev is unavailable -- this is an informational
    heartbeat field, not worth failing the heartbeat over."""
    try:
        lines = _PROC_NET_DEV_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    interfaces = []
    for line in lines[2:]:  # skip the two header lines
        if ":" not in line:
            continue
        name, _, rest = line.partition(":")
        name = name.strip()
        if name == "lo":
            continue
        fields = rest.split()
        if len(fields) < 9:
            continue
        interfaces.append({
            "interface": name,
            "rx_bytes": int(fields[0]),
            "tx_bytes": int(fields[8]),
        })
    return sorted(interfaces, key=lambda entry: entry["interface"])


def _build_dot_status(input_dir: Path, dot_ids: list[str]) -> list[dict[str, str | None]]:
    status = []
    for dot_id in dot_ids:
        dot_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith(f"{dot_id}_"))
        latest_dir = dot_dirs[-1] if dot_dirs else None
        status.append(
            {
                "dot_id": dot_id,
                "last_modified": datetime.fromtimestamp(latest_dir.stat().st_mtime, tz=timezone.utc).isoformat() if latest_dir else None,
            }
        )
    return status


def _build_incoming_status(input_dir: Path, flick_id: str, dot_ids: list[str]) -> dict[str, int]:
    """Backlog waiting in the incoming directory: unprocessed FLIK videos (the
    pipeline deletes them after detection, so presence means pending) and DOT
    tracks whose done.txt marks them ready for processing."""
    flick_videos = flick_video_bytes = dot_dirs = ready_dot_tracks = 0
    if input_dir.exists():
        for entry in input_dir.iterdir():
            if entry.is_file() and entry.suffix == ".mp4" and entry.name.startswith(f"{flick_id}_"):
                flick_videos += 1
                flick_video_bytes += entry.stat().st_size
            elif entry.is_dir() and any(entry.name.startswith(f"{dot_id}_") for dot_id in dot_ids):
                dot_dirs += 1
                crops_dir = entry / "crops"
                if crops_dir.exists():
                    ready_dot_tracks += sum(
                        1 for track in crops_dir.iterdir()
                        if track.is_dir() and (track / "done.txt").exists()
                    )
    return {
        "flick_videos": flick_videos,
        "flick_video_bytes": flick_video_bytes,
        "dot_dirs": dot_dirs,
        "ready_dot_tracks": ready_dot_tracks,
    }


def build_heartbeat_payload(
    flick_id: str,
    input_dir: Path,
    dot_ids: list[str],
    *,
    timestamp: datetime | None = None,
    timezone_name: str | None = None,
    pipeline_status: dict | None = None,
    upload_status: dict | None = None,
) -> dict[str, object]:
    """Build the heartbeat payload. Timestamps are UTC; timezone_name is the
    device's configured IANA zone, shipped as device info.

    ``pipeline_status`` / ``upload_status`` are supplied by ``bugcam run``
    (pipeline health snapshot, Pollen upload stats); the standalone heartbeat
    command has neither and omits those sections."""
    heartbeat_time = timestamp or datetime.now(timezone.utc)
    disk_usage = shutil.disk_usage(input_dir)
    payload: dict[str, object] = {
        "device_id": flick_id,
        "timestamp": heartbeat_time.isoformat(),
        "timezone": timezone_name,
        "cpu_temperature_celsius": _read_cpu_temperature_celsius(),
        "storage_free_bytes": disk_usage.free,
        "storage_total_bytes": disk_usage.total,
        "uptime_seconds": _read_uptime_seconds(),
        "network_interfaces": _read_network_usage(),
        "dot_status": _build_dot_status(input_dir, dot_ids),
        "incoming": _build_incoming_status(input_dir, flick_id, dot_ids),
    }
    if pipeline_status is not None:
        payload["pipeline"] = pipeline_status
    if upload_status is not None:
        payload["upload"] = upload_status
    return payload


def write_heartbeat_snapshot(
    output_dir: Path,
    flick_id: str,
    input_dir: Path,
    dot_ids: list[str],
    *,
    timezone_name: str | None = None,
    pipeline_status: dict | None = None,
    upload_status: dict | None = None,
) -> Path:
    """Write a heartbeat JSON document to the output directory."""
    heartbeat_time = datetime.now(timezone.utc)
    payload = build_heartbeat_payload(
        flick_id, input_dir, dot_ids,
        timestamp=heartbeat_time,
        timezone_name=timezone_name,
        pipeline_status=pipeline_status,
        upload_status=upload_status,
    )
    heartbeat_dir = output_dir / flick_id / "heartbeats"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = heartbeat_dir / f"{heartbeat_time.strftime('%Y%m%d_%H%M%S')}.json"
    heartbeat_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return heartbeat_path


def _resolve_runtime_settings(
    flick_id: str | None,
    dot_ids: str | None,
) -> dict[str, Any]:
    config = load_config()
    resolved_flick_id = flick_id or str(config.get("flick_id") or config.get("device_id") or "")
    resolved_dot_ids = parse_dot_ids(dot_ids) if dot_ids is not None else parse_dot_ids(config.get("dot_ids"))

    missing_fields = [
        field_name
        for field_name, value in (
            ("flick_id", resolved_flick_id),
        )
        if not value
    ]
    if missing_fields:
        joined = ", ".join(missing_fields)
        raise typer.BadParameter(f"Missing required config values: {joined}. Run `bugcam setup` or pass CLI flags.")

    return {
        "flick_id": resolved_flick_id,
        "dot_ids": resolved_dot_ids,
    }


@app.callback()
def heartbeat(
    flick_id: str | None = typer.Option(None, "--flick-id", help="FLICK device ID"),
    dot_ids: str | None = typer.Option(None, "--dot-ids", help="Comma-separated DOT IDs"),
    input_dir: Path = typer.Option(get_input_storage_dir(), "--input-dir", help="Directory containing DOT inputs"),
    output_dir: Path = typer.Option(get_output_storage_dir(), "--output-dir", help="Directory for processed output"),
) -> None:
    """Write a single heartbeat snapshot."""
    settings = _resolve_runtime_settings(flick_id, dot_ids)
    heartbeat_path = write_heartbeat_snapshot(
        output_dir=output_dir,
        flick_id=settings["flick_id"],
        input_dir=input_dir,
        dot_ids=settings["dot_ids"],
        timezone_name=str(load_config().get("timezone") or "") or None,
    )
    console.print(f"[green]Wrote[/green] {heartbeat_path}")
