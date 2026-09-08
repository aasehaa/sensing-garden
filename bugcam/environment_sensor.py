"""Helpers for collecting one-shot SEN55 environmental readings."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bugcam.settings import get_state_dir


SEN55_BINARY_NAME = "sen55_reader"
SEN55_TIMEOUT_SECONDS = 15


def get_sen55_binary_path() -> Path:
    """Return the installed SEN55 reader binary path."""
    return get_state_dir() / "bin" / SEN55_BINARY_NAME


def _require_sen55_binary() -> Path:
    binary_path = get_sen55_binary_path()
    if not binary_path.exists():
        raise FileNotFoundError("SEN55 binary not found. Run `bugcam setup` to compile it.")
    return binary_path


def _parse_binary_output(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("SEN55 reader did not emit valid JSON")


def read_environment_sensor(timeout_seconds: int = SEN55_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run the compiled SEN55 binary once and return the parsed JSON payload."""
    binary_path = _require_sen55_binary()
    result = subprocess.run(
        [str(binary_path), "--oneshot"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"SEN55 reader failed: {stderr}")
    return _parse_binary_output(result.stdout)


def build_environment_payload(flick_id: str, reading: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw SEN55 reading into the backend environmental schema."""
    timestamp = str(reading.get("timestamp") or datetime.now(timezone.utc).isoformat())
    return {
        "device_id": flick_id,
        "timestamp": timestamp,
        "pm1p0": reading.get("pm1p0"),
        "pm2p5": reading.get("pm2p5"),
        "pm4p0": reading.get("pm4p0"),
        "pm10p0": reading.get("pm10p0"),
        "voc_index": reading.get("voc_index"),
        "nox_index": reading.get("nox_index"),
        "temperature": reading.get("temperature"),
        "humidity": reading.get("humidity"),
    }


def collect_environment_reading(
    output_dir: Path,
    flick_id: str,
    timeout_seconds: int = SEN55_TIMEOUT_SECONDS,
) -> tuple[Path, dict[str, Any]]:
    """Collect one environmental reading and write it into the output tree."""
    reading = read_environment_sensor(timeout_seconds=timeout_seconds)
    payload = build_environment_payload(flick_id, reading)
    timestamp = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")).astimezone(timezone.utc)
    environment_dir = output_dir / flick_id / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    output_path = environment_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, payload
