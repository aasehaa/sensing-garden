"""Shared configuration utilities for bugcam."""
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


DEFAULT_API_URL = "https://api.sensinggarden.com/v1"
DEFAULT_S3_BUCKET = "scl-sensing-garden"


def get_config_path() -> Path:
    """Return the persistent BugCam config file path."""
    return Path.home() / ".config" / "bugcam" / "config.json"


def load_config() -> dict[str, Any]:
    """Load persistent BugCam config."""
    path = get_config_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("BugCam config must be a JSON object")
    return data


def save_config(config: dict[str, Any]) -> None:
    """Persist BugCam config."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def parse_dot_ids(value: str | list[str] | None) -> list[str]:
    """Normalize dot IDs from CLI or config values."""
    if value is None:
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def get_hailo_venv_dir() -> Path:
    """Get the directory for the Hailo venv."""
    return Path.home() / ".local" / "share" / "bugcam" / "hailo-venv"


def get_python_for_detection() -> str:
    """Get the Python interpreter to use for detection script.

    Prefers hailo venv if available, otherwise system Python.
    """
    # Check for hailo venv
    hailo_venv_python = get_hailo_venv_dir() / "bin" / "python"
    if hailo_venv_python.is_file():
        return str(hailo_venv_python)

    # Fall back to system Python on Linux
    if platform.system() == "Linux":
        if Path("/usr/bin/python3").is_file():
            return "/usr/bin/python3"
        if Path("/usr/local/bin/python3").is_file():
            return "/usr/local/bin/python3"
    return sys.executable


def get_cache_dir() -> Path:
    """Get the cache directory for bugcam, respecting XDG_CACHE_HOME."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return _expand(xdg_cache) / "bugcam"
    return Path.home() / ".cache" / "bugcam"


def _expand(value: str) -> Path:
    """Expand a leading '~' in a path from an env var or config value."""
    return Path(value).expanduser()


def _validate_state_dir(path: Path) -> None:
    """Warn if state_dir appears to be on external drive."""
    path_str = str(path)
    if "/media/" in path_str or "/mnt/" in path_str:
        console.print("[yellow]Warning: State directory is on external drive. This may cause permission issues.[/yellow]")
        console.print("[yellow]Consider using local filesystem: ~/.local/share/bugcam[/yellow]")


def get_state_dir() -> Path:
    """Get the state directory for bugcam, respecting config and environment."""
    state_dir = os.environ.get("BUGCAM_STATE_DIR")
    if state_dir:
        path = _expand(state_dir)
        _validate_state_dir(path)
        return path

    config = load_config()
    if config.get("state_dir"):
        path = _expand(str(config["state_dir"]))
        _validate_state_dir(path)
        return path

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return _expand(xdg_data) / "bugcam"
    return Path.home() / ".local" / "share" / "bugcam"


def get_input_storage_dir() -> Path:
    """Get the edge26 input storage directory."""
    input_dir = os.environ.get("BUGCAM_INPUT_DIR")
    if input_dir:
        return _expand(input_dir)

    config = load_config()
    if config.get("input_dir"):
        return _expand(str(config["input_dir"]))

    return get_state_dir() / "incoming"


def get_pending_dir() -> Path:
    """Get the pending classification queue directory."""
    pending_dir = os.environ.get("BUGCAM_PENDING_DIR")
    if pending_dir:
        return _expand(pending_dir)

    config = load_config()
    if config.get("pending_dir"):
        return _expand(str(config["pending_dir"]))

    return get_state_dir() / "pending"


def get_output_storage_dir() -> Path:
    """Get the edge26 output storage directory."""
    output_dir = os.environ.get("BUGCAM_OUTPUT_DIR")
    if output_dir:
        return _expand(output_dir)

    config = load_config()
    if config.get("output_dir"):
        return _expand(str(config["output_dir"]))

    return get_state_dir() / "outputs"


def get_default_flick_id() -> str:
    """Get the configured FLICK device identifier."""
    return os.environ.get("BUGCAM_FLICK_ID", "bugcam-rpi")


def get_default_dot_ids() -> list[str]:
    """Get configured DOT device identifiers."""
    raw_value = os.environ.get("BUGCAM_DOT_IDS", "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def get_edge26_model_path() -> Path:
    """Resolve the HEF model path for edge26-backed processing."""
    from .model_bundles import get_models_cache_dir, resolve_model_path

    model_ref = os.environ.get("BUGCAM_EDGE26_MODEL")
    resolved = resolve_model_path(model_ref)
    if resolved:
        return resolved

    return get_models_cache_dir() / "edge26" / "model.hef"


def get_edge26_labels_path(model_path: Path | None = None) -> Path:
    """Resolve the labels file for edge26-backed processing."""
    from .model_bundles import get_models_cache_dir, resolve_labels_path

    labels_path = os.environ.get("BUGCAM_EDGE26_LABELS")
    if labels_path:
        return _expand(labels_path)

    model_ref = os.environ.get("BUGCAM_EDGE26_MODEL")
    resolved = resolve_labels_path(model_ref)
    if resolved:
        return resolved

    if model_path is not None and model_path.parent.name:
        candidate = model_path.parent / "labels.txt"
        if candidate.exists():
            return candidate
        return candidate

    return get_models_cache_dir() / "edge26" / "labels.txt"


def get_edge26_taxonomy_cache_path() -> Path:
    """Get the taxonomy cache file path."""
    cache_path = os.environ.get("BUGCAM_EDGE26_TAXONOMY_CACHE")
    if cache_path:
        return _expand(cache_path)
    return get_state_dir() / "taxonomy-cache.json"


def is_edge26_classification_enabled() -> bool:
    """Return whether edge26 classification is enabled."""
    value = os.environ.get("BUGCAM_EDGE26_CLASSIFICATION", "0").lower()
    return value not in {"0", "false", "no"}


def is_edge26_continuous_tracking_enabled() -> bool:
    """Return whether edge26 continuous tracking is enabled."""
    value = os.environ.get("BUGCAM_EDGE26_CONTINUOUS_TRACKING", "0").lower()
    return value not in {"0", "false", "no"}
