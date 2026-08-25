"""BugCam edge26 configuration bridge."""
from __future__ import annotations

import os
import yaml
from importlib import resources
from pathlib import Path
from typing import Any

from .config import (
    get_edge26_taxonomy_cache_path,
)
from .model_bundles import sha256_file

DEFAULT_CAPTURE_RESOLUTION = (1080, 1080)
MAX_CAPTURE_WIDTH = 3840
MAX_CAPTURE_HEIGHT = 2160

EDGE26_DETECTION_DEFAULTS = {
    "gmm_history": 500,
    "gmm_var_threshold": 16,
    "morph_kernel_size": 5,
    "min_area": 0.00012,
    "max_area": 0.0015,
    "min_density": 3.0,
    "min_solidity": 0.55,
    "min_largest_blob_ratio": 0.85,
    "max_num_blobs": 3,
    "min_motion_ratio": 0.20,
    "max_frame_jump": 0.06,
    "max_area_change_ratio": 3.0,
    "min_path_points": 10,
    "min_displacement": 0.25,
    "max_revisit_ratio": 0.30,
    "min_progression_ratio": 0.70,
    "max_directional_variance": 0.85,
    "revisit_radius": 0.025,
    # Explicit (width, height) in pixels to run the detector at. None = detect
    # at native resolution. See bugspot for details — bounding boxes are scaled
    # back to native res so crops/composites stay full resolution.
    "detection_resolution": [1920, 1080],
    # Resolution morph_kernel_size/min_density were tuned for. None = treat the
    # native frame size as the reference. Set to the tuned resolution (e.g.
    # [3840, 2160] for 4K) to auto-scale those params to the detection
    # resolution. See bugspot ScaledDetector for details.
    "reference_resolution": [3840, 2160],
}

EDGE26_TRACKING_DEFAULTS = {
    "w_dist": 0.6,
    "w_area": 0.4,
    "cost_threshold": 0.25,
    "max_lost_frames": 45,
}

DETECTION_KEYS = set(EDGE26_DETECTION_DEFAULTS.keys())
TRACKING_KEYS = set(EDGE26_TRACKING_DEFAULTS.keys())

YAML_TO_CONFIG_KEYS = {
    "tracker_w_dist": "w_dist",
    "tracker_w_area": "w_area",
    "tracker_cost_threshold": "cost_threshold",
}


def get_bundled_detection_config_path() -> Path | None:
    """Get the path to the bundled detection.yaml in the package."""
    try:
        import bugcam
        return Path(resources.files(bugcam) / "detection.yaml")
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def get_detection_config_path(custom_path: Path | None = None) -> Path | None:
    """Get the detection config file path.

    Priority:
    1. Custom path provided via parameter (e.g., --detection-config CLI flag)
    2. Bundled default: bugcam/detection.yaml in the package

    Returns None if no config found (will use hardcoded defaults).
    """
    if custom_path:
        return custom_path

    if os.environ.get("BUGCAM_SKIP_DETECTION_CONFIG"):
        return None

    return get_bundled_detection_config_path()


def load_detection_config(
    config_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load detection and tracking config from a YAML file.

    Args:
        config_path: Path to the YAML config file. If None, checks default location.

    Returns:
        Tuple of (detection_dict, tracking_dict) if config loaded, None otherwise.

    Raises:
        FileNotFoundError: If specified config file doesn't exist.
        ValueError: If YAML contains unknown keys.
    """
    if config_path is None:
        config_path = get_detection_config_path()

    if config_path is None or not config_path.exists():
        return None

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not data:
        return None

    detection = {}
    tracking = {}

    for yaml_key, value in data.items():
        if yaml_key in YAML_TO_CONFIG_KEYS:
            config_key = YAML_TO_CONFIG_KEYS[yaml_key]
            if config_key in TRACKING_KEYS:
                tracking[config_key] = value
            else:
                detection[config_key] = value
        elif yaml_key in DETECTION_KEYS:
            detection[yaml_key] = value
        elif yaml_key in TRACKING_KEYS:
            tracking[yaml_key] = value
        else:
            valid_keys = sorted(DETECTION_KEYS | TRACKING_KEYS | set(YAML_TO_CONFIG_KEYS.keys()))
            raise ValueError(
                f"Unknown key '{yaml_key}' in detection config. Valid keys: {valid_keys}"
            )

    return detection, tracking


def parse_capture_resolution(value: str) -> tuple[int, int]:
    """Parse a capture resolution in WxH format."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("resolution must be in WxH format")

    width = int(parts[0])
    height = int(parts[1])
    if width < 1 or height < 1:
        raise ValueError("resolution values must be positive")
    if width > MAX_CAPTURE_WIDTH or height > MAX_CAPTURE_HEIGHT:
        raise ValueError(f"resolution cannot exceed {MAX_CAPTURE_WIDTH}x{MAX_CAPTURE_HEIGHT}")
    return width, height


def build_edge26_config(
    flick_id: str,
    dot_ids: list[str],
    input_dir: str,
    output_dir: str,
    model_path: str,
    labels_path: str,
    recording_mode: str = "continuous",
    recording_interval: int = 5,
    chunk_duration: int = 60,
    fps: int = 30,
    resolution: tuple[int, int] = DEFAULT_CAPTURE_RESOLUTION,
    bitrate: int = 20_000_000,
    enable_recording: bool = True,
    enable_processing: bool = True,
    enable_classification: bool = True,
    continuous_tracking: bool = False,
    detection_in_subprocess: bool = True,
    watch_input: bool = False,
    model_metadata: dict[str, Any] | None = None,
    detection_config_path: Path | None = None,
    timezone_name: str | None = None,
    record_window: str | None = None,
    video_sample_interval: int = 10,
    random_sampling: bool = False,
) -> dict[str, Any]:
    """Build the edge26 pipeline config from BugCam-owned settings."""
    results_dir = Path(output_dir)

    config_path = get_detection_config_path(detection_config_path)
    loaded_config = load_detection_config(config_path)
    if loaded_config:
        detection_overrides, tracking_overrides = loaded_config
        detection_config = detection_overrides
        tracking_config = tracking_overrides
        detection_config_source = str(config_path)
    else:
        detection_config = dict(EDGE26_DETECTION_DEFAULTS)
        tracking_config = dict(EDGE26_TRACKING_DEFAULTS)
        detection_config_source = (
            f"hardcoded defaults (empty or unreadable: {config_path})"
            if config_path is not None
            else "hardcoded defaults (no detection.yaml found)"
        )

    return {
        "detection_config_source": detection_config_source,
        "device": {
            "flick_id": flick_id,
            "dot_ids": dot_ids,
            "timezone": timezone_name,
        },
        "paths": {
            "input_storage": input_dir,
            "logs_dir": str(results_dir / flick_id / "logs"),
        },
        "pipeline": {
            "enable_recording": enable_recording,
            "enable_processing": enable_processing,
            "enable_classification": enable_classification,
            "continuous_tracking": continuous_tracking,
            "detection_in_subprocess": detection_in_subprocess,
            "watch_input": watch_input,
            "recording_mode": recording_mode,
            "recording_interval_minutes": recording_interval,
            "record_window": record_window,
            "video_sample_interval": video_sample_interval,
            "random_sampling": random_sampling,
        },
        "capture": {
            "camera_index": 0,
            "use_picamera": True,
            "fps": fps,
            "chunk_duration_seconds": chunk_duration,
            "resolution": list(resolution),
            "bitrate": bitrate,
        },
        "detection": detection_config,
        "tracking": tracking_config,
        "classification": {
            "model": model_path,
            "labels": labels_path,
            "taxonomy_cache": str(get_edge26_taxonomy_cache_path()),
            "normalize": False,
        },
        "model": dict(model_metadata or {}),
        "output": {
            "results_dir": output_dir,
            "save_crops": True,
            "save_composites": True,
        },
    }


def build_bundle_provenance(model_path: Path, labels_path: Path) -> dict[str, Any]:
    """Build technical provenance for the active bundle."""
    provenance = {
        "model_id": model_path.parent.name,
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "model_sha256": sha256_file(model_path),
        "labels_sha256": sha256_file(labels_path),
    }
    return provenance
