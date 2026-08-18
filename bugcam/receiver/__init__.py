"""DOT Receiver module for bugcam.

This module provides an HTTP server for receiving insect track data from iOS devices.
"""

from flask import Flask
from pathlib import Path
import logging

from .config import RECEIVER_DEFAULT_PORT, RECEIVER_DEFAULT_HOST
from .tracker import PendingTrackTracker
from .routes import register_routes

logger = logging.getLogger(__name__)

_default_tracker = None


def create_app(config=None) -> Flask:
    """Create and configure the Flask application for DOT data reception.

    Args:
        config: Optional dictionary with configuration overrides.
               Supported keys: port, host, input_storage

    Returns:
        Configured Flask application.
    """
    global _default_tracker

    app = Flask(__name__)

    config = config or {}
    input_storage = Path(config.get("input_storage") or _get_input_storage())
    port = config.get("port") or RECEIVER_DEFAULT_PORT
    host = config.get("host") or RECEIVER_DEFAULT_HOST

    app.config["INPUT_STORAGE"] = input_storage
    app.config["PORT"] = port
    app.config["HOST"] = host

    input_storage.mkdir(parents=True, exist_ok=True)

    _default_tracker = PendingTrackTracker(input_storage)
    app.config["TRACKER"] = _default_tracker

    register_routes(app)

    logger.info(f"DOT Receiver initialized - storage: {input_storage}, port: {port}")
    return app


def _get_input_storage() -> Path:
    """Get the input storage directory from bugcam config."""
    from ..settings import get_input_storage_dir
    return get_input_storage_dir()


def get_tracker() -> PendingTrackTracker:
    """Get the default tracker instance."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = PendingTrackTracker(_get_input_storage())
    return _default_tracker
