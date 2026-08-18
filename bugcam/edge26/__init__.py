"""Vendored edge26 pipeline package."""

from bugcam.edge26.recorder import VideoRecorder
from bugcam.edge26.main import Pipeline, setup_logging
from bugcam.edge26.output import ResultsWriter
from bugcam.edge26.processing import VideoProcessor

__all__ = [
    "Pipeline",
    "ResultsWriter",
    "VideoProcessor",
    "VideoRecorder",
    "setup_logging",
]
