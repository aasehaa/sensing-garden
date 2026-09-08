"""Structural interfaces for the two swap points in the edge26 pipeline:
what does detection/tracking need to return, and what does a classifier
need to expose. Pure typing -- no runtime behavior -- so a substitute
(bugspot's own classes, a research fork, a stub for tests) only needs to
match the shape, not inherit from anything here.

VideoProcessor (detection.py) accepts a backend_factory / classifier_cls
built against these Protocols instead of hardcoding bugspot.DetectionPipeline
and HailoClassifier; Pipeline (pipeline.py) drives them only through
VideoProcessor's own methods, never through '_pipeline'/'_classifier' directly.
"""
from __future__ import annotations

from typing import Protocol, Sized, runtime_checkable


@runtime_checkable
class Track(Protocol):
    """One confirmed track. `.crops` is the only hard requirement (used
    unguarded for the queued crop count); pipeline.py also reads
    `.num_detections`, `.first_frame_time`, `.last_frame_time`, `.duration`,
    and `.topology_metrics` if present (each individually `hasattr`-checked
    and written as null in detection metadata when absent), so a backend
    that omits them degrades gracefully rather than crashing."""

    crops: Sized


@runtime_checkable
class DetectionResult(Protocol):
    """Return value of DetectionBackend.process_video. Track identity is the
    `confirmed_tracks` dict key (a `{uuid}_{timestamp}` string, see
    `crop_dir_name` below) -- Track objects carry no id of their own."""

    confirmed_tracks: dict[str, Track]
    track_paths: Sized  # only len() is used, for the "N total tracks" log line


@runtime_checkable
class DetectionBackend(Protocol):
    """What VideoProcessor needs from a detection+tracking engine. bugspot's
    DetectionPipeline satisfies this today; a substitute (a different
    detector, a different tracker, a mock for tests) just needs to match it.

    Detection and tracking are not separately swappable at this boundary --
    bugspot bundles both behind one process_video() call, and this interface
    mirrors that. Splitting them further would mean depending on bugspot's
    own MotionDetector/InsectTracker split (see the classification/detection
    module docstrings) rather than bugcam's.
    """

    def process_video(
        self,
        video_path: str,
        *,
        extract_crops: bool,
        render_composites: bool,
        save_crops_dir: str,
        save_composites_dir: str | None,
    ) -> DetectionResult: ...

    def reset(self) -> None:
        """Full reset, including tracker state (device restart, day change)."""
        ...

    def clear(self) -> None:
        """Per-video clear; tracker state persists (continuous tracking)."""
        ...


def crop_dir_name(track_id: str) -> str:
    """Derive a track's crop-output subdirectory name from its id.

    Centralizes what was previously inlined at the pipeline.py call site:
    bugspot ids are `{uuid}_{timestamp}`, and only the first 8 hex chars of
    the uuid (its first '-'-delimited segment) are used as the directory
    name, matching where bugspot itself writes crops. A substitute backend
    using a different id scheme should either match this convention or the
    caller must be updated to match its Track.id format instead.
    """
    return track_id.split("-")[0]


@runtime_checkable
class ClassificationResult(Protocol):
    """Return value of ClassifierBackend.classify."""

    family: str
    genus: str
    species: str
    family_confidence: float
    genus_confidence: float
    species_confidence: float
    family_probs: list[float]
    genus_probs: list[float]
    species_probs: list[float]


@runtime_checkable
class ClassifierBackend(Protocol):
    """What VideoProcessor needs from a classifier. HailoClassifier satisfies
    this today; a substitute (ONNX, TensorRT, a CPU fallback, a mock for
    tests) just needs to match it -- construction (`__init__(config)`) is not
    part of the interface since it's backend-specific, only the two calls
    VideoProcessor actually makes."""

    def classify(self, crop) -> ClassificationResult: ...

    def hierarchical_aggregate(
        self, classifications: list[ClassificationResult]
    ) -> dict | None: ...


@runtime_checkable
class ModelRunner(Protocol):
    """What HierarchicalClassifier (classification.py) needs from an
    inference backend. This is one level narrower than ClassifierBackend
    above: preprocessing (crop resize/colorspace), output parsing/softmax,
    taxonomy resolution, and hierarchical aggregation are shared and
    hardware-agnostic, living on HierarchicalClassifier itself; only the
    actual model load + forward pass are backend-specific, and that's all
    this Protocol covers. HailoModelRunner satisfies it today; an
    ONNX/TensorRT/CPU backend only needs to implement this, not reimplement
    softmax or taxonomy lookup."""

    def load(self) -> None:
        """Load the model and prepare for inference. Idempotent."""
        ...

    def input_hw(self) -> tuple[int, int]:
        """Return the model's expected (height, width) input."""
        ...

    def output_head_sizes(self) -> list[int]:
        """Each output head's class count, in model output order. Used only
        to build numeric placeholder labels when no labels file is
        configured."""
        ...

    def run(self, preprocessed) -> list:
        """Run one forward pass; return raw (pre-softmax) per-head outputs,
        batch dimension already stripped."""
        ...
