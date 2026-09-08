"""Pins the swap points added for DetectionBackend/ClassifierBackend
(interfaces.py): VideoProcessor must drive an injected backend/classifier
through backend_factory/classifier_cls, not the bugspot/Hailo defaults, and
callers must go through process_video()/ensure_classifier() rather than
reaching into _pipeline/_classifier directly."""
from pathlib import Path

import numpy as np


class _FakeBackend:
    """Records calls instead of running bugspot; stands in for a
    researcher's own detection/tracking backend."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.reset_calls = 0
        self.clear_calls = 0
        self.process_video_calls: list[tuple] = []

    def process_video(self, video_path, *, extract_crops, render_composites,
                       save_crops_dir, save_composites_dir):
        self.process_video_calls.append(
            (video_path, extract_crops, render_composites, save_crops_dir, save_composites_dir)
        )
        return "fake-result"

    def reset(self) -> None:
        self.reset_calls += 1

    def clear(self) -> None:
        self.clear_calls += 1


class _FakeClassifier:
    """Stands in for HailoClassifier; records how many times it was built."""

    instances_built = 0

    def __init__(self, config: dict) -> None:
        self.config = config
        _FakeClassifier.instances_built += 1

    def classify(self, crop: np.ndarray):
        raise NotImplementedError

    def hierarchical_aggregate(self, classifications):
        raise NotImplementedError


def _processor(**kwargs):
    from bugcam.edge26.detection import VideoProcessor

    return VideoProcessor(
        {"detection": {}, "tracking": {}, "classification": {}}, **kwargs
    )


def test_process_video_delegates_to_injected_backend():
    processor = _processor(backend_factory=_FakeBackend)

    result = processor.process_video(
        "video.mp4",
        extract_crops=True,
        render_composites=False,
        save_crops_dir="crops",
        save_composites_dir=None,
    )

    assert result == "fake-result"
    assert processor._pipeline.process_video_calls == [
        ("video.mp4", True, False, "crops", None)
    ]


def test_reset_and_clear_reach_injected_backend():
    processor = _processor(backend_factory=_FakeBackend)

    processor.reset_tracker()
    processor.clear_video_detections()  # continuous_tracking defaults False -> reset()

    assert processor._pipeline.reset_calls == 2
    assert processor._pipeline.clear_calls == 0


def test_ensure_classifier_uses_injected_class_and_builds_once():
    _FakeClassifier.instances_built = 0
    processor = _processor(classifier_cls=_FakeClassifier)

    first = processor.ensure_classifier()
    second = processor.ensure_classifier()

    assert isinstance(first, _FakeClassifier)
    assert first is second
    assert _FakeClassifier.instances_built == 1


def test_default_backend_and_classifier_are_bugspot_and_hailo():
    from bugcam.edge26.classification import HailoClassifier
    from unittest.mock import patch

    with patch("bugcam.edge26.detection.DetectionPipeline") as pipeline_cls:
        processor = _processor()

    pipeline_cls.assert_called_once()
    assert processor._pipeline is pipeline_cls.return_value
    assert processor._classifier_cls is HailoClassifier
