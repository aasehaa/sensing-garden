"""Pins the consolidation of FLIK and DOT crop classification onto one shared
VideoProcessor method (classify_dot_track, renamed classify_track_crops --
nothing in it is actually DOT-specific). Before this change,
Pipeline._classify_flik_track re-implemented the same crop-load/classify/
aggregate loop inline and silently skipped unreadable crop files instead of
warning like the DOT path does."""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from bugcam.edge26.classification import HierarchicalClassification
from bugcam.edge26.queue import QueueEntry


class _FakeClassifier:
    """Stands in for HailoClassifier -- no Hailo runtime needed in tests."""

    def classify(self, crop: np.ndarray) -> HierarchicalClassification:
        return HierarchicalClassification(
            family="Insecta", genus="Testus", species="fakus",
            family_confidence=0.9, genus_confidence=0.8, species_confidence=0.7,
        )

    def hierarchical_aggregate(self, classifications):
        if not classifications:
            return None
        return {
            "family": "Insecta", "genus": "Testus", "species": "fakus",
            "family_confidence": 0.9, "genus_confidence": 0.8, "species_confidence": 0.7,
        }


def _make_processor():
    with patch("bugcam.edge26.detection.DetectionPipeline"):
        from bugcam.edge26.detection import VideoProcessor
        processor = VideoProcessor({"detection": {}, "tracking": {}, "classification": {}})
    processor._classifier = _FakeClassifier()
    return processor


def _write_readable_crop(path: Path) -> None:
    cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))


def _write_unreadable_crop(path: Path) -> None:
    path.write_bytes(b"not a real image")


# --- VideoProcessor.classify_track_crops (the shared method) ---

def test_classify_track_crops_returns_none_when_no_crops(tmp_path: Path) -> None:
    processor = _make_processor()
    track_dir = tmp_path / "empty"
    track_dir.mkdir()

    assert processor.classify_track_crops(track_dir, "track1") is None


def test_classify_track_crops_classifies_readable_crops(tmp_path: Path) -> None:
    processor = _make_processor()
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    _write_readable_crop(track_dir / "frame_000001.jpg")

    result = processor.classify_track_crops(track_dir, "track1", timestamp="120000")

    assert result["track_id"] == "track1"
    assert result["timestamp"] == "120000"
    assert result["num_detections"] == 1
    assert result["final_prediction"]["species"] == "fakus"


def test_classify_track_crops_warns_and_skips_unreadable_crop(tmp_path: Path, caplog) -> None:
    processor = _make_processor()
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    _write_readable_crop(track_dir / "frame_000001.jpg")
    _write_unreadable_crop(track_dir / "frame_000002.jpg")

    with caplog.at_level("WARNING"):
        result = processor.classify_track_crops(track_dir, "track1")

    assert result["num_detections"] == 1, "only the readable crop should be counted"
    assert any(
        "Could not read crop" in r.message and "frame_000002" in r.message
        for r in caplog.records
    )


def test_classify_track_crops_returns_none_when_all_crops_unreadable(tmp_path: Path) -> None:
    processor = _make_processor()
    track_dir = tmp_path / "track"
    track_dir.mkdir()
    _write_unreadable_crop(track_dir / "frame_000001.jpg")

    assert processor.classify_track_crops(track_dir, "track1") is None


# --- Pipeline._classify_flik_track, now built on the shared method ---

def _flik_pipeline(tmp_path: Path):
    from bugcam.edge26 import pipeline as edge26_main

    with patch("bugcam.edge26.detection.DetectionPipeline"):
        pipeline = edge26_main.Pipeline(
            {
                "device": {"flick_id": "flick01", "dot_ids": []},
                "paths": {
                    "input_storage": str(tmp_path / "input"),
                    "pending_dir": str(tmp_path / "pending"),
                },
                "pipeline": {
                    "enable_recording": False,
                    "enable_processing": True,
                    "enable_classification": False,  # skip eager real HailoClassifier
                },
                "output": {"results_dir": str(tmp_path / "output")},
            }
        )
    pipeline.processor._classifier = _FakeClassifier()
    return pipeline


def test_classify_flik_track_warns_on_unreadable_crop_like_dot_path(tmp_path: Path, caplog) -> None:
    pipeline = _flik_pipeline(tmp_path)

    track_dir = tmp_path / "input" / "crops" / "abc12345"
    track_dir.mkdir(parents=True)
    _write_readable_crop(track_dir / "frame_000001.jpg")
    _write_unreadable_crop(track_dir / "frame_000002.jpg")

    output_dir = tmp_path / "output" / "flick01" / "20260101_120000"
    output_dir.mkdir(parents=True)
    (output_dir / ".expected_tracks").write_text("1")

    entry = QueueEntry(
        entry_type="flik",
        source_device="flick01",
        date="20260101",
        time="120000",
        track_id="abc12345-20260101_120000",
        track_dir=str(track_dir),
        output_dir=str(output_dir),
        num_crops=2,
    )

    with caplog.at_level("WARNING"):
        pipeline._classify_flik_track(entry)

    assert any("Could not read crop" in r.message for r in caplog.records), (
        "FLIK classification should warn on an unreadable crop, same as the DOT path"
    )

    results = json.loads((output_dir / "results.json").read_text())
    assert results["tracks"][0]["num_detections"] == 1
    assert results["tracks"][0]["final_prediction"]["species"] == "fakus"
    assert (output_dir / ".done").exists()
