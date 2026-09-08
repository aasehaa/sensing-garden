"""Tests for Pipeline._process_dot_directory_detection (detection-side DOT
handling: copying ready track crops/labels, queuing for classification).

DOT upload/produce-site behavior is covered separately by test_dot_per_track.py.
"""
from pathlib import Path
from unittest.mock import patch


def test_dot_done_signal_processing_uses_existing_track_crops(tmp_path: Path) -> None:
    from bugcam.edge26 import pipeline as edge26_main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    pending_dir = tmp_path / "pending"
    track_dir = input_dir / "dot01_20260417" / "crops" / "track-a_123456"
    labels_dir = input_dir / "dot01_20260417" / "labels"
    videos_dir = input_dir / "dot01_20260417" / "videos"
    track_dir.mkdir(parents=True)
    labels_dir.mkdir()
    videos_dir.mkdir()
    (track_dir / "done.txt").write_text("", encoding="utf-8")
    (track_dir / "frame_000001.jpg").write_bytes(b"crop")
    (labels_dir / "track-a.json").write_text("{}", encoding="utf-8")
    (videos_dir / "dot01_20260417_123456.mp4").write_bytes(b"video")

    with patch.object(edge26_main, "VideoProcessor"):
        pipeline = edge26_main.Pipeline(
            {
                "device": {"flick_id": "flick01", "dot_ids": ["dot01"]},
                "paths": {"input_storage": str(input_dir), "pending_dir": str(pending_dir)},
                "pipeline": {"enable_recording": False, "enable_processing": True, "enable_classification": True},
                "output": {"results_dir": str(output_dir)},
            }
        )
        pipeline._process_dot_directory_detection(input_dir / "dot01_20260417")

    # Verify track was queued for classification
    assert pipeline.classification_queue.count() == 1
    entries = pipeline.classification_queue.get_pending_entries()
    assert len(entries) == 1
    filepath, entry = entries[0]
    assert entry.entry_type == "dot"
    assert entry.source_device == "dot01"
    assert entry.track_id == "track-a"
    assert entry.time == "123456"
    assert entry.num_crops == 1
    # Per-track terminal dir: <dot>/<YYYYMMDD>/<track_id>_<HHMMSS>/
    track_out = output_dir / "dot01" / "20260417" / "track-a_123456"
    assert Path(entry.output_dir) == track_out
    # Track should be deleted from input after queuing
    assert not track_dir.exists()
    # Crops should be copied into the per-track dir
    assert (track_out / "crops" / "track-a_123456" / "frame_000001.jpg").exists()
    # Labels should be copied into the per-track dir
    assert (track_out / "labels" / "track-a.json").exists()
    # Expected-tracks marker is 1 (one track per dir)
    expected_path = track_out / ".expected_tracks"
    assert expected_path.exists()
    assert expected_path.read_text().strip() == "1"
