"""P-01: a video whose stem doesn't parse as YYYYMMDD_HHMMSS used to raise and
get logged as a failure. Now it silently ships video_timestamp: null, which
contradicts the .detection.json comment saying the backend parses that field
as ISO-8601. A malformed stem with confirmed detections must at least log a
warning so it's visible instead of shipping quietly."""
import json
from types import SimpleNamespace
from unittest.mock import patch


def _pipeline(tmp_path):
    from bugcam.edge26 import pipeline as edge26_main

    with patch.object(edge26_main, "VideoProcessor"):
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
                    "enable_classification": False,
                },
                "output": {"results_dir": str(tmp_path / "output")},
            }
        )
    pipeline.processor.output_config = {"save_composites": False}
    return pipeline


def _install_fake_detection_result(pipeline, tmp_path, *, track_id: str) -> None:
    """Wire processor._pipeline.process_video() to return one confirmed track
    whose crops dir already exists on disk, so it survives the
    queueable_tracks filter and confirmed_count > 0."""
    fake_track = SimpleNamespace(crops=[object(), object()])
    fake_result = SimpleNamespace(
        confirmed_tracks={track_id: fake_track},
        track_paths={track_id: fake_track},
        all_detections=[],
        video_info={},
    )
    pipeline.processor._pipeline.process_video.return_value = fake_result


def test_malformed_video_stem_logs_warning_and_ships_null_timestamp(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    track_id = "abcd1234-track_20260101_000000"
    _install_fake_detection_result(pipeline, tmp_path, track_id=track_id)

    date_time = "not-a-timestamp"
    video_path = tmp_path / "input" / f"flick01_{date_time}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake video")

    output_dir = pipeline.results_dir / "flick01" / date_time
    (output_dir / "crops" / "abcd1234").mkdir(parents=True, exist_ok=True)

    with caplog.at_level("WARNING"):
        pipeline._process_video_detection(video_path)

    assert any(
        "not-a-timestamp" in record.message and "video_timestamp" in record.message
        for record in caplog.records
    ), "expected a warning naming the unparseable stem"

    meta = json.loads((output_dir / ".detection.json").read_text())
    assert meta["video_timestamp"] is None


def test_well_formed_video_stem_does_not_warn(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    track_id = "abcd1234-track_20260101_000000"
    _install_fake_detection_result(pipeline, tmp_path, track_id=track_id)

    date_time = "20260101_120000_000000"
    video_path = tmp_path / "input" / f"flick01_{date_time}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake video")

    output_dir = pipeline.results_dir / "flick01" / date_time
    (output_dir / "crops" / "abcd1234").mkdir(parents=True, exist_ok=True)

    with caplog.at_level("WARNING"):
        pipeline._process_video_detection(video_path)

    assert not any("video_timestamp" in record.message for record in caplog.records)
    meta = json.loads((output_dir / ".detection.json").read_text())
    assert meta["video_timestamp"] is not None
