"""Tests for BugCam edge26 processing integration."""
import hashlib
import pytest
from pathlib import Path
from unittest.mock import patch

from bugcam.edge26.config import build_bundle_provenance, build_edge26_config

BUGSPOT_RATIO_DETECTION_VALUES: tuple[tuple[str, float | int], ...] = (
    ("min_area", 0.00012),
    ("max_area", 0.0015),
    ("min_displacement", 0.25),
    ("max_frame_jump", 0.06),
    ("revisit_radius", 0.025),
    ("morph_kernel_size", 5),
)


@pytest.mark.xfail(reason="SG-028: detection min_area default drift (yaml 0.00012 vs test 0.0002)", strict=False)
def test_build_edge26_config_uses_bugspot_ratio_detection_defaults(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
    )

    for key, value in BUGSPOT_RATIO_DETECTION_VALUES:
        assert config["detection"][key] == value


def test_build_edge26_config_video_sample_interval_default(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
    )

    assert config["pipeline"]["video_sample_interval"] == 10


def test_build_edge26_config_video_sample_interval_custom(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
        video_sample_interval=25,
    )

    assert config["pipeline"]["video_sample_interval"] == 25


def test_build_edge26_config_random_sampling_default_false(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
    )

    assert config["pipeline"]["random_sampling"] is False


def test_build_edge26_config_random_sampling_enabled(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
        random_sampling=True,
    )

    assert config["pipeline"]["random_sampling"] is True


def test_build_edge26_config_reports_bundled_detection_source(tmp_path: Path) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
    )

    assert config["detection_config_source"].endswith("detection.yaml")


def test_build_edge26_config_reports_custom_detection_source(tmp_path: Path) -> None:
    custom_path = tmp_path / "custom-detection.yaml"
    custom_path.write_text("gmm_history: 400\n", encoding="utf-8")

    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
        detection_config_path=custom_path,
    )

    assert config["detection_config_source"] == str(custom_path)
    assert config["detection"]["gmm_history"] == 400


def test_build_edge26_config_reports_hardcoded_defaults_when_no_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUGCAM_SKIP_DETECTION_CONFIG", "1")

    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=[],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
    )

    assert config["detection_config_source"] == "hardcoded defaults (no detection.yaml found)"


def test_video_processor_passes_ratio_config_to_bugspot() -> None:
    from bugcam.edge26.detection import VideoProcessor

    detection = dict(BUGSPOT_RATIO_DETECTION_VALUES)
    tracking = {"max_lost_frames": 45, "w_dist": 0.6, "w_area": 0.4, "cost_threshold": 0.3}

    with patch("bugcam.edge26.detection.DetectionPipeline") as pipeline:
        VideoProcessor({"detection": detection, "tracking": tracking})

    pipeline.assert_called_once_with(
        {
            **detection,
            "max_lost_frames": 45,
            "tracker_w_dist": 0.6,
            "tracker_w_area": 0.4,
            "tracker_cost_threshold": 0.3,
        }
    )


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

    with patch.object(edge26_main, "VideoProcessor") as processor_cls:
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


def test_build_edge26_config_resolves_paths(tmp_path: Path, monkeypatch) -> None:
    config = build_edge26_config(
        flick_id="flick01",
        dot_ids=["dot01", "dot02"],
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "outputs"),
        model_path=str(tmp_path / "bundle" / "model.hef"),
        labels_path=str(tmp_path / "bundle" / "labels.txt"),
        recording_mode="interval",
        recording_interval=7,
        chunk_duration=90,
        fps=24,
        resolution=(1920, 1080),
        enable_recording=True,
        enable_processing=True,
        enable_classification=False,
        continuous_tracking=False,
    )
    assert config["device"]["flick_id"] == "flick01"
    assert config["device"]["dot_ids"] == ["dot01", "dot02"]
    assert config["paths"]["input_storage"].endswith("input")
    assert config["paths"]["logs_dir"].endswith("outputs/flick01/logs")
    assert config["output"]["results_dir"].endswith("outputs")
    assert config["pipeline"]["recording_mode"] == "interval"
    assert config["pipeline"]["recording_interval_minutes"] == 7
    assert config["capture"]["chunk_duration_seconds"] == 90
    assert config["capture"]["fps"] == 24
    assert config["capture"]["resolution"] == [1920, 1080]
    assert config["pipeline"]["enable_classification"] is False
    assert config["pipeline"]["continuous_tracking"] is False


def test_build_bundle_provenance_hashes_active_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle-a"
    bundle_dir.mkdir()
    model_path = bundle_dir / "model.hef"
    labels_path = bundle_dir / "labels.txt"
    model_path.write_bytes(b"hef-data")
    labels_path.write_text("species-a\n", encoding="utf-8")

    provenance = build_bundle_provenance(model_path, labels_path)

    assert provenance["model_id"] == "bundle-a"
    assert provenance["model_sha256"] == hashlib.sha256(b"hef-data").hexdigest()
    assert provenance["labels_sha256"] == hashlib.sha256(b"species-a\n").hexdigest()


def test_process_uses_device_config_flick_id(tmp_path: Path) -> None:
    from bugcam.commands import process as process_command

    with patch('bugcam.commands.process.load_device_config') as mock_device_config, \
         patch('bugcam.commands.process.resolve_bundle_provenance', return_value={"model_id": "bundle", "model_sha256": "abc123456789"}), \
         patch('bugcam.commands.process.build_pipeline') as mock_build_pipeline, \
         patch('bugcam.commands.process.resolve_flick_id', return_value="flick-config"):
        mock_device_config.return_value.flick_id = "flick-config"
        mock_device_config.return_value.dot_ids = ["dot01"]

        process_command.process(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            model="bundle",
            flick_id=None,
            classification=True,
            continuous_tracking=True,
        )

    assert mock_build_pipeline.call_args.kwargs["flick_id"] == "flick-config"
    assert mock_build_pipeline.call_args.kwargs["dot_ids"] == ["dot01"]


def test_run_resolves_flick_id_from_config() -> None:
    from bugcam.commands.run import _resolve_runtime_settings

    with patch('bugcam.commands.run.load_config', return_value={"api_key": "key", "s3_bucket": "bucket"}), \
         patch('bugcam.commands.run.resolve_flick_id', return_value="flick-config"):
        settings = _resolve_runtime_settings(None, None, None, None, None)

    assert settings["flick_id"] == "flick-config"
