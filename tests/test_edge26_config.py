"""Tests for edge26/config.py: build_edge26_config and build_bundle_provenance."""
import hashlib
import pytest
from pathlib import Path

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
