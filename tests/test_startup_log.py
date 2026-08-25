"""Tests for the startup configuration log.

Spec: on `bugcam run` startup, the log must contain the fully-resolved
configuration -- detection.yaml (or hardcoded defaults) and device config --
plus the source commit when available, so a shipped log alone is enough to
reconstruct what a device was running. api_key is the only field redacted.
Gathering any piece must never raise.
"""
import logging
import subprocess
from pathlib import Path

from bugcam.config import get_source_commit, log_startup_config, redact_device_config


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "marker.py").write_text("# marker\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=path, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_get_source_commit_returns_none_outside_git_checkout(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "not_a_repo" / "bugcam"
    package_dir.mkdir(parents=True)
    fake_file = package_dir / "__init__.py"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("bugcam.__file__", str(fake_file))

    assert get_source_commit() is None


def test_get_source_commit_returns_short_sha_in_git_checkout(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "checkout"
    package_dir = repo_dir / "bugcam"
    package_dir.mkdir(parents=True)
    expected_sha = _init_git_repo(repo_dir)
    fake_file = package_dir / "__init__.py"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("bugcam.__file__", str(fake_file))

    assert get_source_commit() == expected_sha


def test_get_source_commit_survives_missing_git_binary(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "checkout"
    package_dir = repo_dir / "bugcam"
    package_dir.mkdir(parents=True)
    _init_git_repo(repo_dir)
    fake_file = package_dir / "__init__.py"
    fake_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("bugcam.__file__", str(fake_file))

    def _raise_missing_binary(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise_missing_binary)

    assert get_source_commit() is None


def test_redact_device_config_masks_api_key() -> None:
    device_config = {
        "flick_id": "flick01",
        "dot_ids": ["dot01"],
        "api_url": "https://api.sensinggarden.com/v1",
        "api_key": "super-secret-key",
        "s3_bucket": "scl-sensing-garden",
    }

    redacted = redact_device_config(device_config)

    assert redacted["api_key"] == "***redacted***"
    assert redacted["flick_id"] == "flick01"
    assert redacted["dot_ids"] == ["dot01"]
    assert redacted["api_url"] == "https://api.sensinggarden.com/v1"
    assert redacted["s3_bucket"] == "scl-sensing-garden"
    # original is untouched
    assert device_config["api_key"] == "super-secret-key"


def test_redact_device_config_handles_missing_api_key() -> None:
    redacted = redact_device_config({"flick_id": "flick01", "api_key": ""})

    assert redacted["api_key"] == ""


def test_log_startup_config_logs_full_detection_and_tracking(caplog, monkeypatch) -> None:
    monkeypatch.setattr("bugcam.config.get_source_commit", lambda: "abc1234")
    caplog.set_level(logging.INFO, logger="bugcam.startup")

    log_startup_config(
        version="0.8.3",
        device_config={
            "flick_id": "flick01",
            "dot_ids": ["dot01", "dot02"],
            "api_url": "https://api.sensinggarden.com/v1",
            "api_key": "super-secret-key",
            "s3_bucket": "scl-sensing-garden",
        },
        detection_config={"min_area": 0.00012, "gmm_history": 500},
        tracking_config={"max_lost_frames": 45, "w_dist": 0.6},
        detection_config_source="/opt/bugcam/detection.yaml",
    )

    text = caplog.text
    assert "0.8.3" in text
    assert "abc1234" in text
    assert "super-secret-key" not in text
    assert "***redacted***" in text
    assert "flick01" in text
    assert "dot01" in text
    assert "min_area" in text and "0.00012" in text
    assert "max_lost_frames" in text and "45" in text
    assert "/opt/bugcam/detection.yaml" in text


def test_log_startup_config_reports_commit_unavailable(caplog, monkeypatch) -> None:
    monkeypatch.setattr("bugcam.config.get_source_commit", lambda: None)
    caplog.set_level(logging.INFO, logger="bugcam.startup")

    log_startup_config(
        version="0.8.3",
        device_config={"flick_id": "flick01", "api_key": ""},
        detection_config={},
        tracking_config={},
        detection_config_source="hardcoded defaults (no detection.yaml found)",
    )

    assert "unavailable" in caplog.text
    assert "hardcoded defaults" in caplog.text


def test_log_startup_config_never_raises_when_commit_lookup_fails(caplog, monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("bugcam.config.get_source_commit", _raise)
    caplog.set_level(logging.INFO, logger="bugcam.startup")

    log_startup_config(
        version="0.8.3",
        device_config={"flick_id": "flick01", "api_key": "secret"},
        detection_config={"min_area": 0.0001},
        tracking_config={"max_lost_frames": 45},
        detection_config_source="bundled default",
    )

    assert "unavailable" in caplog.text
    assert "secret" not in caplog.text
