"""Tests for bugcam/commands/__init__.py shared CLI helpers."""
from pathlib import Path

import pytest
import typer

from bugcam.commands import require_configured_flick_id


def _isolate_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("BUGCAM_FLICK_ID", raising=False)


def test_require_configured_flick_id_raises_when_nothing_configured(tmp_path: Path, monkeypatch) -> None:
    """run/process/record/environment must not silently operate as the
    hardcoded default device (bugcam-rpi) on a fresh, never-setup install."""
    _isolate_config(tmp_path, monkeypatch)

    with pytest.raises(typer.BadParameter, match="No device id configured"):
        require_configured_flick_id(None)


def test_require_configured_flick_id_accepts_cli_override(tmp_path: Path, monkeypatch) -> None:
    _isolate_config(tmp_path, monkeypatch)

    assert require_configured_flick_id("flick-cli") == "flick-cli"


def test_require_configured_flick_id_accepts_persisted_config(tmp_path: Path, monkeypatch) -> None:
    _isolate_config(tmp_path, monkeypatch)
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"flick_id": "flick-config"}', encoding="utf-8")

    assert require_configured_flick_id(None) == "flick-config"


def test_require_configured_flick_id_accepts_env_var(tmp_path: Path, monkeypatch) -> None:
    _isolate_config(tmp_path, monkeypatch)
    monkeypatch.setenv("BUGCAM_FLICK_ID", "flick-env")

    assert require_configured_flick_id(None) == "flick-env"
