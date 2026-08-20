from pathlib import Path

from bugcam.settings import get_configured_flick_id, load_device_config, resolve_flick_id


def test_load_device_config_reads_flick_id_from_persistent_config(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(
        '{"flick_id": "flick-config", "dot_ids": ["dot01"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    device_config = load_device_config()

    assert device_config.flick_id == "flick-config"
    assert device_config.dot_ids == ["dot01"]


def test_resolve_flick_id_prefers_cli_override(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"flick_id": "flick-config"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert resolve_flick_id("flick-cli") == "flick-cli"


def test_load_device_config_falls_back_to_legacy_device_id(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"device_id": "legacy-flick"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    device_config = load_device_config()

    assert device_config.flick_id == "legacy-flick"


def test_get_configured_flick_id_empty_when_unconfigured(tmp_path: Path, monkeypatch) -> None:
    """Unlike load_device_config()/resolve_flick_id(), this must never default --
    callers (status/dot-info/heartbeat, and run/process/record/environment's
    fail-fast guard) rely on "" meaning 'not configured'."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("BUGCAM_FLICK_ID", raising=False)

    assert get_configured_flick_id() == ""


def test_get_configured_flick_id_reads_env_var(tmp_path: Path, monkeypatch) -> None:
    """A device configured purely via BUGCAM_FLICK_ID (no config.json, e.g. a
    fleet deployed through systemd env files) counts as configured."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("BUGCAM_FLICK_ID", "flick-env")

    assert get_configured_flick_id() == "flick-env"


def test_get_configured_flick_id_prefers_persisted_config_over_env_var(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"flick_id": "flick-config"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("BUGCAM_FLICK_ID", "flick-env")

    assert get_configured_flick_id() == "flick-config"


def test_get_configured_flick_id_reads_persisted_value(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"flick_id": "flick-config"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert get_configured_flick_id() == "flick-config"


def test_get_configured_flick_id_falls_back_to_legacy_device_id(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / ".config" / "bugcam"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text('{"device_id": "legacy-flick"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert get_configured_flick_id() == "legacy-flick"
