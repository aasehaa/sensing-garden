from pathlib import Path

from bugcam.settings import load_device_config, resolve_flick_id


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
