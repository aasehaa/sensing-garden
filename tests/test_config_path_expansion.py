"""config.py path getters must expand a leading '~' from env vars and config.json values."""
import pytest

from bugcam import config


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in (
        "XDG_DATA_HOME", "XDG_CACHE_HOME", "BUGCAM_STATE_DIR", "BUGCAM_INPUT_DIR",
        "BUGCAM_OUTPUT_DIR", "BUGCAM_PENDING_DIR", "BUGCAM_EDGE26_LABELS",
        "BUGCAM_EDGE26_TAXONOMY_CACHE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "load_config", lambda: {})
    return tmp_path


def test_state_dir_env_var_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("BUGCAM_STATE_DIR", "~/data/bugcam")
    assert config.get_state_dir() == _home / "data" / "bugcam"


def test_state_dir_config_value_tilde_expands(monkeypatch, _home):
    monkeypatch.setattr(config, "load_config", lambda: {"state_dir": "~/data/bugcam"})
    assert config.get_state_dir() == _home / "data" / "bugcam"


def test_xdg_data_home_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("XDG_DATA_HOME", "~/xdg-data")
    assert config.get_state_dir() == _home / "xdg-data" / "bugcam"


def test_xdg_cache_home_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("XDG_CACHE_HOME", "~/xdg-cache")
    assert config.get_cache_dir() == _home / "xdg-cache" / "bugcam"


def test_input_storage_dir_env_var_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("BUGCAM_INPUT_DIR", "~/incoming")
    assert config.get_input_storage_dir() == _home / "incoming"


def test_pending_dir_config_value_tilde_expands(monkeypatch, _home):
    monkeypatch.setattr(config, "load_config", lambda: {"pending_dir": "~/pending"})
    assert config.get_pending_dir() == _home / "pending"


def test_output_storage_dir_env_var_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("BUGCAM_OUTPUT_DIR", "~/outputs")
    assert config.get_output_storage_dir() == _home / "outputs"


def test_edge26_taxonomy_cache_env_var_tilde_expands(monkeypatch, _home):
    monkeypatch.setenv("BUGCAM_EDGE26_TAXONOMY_CACHE", "~/taxonomy.json")
    assert config.get_edge26_taxonomy_cache_path() == _home / "taxonomy.json"


def test_absolute_paths_pass_through_unchanged(monkeypatch, _home, tmp_path):
    abs_dir = tmp_path / "elsewhere"
    monkeypatch.setenv("BUGCAM_STATE_DIR", str(abs_dir))
    assert config.get_state_dir() == abs_dir
