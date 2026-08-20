"""Tests for the `bugcam run` CLI command."""
from unittest.mock import patch


def test_run_resolves_flick_id_from_config() -> None:
    from bugcam.commands.run import _resolve_runtime_settings

    with patch('bugcam.commands.run.load_config', return_value={"api_key": "key", "s3_bucket": "bucket"}), \
         patch('bugcam.commands.run.resolve_flick_id', return_value="flick-config"):
        settings = _resolve_runtime_settings(None, None, None, None, None)

    assert settings["flick_id"] == "flick-config"
