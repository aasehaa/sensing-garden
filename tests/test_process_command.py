"""Tests for the `bugcam process` CLI command."""
from pathlib import Path
from unittest.mock import patch


def test_process_uses_device_config_flick_id(tmp_path: Path) -> None:
    from bugcam.commands import process as process_command

    with patch('bugcam.commands.process.load_device_config') as mock_device_config, \
         patch('bugcam.commands.process.resolve_bundle_provenance', return_value={"model_id": "bundle", "model_sha256": "abc123456789"}), \
         patch('bugcam.commands.process.build_pipeline') as mock_build_pipeline, \
         patch('bugcam.commands.process.require_configured_flick_id', return_value="flick-config"):
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
