"""Pipeline wiring for the --random-sampling flag: config['pipeline']['random_sampling']
must reach Pipeline._random_sampling, defaulting to False (today's confirmed-track
priority) when unset."""
from unittest.mock import patch

from bugcam.edge26 import pipeline as edge26_main


def _pipeline(tmp_path, *, random_sampling=None):
    pipeline_config = {
        "enable_recording": False,
        "enable_processing": True,
        "enable_classification": False,
    }
    if random_sampling is not None:
        pipeline_config["random_sampling"] = random_sampling
    with patch.object(edge26_main, "VideoProcessor"):
        return edge26_main.Pipeline(
            {
                "device": {"flick_id": "flick01", "dot_ids": [], "timezone": None},
                "paths": {
                    "input_storage": str(tmp_path / "input"),
                    "pending_dir": str(tmp_path / "pending"),
                },
                "pipeline": pipeline_config,
                "output": {"results_dir": str(tmp_path / "output")},
            }
        )


def test_random_sampling_defaults_to_false(tmp_path):
    pipeline = _pipeline(tmp_path)
    assert pipeline._random_sampling is False


def test_random_sampling_true_is_wired_through(tmp_path):
    pipeline = _pipeline(tmp_path, random_sampling=True)
    assert pipeline._random_sampling is True


def test_random_sampling_false_is_wired_through(tmp_path):
    pipeline = _pipeline(tmp_path, random_sampling=False)
    assert pipeline._random_sampling is False
