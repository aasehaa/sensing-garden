"""Tests for edge26/detection.py: VideoProcessor construction."""
from unittest.mock import patch


def test_video_processor_passes_ratio_config_to_bugspot() -> None:
    from bugcam.edge26.detection import VideoProcessor

    detection = {
        "min_area": 0.00012,
        "max_area": 0.0015,
        "min_displacement": 0.25,
        "max_frame_jump": 0.06,
        "revisit_radius": 0.025,
        "morph_kernel_size": 5,
    }
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
