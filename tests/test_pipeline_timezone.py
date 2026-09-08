"""P-03: Pipeline.__init__ hand-rolled its own ZoneInfo(name) + except,
duplicating record_window.resolve_zone()'s validation with a different
failure mode (raise vs. silent fallback+log) and a different error message.
A future change to what counts as a valid zone name could land in one copy
and not the other, letting the recording window's zone and the pipeline's
day-boundary zone silently diverge for the same configured value. Pipeline
must delegate the actual validation to resolve_zone()."""
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bugcam.edge26 import pipeline as edge26_main


def _pipeline(tmp_path, timezone_name):
    with patch.object(edge26_main, "VideoProcessor"):
        return edge26_main.Pipeline(
            {
                "device": {"flick_id": "flick01", "dot_ids": [], "timezone": timezone_name},
                "paths": {
                    "input_storage": str(tmp_path / "input"),
                    "pending_dir": str(tmp_path / "pending"),
                },
                "pipeline": {
                    "enable_recording": False,
                    "enable_processing": True,
                    "enable_classification": False,
                },
                "output": {"results_dir": str(tmp_path / "output")},
            }
        )


def test_valid_timezone_resolves_via_resolve_zone(tmp_path):
    pipeline = _pipeline(tmp_path, "Europe/Amsterdam")
    assert pipeline.timezone_name == "Europe/Amsterdam"
    assert pipeline._local_zone == ZoneInfo("Europe/Amsterdam")


def test_invalid_timezone_falls_back_to_utc_and_logs(tmp_path, caplog):
    with caplog.at_level("ERROR"):
        pipeline = _pipeline(tmp_path, "Not/AZone")
    assert pipeline.timezone_name is None
    assert pipeline._local_zone is None
    assert any("Not/AZone" in r.message for r in caplog.records)


def test_unconfigured_timezone_stays_none_not_system_local(tmp_path):
    """Unlike resolve_zone(None) (used for the recording window, which needs
    *some* zone to evaluate is_open() against), Pipeline's day-boundary zone
    must stay None when unconfigured -- day boundaries are UTC, not the
    system's local zone, per this attribute's own contract."""
    pipeline = _pipeline(tmp_path, None)
    assert pipeline.timezone_name is None
    assert pipeline._local_zone is None


def test_delegates_validation_to_resolve_zone(tmp_path, caplog):
    """Pin the actual fix, not just its externally-visible effect: Pipeline
    must call record_window.resolve_zone() (and catch what it raises) rather
    than re-validating with its own inline ZoneInfo() try/except."""
    with patch.object(edge26_main, "resolve_zone", side_effect=ValueError("boom")) as mock_resolve:
        with caplog.at_level("ERROR"):
            pipeline = _pipeline(tmp_path, "Europe/Amsterdam")

    mock_resolve.assert_called_once_with("Europe/Amsterdam")
    assert pipeline.timezone_name is None
    assert pipeline._local_zone is None
