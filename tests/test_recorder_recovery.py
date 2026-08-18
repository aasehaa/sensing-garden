"""
Failure recovery in the hardware (picamera2) recording path.

Covers the encoder-occupied faulty state: a failed stop_recording() leaves the
encoder attached, every subsequent start_recording() opens a new output file
and then throws, and the old code retried in a tight loop forever. The
recorder must instead clean up stray files, rebuild the camera + encoder, and
after repeated consecutive failures exit the process (nonzero) so systemd's
Restart=on-failure recreates it.
"""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import bugcam.edge26.recorder as recorder_module
from bugcam.edge26.recorder import VideoRecorder


def make_recorder(tmp_path: Path, **overrides) -> VideoRecorder:
    kwargs = dict(
        output_dir=str(tmp_path),
        fps=10,
        chunk_duration=0,
        resolution=(640, 480),
        device_id="test",
        use_picamera=True,
    )
    kwargs.update(overrides)
    rec = VideoRecorder(**kwargs)
    # Skip the remux path: chunks are renamed .h264 -> .mp4 directly.
    rec._check_ffmpeg_available = lambda: False
    return rec


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    monkeypatch.setattr(recorder_module, "RECOVERY_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(recorder_module, "NO_CHUNK_RETRY_SECONDS", 0.01)


def install_scripted_camera(rec: VideoRecorder, behaviors: list) -> list:
    """Replace rec._init_camera with one that installs a fresh fake camera.

    Each start_recording() call consumes the next behavior:
      - "fail": create the output file (as picamera2 does), then raise the
        encoder-occupied error.
      - "ok": create a non-empty output file and return.
      - "end" (implicit once the script runs out): create a non-empty output
        file and set the recorder's stop_event so the loop terminates.

    Returns the list of fake cameras created, in order.
    """
    cameras = []
    script = iter(behaviors)

    def fake_init():
        cam = MagicMock(name=f"camera{len(cameras)}")

        def fake_start_recording(encoder, path, quality=None):
            action = next(script, "end")
            if action == "fail":
                Path(path).touch()
                raise RuntimeError("encoder already running")
            Path(path).write_bytes(b"data")
            if action == "end":
                rec.stop_event.set()

        cam.start_recording.side_effect = fake_start_recording
        cameras.append(cam)
        rec.camera = cam
        rec.encoder = MagicMock()
        rec.encoder_quality = None

    rec._init_camera = fake_init
    return cameras


def patched_exit(monkeypatch) -> MagicMock:
    exit_mock = MagicMock(side_effect=SystemExit)
    monkeypatch.setattr(recorder_module.os, "_exit", exit_mock)
    return exit_mock


def test_failed_start_leaves_no_stray_h264_files(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    install_scripted_camera(rec, ["fail", "fail"])
    exit_mock = patched_exit(monkeypatch)

    rec.start()

    assert list(tmp_path.glob("*.h264")) == []
    exit_mock.assert_not_called()
    # Only the final "end" chunk remains.
    assert len(list(tmp_path.glob("*.mp4"))) == 1


def test_recovery_releases_encoder_and_reinits_camera(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    cameras = install_scripted_camera(rec, ["fail"])
    exit_mock = patched_exit(monkeypatch)

    rec.start()

    # The failed camera must be released (stop_recording to detach the
    # encoder, close to free the device) and a fresh camera built.
    assert len(cameras) == 2
    assert cameras[0].stop_recording.called
    assert cameras[0].close.called
    exit_mock.assert_not_called()


def test_success_resets_consecutive_failure_counter(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    install_scripted_camera(rec, ["fail", "fail", "ok", "fail", "fail"])
    exit_mock = patched_exit(monkeypatch)

    rec.start()  # raises SystemExit if the counter never reset

    exit_mock.assert_not_called()


def test_exits_process_after_max_consecutive_failures(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    cameras = install_scripted_camera(rec, ["fail"] * 10)
    exit_mock = patched_exit(monkeypatch)

    with pytest.raises(SystemExit):
        rec.start()

    exit_mock.assert_called_once()
    assert exit_mock.call_args[0][0] != 0
    # Exactly MAX attempts were made, not an unbounded retry loop.
    attempts = sum(c.start_recording.call_count for c in cameras)
    assert attempts == recorder_module.MAX_CONSECUTIVE_FAILURES


def test_no_escalation_when_stop_requested(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    monkeypatch.setattr(recorder_module, "MAX_CONSECUTIVE_FAILURES", 1)
    exit_mock = patched_exit(monkeypatch)

    def fake_init():
        cam = MagicMock()

        def boom(encoder, path, quality=None):
            rec.stop_event.set()
            raise RuntimeError("camera died during shutdown")

        cam.start_recording.side_effect = boom
        rec.camera = cam
        rec.encoder = MagicMock()
        rec.encoder_quality = None

    rec._init_camera = fake_init

    rec.start()

    exit_mock.assert_not_called()


def test_startup_init_failure_escalates(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    exit_mock = patched_exit(monkeypatch)
    init_calls = []

    def failing_init():
        init_calls.append(1)
        raise RuntimeError("failed to start camera")

    rec._init_camera = failing_init

    with pytest.raises(SystemExit):
        rec.start()

    exit_mock.assert_called_once()
    assert len(init_calls) == recorder_module.MAX_CONSECUTIVE_FAILURES


def test_interval_mode_failure_counts_and_escalates(tmp_path, monkeypatch):
    rec = make_recorder(
        tmp_path, recording_mode="interval", interval_minutes=0
    )
    exit_mock = patched_exit(monkeypatch)
    init_calls = []

    def failing_init():
        init_calls.append(1)
        raise RuntimeError("failed to start camera")

    rec._init_camera = failing_init

    with pytest.raises(SystemExit):
        rec.start()

    exit_mock.assert_called_once()
    assert len(init_calls) == recorder_module.MAX_CONSECUTIVE_FAILURES


def install_scripted_opencv(rec: VideoRecorder, behaviors: list) -> dict:
    """Fake the legacy OpenCV path's per-iteration calls.

    Each _record_chunk() call consumes the next behavior:
      - "fail": raise RuntimeError (simulates a frame-grab/writer failure).
      - "ok": write and return a fresh chunk path.
      - "end" (implicit once the script runs out): write and return a fresh
        chunk path, then set the recorder's stop_event so the loop terminates.

    Returns call counters for _init_camera/_start_grabber, so a test can
    assert whether a failure re-initialized the camera.
    """
    calls = {"init": 0, "grabber": 0}
    script = iter(behaviors)
    chunk_index = [0]

    def fake_init():
        calls["init"] += 1

    def fake_grabber():
        calls["grabber"] += 1

    def fake_record_chunk():
        action = next(script, "end")
        if action == "fail":
            raise RuntimeError("frame grab failed")
        chunk_index[0] += 1
        path = Path(rec.output_dir) / f"chunk{chunk_index[0]}.mp4"
        path.write_bytes(b"data")
        if action == "end":
            rec.stop_event.set()
        return path

    rec._init_camera = fake_init
    rec._start_grabber = fake_grabber
    rec._record_chunk = fake_record_chunk
    return calls


def test_opencv_path_recovers_from_chunk_failure_instead_of_dying(tmp_path, monkeypatch):
    """Before this fix, any exception in the OpenCV path's while-loop body was
    caught by one blanket except around the whole loop, logged once, and the
    loop exited for good -- no retry. It must instead recover like the
    hardware path: teardown, backoff, re-init, keep recording."""
    rec = make_recorder(tmp_path, use_picamera=False)
    calls = install_scripted_opencv(rec, ["ok", "fail", "end"])
    exit_mock = patched_exit(monkeypatch)

    rec.start()

    exit_mock.assert_not_called()
    # Recovered and kept going: both the "ok" and "end" chunks landed.
    assert len(list(tmp_path.glob("chunk*.mp4"))) == 2
    # The failure forced a teardown + re-init before the next attempt.
    assert calls["init"] == 2


def test_opencv_path_escalates_after_max_consecutive_failures(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path, use_picamera=False)
    calls = install_scripted_opencv(rec, ["fail"] * 10)
    exit_mock = patched_exit(monkeypatch)

    with pytest.raises(SystemExit):
        rec.start()

    exit_mock.assert_called_once()
    assert exit_mock.call_args[0][0] != 0
    assert calls["init"] == recorder_module.MAX_CONSECUTIVE_FAILURES


class _OneShotEvent(threading.Event):
    """Event that records wait() timeouts and trips after the first wait."""

    def __init__(self):
        super().__init__()
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        self.set()
        return True


def test_finalize_disk_error_is_not_counted_as_camera_failure(tmp_path, monkeypatch):
    """A rename/stat error after a successful start/stop_recording is a
    filesystem hiccup, not a camera fault: it must not tear down or rebuild a
    healthy camera, and must not count toward failure escalation."""
    rec = make_recorder(tmp_path)
    cameras = install_scripted_camera(rec, ["ok", "end"])
    exit_mock = patched_exit(monkeypatch)
    teardown_mock = MagicMock()
    rec._teardown_for_recovery = teardown_mock

    def failing_rename(self, target):
        raise OSError("simulated disk write error")

    monkeypatch.setattr(Path, "rename", failing_rename)

    rec.start()

    exit_mock.assert_not_called()
    teardown_mock.assert_not_called()
    # Only one camera ever created -- a finalize error didn't trigger a rebuild.
    assert len(cameras) == 1


def test_skipped_chunk_waits_instead_of_spinning(tmp_path, monkeypatch):
    rec = make_recorder(tmp_path)
    cameras = install_scripted_camera(rec, [])
    rec.stop_event = _OneShotEvent()
    monkeypatch.setattr(
        recorder_module.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(total=1, used=1, free=0),
    )
    exit_mock = patched_exit(monkeypatch)

    rec.start()

    # Low disk skipped the chunk: no recording attempted, and the loop
    # waited on stop_event instead of retrying immediately.
    assert all(not c.start_recording.called for c in cameras)
    assert recorder_module.NO_CHUNK_RETRY_SECONDS in rec.stop_event.wait_timeouts
    exit_mock.assert_not_called()
