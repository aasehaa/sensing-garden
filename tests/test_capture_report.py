"""Sampling-effort capture reports.

Every completed recorded chunk is appended durably to captures/current.jsonl;
rotate() seals a period into a report JSON that the run command ships through
Pollen (kind="capture" -> v1/<device>/captures/<ts>.json), so with batching it
rides the per-device tar next to the results it describes.
"""
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugcam.capture_report import CaptureLog
from bugcam.pollen.pollen import Pollen, PollenConfig


class _OneShotEvent(threading.Event):
    """wait() returns False exactly once (letting one loop iteration run),
    then behaves like a normal already-set event."""

    def __init__(self):
        super().__init__()
        self._waited = False

    def wait(self, timeout=None):
        if not self._waited:
            self._waited = True
            return False
        self.set()
        return True


def _clock(*stamps: datetime):
    """A clock that returns each stamp in turn, then keeps returning the last."""
    remaining = list(stamps)

    def now() -> datetime:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return now


def _dt(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 7, 14, hour, minute, second, tzinfo=timezone.utc)


def _chunk(tmp_path: Path, name: str = "edge26_20260714_100000_000000.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"x" * 10)
    return path


class TestCaptureLogRecord:
    def test_record_chunk_appends_durable_line(self, tmp_path):
        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10)))
        log.record_chunk(_chunk(tmp_path), 60.0)

        lines = (tmp_path / "captures" / "current.jsonl").read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["video_file"] == "edge26_20260714_100000_000000.mp4"
        assert entry["duration_seconds"] == 60.0
        assert entry["size_bytes"] == 10
        assert entry["recorded_at"] == _dt(10).isoformat()

    def test_record_chunk_survives_missing_file(self, tmp_path):
        """A chunk deleted before stat (or a hw-path race) still gets a record."""
        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10)))
        log.record_chunk(tmp_path / "gone.mp4", 12.5)
        entry = json.loads((tmp_path / "captures" / "current.jsonl").read_text())
        assert entry["size_bytes"] is None
        assert entry["duration_seconds"] == 12.5


class TestCaptureLogRotate:
    def test_rotate_seals_period_report(self, tmp_path):
        log = CaptureLog(
            tmp_path / "captures", "edge26",
            clock=_clock(_dt(10), _dt(10, 5), _dt(10, 10), _dt(11)),
        )
        log.record_chunk(_chunk(tmp_path, "a.mp4"), 60.0)
        log.record_chunk(_chunk(tmp_path, "b.mp4"), 30.5)
        report_path = log.rotate()

        assert report_path == tmp_path / "captures" / "20260714_110000.json"
        report = json.loads(report_path.read_text())
        assert report["device_id"] == "edge26"
        assert report["period_start"] == _dt(10).isoformat()
        assert report["period_end"] == _dt(11).isoformat()
        assert report["sample_count"] == 2
        assert report["total_duration_seconds"] == 90.5
        assert [s["video_file"] for s in report["samples"]] == ["a.mp4", "b.mp4"]
        assert not (tmp_path / "captures" / "current.jsonl").exists()

    def test_rotate_empty_period_writes_zero_report(self, tmp_path):
        """No samples still yields a report: 'up but not sampling' is a signal."""
        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10), _dt(11)))
        report = json.loads(log.rotate().read_text())
        assert report["sample_count"] == 0
        assert report["total_duration_seconds"] == 0
        assert report["samples"] == []

    def test_periods_are_contiguous_across_rotations(self, tmp_path):
        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10), _dt(11), _dt(12)))
        first = json.loads(log.rotate().read_text())
        second = json.loads(log.rotate().read_text())
        assert first["period_end"] == second["period_start"] == _dt(11).isoformat()

    def test_corrupt_lines_are_skipped(self, tmp_path):
        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10), _dt(10, 5), _dt(11)))
        log.record_chunk(_chunk(tmp_path), 60.0)
        with open(tmp_path / "captures" / "current.jsonl", "a") as f:
            f.write("{not json\n")
        report = json.loads(log.rotate().read_text())
        assert report["sample_count"] == 1


class TestCaptureLogLockContention:
    def test_record_chunk_does_not_block_on_rotates_slow_report_write(self, tmp_path):
        """rotate() used to hold one lock across a read, a report write, and an
        unlink -- if a chunk finished mid-rotate, the recorder thread's
        record_chunk() would block on that I/O before it could start the next
        chunk, undermining the "no gaps" continuous-recording guarantee. Only
        the rename that swaps current.jsonl out needs the lock; the slow part
        must run outside it."""
        import threading

        log = CaptureLog(tmp_path / "captures", "edge26", clock=_clock(_dt(10), _dt(11)))
        log.record_chunk(_chunk(tmp_path, "a.mp4"), 60.0)

        write_started = threading.Event()
        release_write = threading.Event()
        original_write_report = log._write_report

        def slow_write_report(*args, **kwargs):
            write_started.set()
            assert release_write.wait(timeout=5), "test deadlocked"
            return original_write_report(*args, **kwargs)

        log._write_report = slow_write_report

        rotate_thread = threading.Thread(target=log.rotate)
        rotate_thread.start()
        try:
            assert write_started.wait(timeout=5), "rotate() never reached the slow report write"
            # The lock must already be released at this point: record_chunk()
            # should complete immediately, not wait for rotate() to finish.
            done = threading.Event()

            def do_record():
                log.record_chunk(_chunk(tmp_path, "b.mp4"), 5.0)
                done.set()

            recorder_thread = threading.Thread(target=do_record)
            recorder_thread.start()
            recorder_thread.join(timeout=1.0)
            assert done.is_set(), "record_chunk() blocked on rotate()'s slow report write"
        finally:
            release_write.set()
            rotate_thread.join(timeout=5)

        # b.mp4's chunk landed in the fresh (post-rotate) current.jsonl, not
        # folded into the report that was still being written when it recorded.
        lines = (tmp_path / "captures" / "current.jsonl").read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["video_file"] == "b.mp4"


class TestCaptureLogRecover:
    def test_recover_seals_leftover_current_and_lists_unshipped(self, tmp_path):
        captures = tmp_path / "captures"
        captures.mkdir(parents=True)
        (captures / "20260713_090000.json").write_text("{}")  # unshipped from prior run
        old = CaptureLog(captures, "edge26", clock=_clock(_dt(8), _dt(9), _dt(9, 30)))
        old.record_chunk(_chunk(tmp_path, "a.mp4"), 60.0)
        old.record_chunk(_chunk(tmp_path, "b.mp4"), 60.0)

        pending = CaptureLog(captures, "edge26", clock=_clock(_dt(12))).recover()

        assert not (captures / "current.jsonl").exists()
        assert len(pending) == 2
        assert pending[0].name == "20260713_090000.json"
        # The sealed leftover takes its period bounds from its sample timestamps.
        report = json.loads(pending[1].read_text())
        assert report["sample_count"] == 2
        assert report["period_start"] == _dt(9).isoformat()
        assert report["period_end"] == _dt(9, 30).isoformat()

    def test_recover_with_nothing_returns_empty(self, tmp_path):
        assert CaptureLog(tmp_path / "captures", "edge26").recover() == []


class FakeUploader:
    def upload(self, row):
        pass


class TestCaptureReportShipsLikeAnArtifact:
    def test_capture_report_key_derives_under_device(self, tmp_path):
        """kind="capture" reports live under <device>/captures/ and key to
        v1/<device>/captures/<ts>.json, so batching packs them with results."""
        cfg = PollenConfig(
            db_path=tmp_path / "pollen.db",
            output_root=tmp_path / "out",
            staging_dir=tmp_path / "staging",
        )
        cfg.output_root.mkdir(parents=True)
        pol = Pollen(cfg, uploader=FakeUploader())
        report = cfg.output_root / "flick1" / "captures" / "20260714_110000.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}")

        pol.enqueue_set([report], device="flick1", kind="capture")

        rows = pol.store.claim_pending()
        assert len(rows) == 1
        assert rows[0].s3_key == "v1/flick1/captures/20260714_110000.json"
        assert rows[0].kind == "capture"


class TestCaptureReportLoop:
    def test_loop_ships_recovered_reports_then_exits(self, tmp_path):
        """With stop pre-set the loop still ships prior-run leftovers on startup."""
        import threading
        from bugcam.commands.run import _capture_report_loop

        captures = tmp_path / "captures"
        captures.mkdir(parents=True)
        leftover = captures / "20260713_090000.json"
        leftover.write_text("{}")
        log = CaptureLog(captures, "flick1")

        shipped = []

        class FakePollen:
            def enqueue_set(self, files, *, device, kind):
                shipped.append((files[0].name, device, kind))
                return [1]

        stop = threading.Event()
        stop.set()
        _capture_report_loop(log, "flick1", stop, FakePollen(), interval=0.01)

        assert shipped == [("20260713_090000.json", "flick1", "capture")]
        assert not leftover.exists()  # shipped -> local copy dropped

    def test_loop_keeps_report_when_enqueue_fails(self, tmp_path):
        import threading
        from bugcam.commands.run import _capture_report_loop

        captures = tmp_path / "captures"
        captures.mkdir(parents=True)
        leftover = captures / "20260713_090000.json"
        leftover.write_text("{}")

        class BrokenPollen:
            def enqueue_set(self, files, *, device, kind):
                raise RuntimeError("presign down")

        stop = threading.Event()
        stop.set()
        _capture_report_loop(CaptureLog(captures, "flick1"), "flick1", stop, BrokenPollen(), interval=0.01)

        assert leftover.exists()  # kept for the next recovery pass

    def test_loop_ticks_rotate_and_ships_on_the_timer(self, tmp_path):
        """The loop's core job -- rotate() and ship on a timer -- was
        untested: both tests above pre-set stop before calling, so the loop
        body (`while not stop_event.wait(interval): _ship(capture_log.rotate())`)
        never ran; only the startup recover() path was covered."""
        from bugcam.commands.run import _capture_report_loop

        log = CaptureLog(tmp_path / "captures", "flick1", clock=_clock(_dt(10), _dt(11)))

        shipped = []

        class FakePollen:
            def enqueue_set(self, files, *, device, kind):
                shipped.append((files[0].name, device, kind))
                return [1]

        stop = _OneShotEvent()
        _capture_report_loop(log, "flick1", stop, FakePollen(), interval=0.01)

        assert shipped == [("20260714_110000.json", "flick1", "capture")]


class TestRecorderNotifiesChunkComplete:
    def _recorder(self, tmp_path, callback):
        from bugcam.edge26.recorder import VideoRecorder

        return VideoRecorder(
            output_dir=str(tmp_path),
            fps=2,
            chunk_duration=1,
            resolution=(64, 64),
            device_id="edge26",
            on_chunk_complete=callback,
        )

    def test_record_chunk_reports_path_and_measured_duration(self, tmp_path):
        seen = []
        recorder = self._recorder(tmp_path, lambda path, duration: seen.append((path, duration)))
        recorder.resolution = (64, 64)
        recorder.frame_queue = queue.Queue()
        recorder.frame_queue.put("frame1")
        recorder.frame_queue.put("frame2")

        def fake_writer(path, *args, **kwargs):
            writer = MagicMock()
            writer.isOpened.return_value = True
            writer.release.side_effect = lambda: Path(path).write_bytes(b"v")
            return writer

        with patch("bugcam.edge26.recorder.cv2.VideoWriter", side_effect=fake_writer):
            chunk_path = recorder._record_chunk()

        assert chunk_path is not None
        assert seen == [(chunk_path, 1.0)]  # 2 frames @ 2 fps

    def test_callback_failure_does_not_break_recording(self, tmp_path):
        def boom(path, duration):
            raise RuntimeError("consumer bug")

        recorder = self._recorder(tmp_path, boom)
        recorder.resolution = (64, 64)
        recorder.frame_queue = queue.Queue()
        recorder.frame_queue.put("frame1")
        recorder.frame_queue.put("frame2")

        def fake_writer(path, *args, **kwargs):
            writer = MagicMock()
            writer.isOpened.return_value = True
            writer.release.side_effect = lambda: Path(path).write_bytes(b"v")
            return writer

        with patch("bugcam.edge26.recorder.cv2.VideoWriter", side_effect=fake_writer):
            assert recorder._record_chunk() is not None  # chunk still completes
