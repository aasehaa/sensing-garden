"""Pipeline metrics for heartbeat reporting: windowed stage timings that work
across the detection-subprocess boundary, plus the Pipeline health snapshot."""
import multiprocessing
from unittest.mock import patch

import pytest

from bugcam.edge26.metrics import StageTimings


def _record_two(timings: StageTimings) -> None:
    timings.record(1.0)
    timings.record(3.0)


class TestStageTimings:
    def test_snapshot_reports_count_avg_max_and_totals(self):
        t = StageTimings()
        for seconds in (1.0, 3.0, 2.0):
            t.record(seconds)
        snap = t.snapshot()
        assert snap["count"] == 3
        assert snap["avg_seconds"] == pytest.approx(2.0)
        assert snap["max_seconds"] == pytest.approx(3.0)
        assert snap["total_count"] == 3
        assert snap["total_seconds"] == pytest.approx(6.0)

    def test_snapshot_resets_window_but_keeps_totals(self):
        t = StageTimings()
        t.record(1.0)
        t.record(3.0)
        t.snapshot()
        t.record(5.0)
        snap = t.snapshot()
        assert snap["count"] == 1
        assert snap["avg_seconds"] == pytest.approx(5.0)
        assert snap["max_seconds"] == pytest.approx(5.0)
        assert snap["total_count"] == 3
        assert snap["total_seconds"] == pytest.approx(9.0)

    def test_empty_window_has_null_avg_and_max(self):
        snap = StageTimings().snapshot()
        assert snap["count"] == 0
        assert snap["avg_seconds"] is None
        assert snap["max_seconds"] is None
        assert snap["total_count"] == 0

    def test_snapshot_without_reset_leaves_window(self):
        t = StageTimings()
        t.record(2.0)
        t.snapshot(reset=False)
        snap = t.snapshot()
        assert snap["count"] == 1

    def test_child_process_records_are_visible_to_parent(self):
        # The pipeline spawns its detection child with the "spawn" context;
        # timings recorded there must land in the parent's snapshot.
        ctx = multiprocessing.get_context("spawn")
        t = StageTimings(ctx)
        child = ctx.Process(target=_record_two, args=(t,))
        child.start()
        child.join(timeout=30)
        assert child.exitcode == 0
        snap = t.snapshot()
        assert snap["count"] == 2
        assert snap["max_seconds"] == pytest.approx(3.0)


class TestPipelineHealthSnapshot:
    def _pipeline(self, tmp_path):
        from bugcam.edge26 import pipeline as edge26_main

        with patch.object(edge26_main, "VideoProcessor"):
            return edge26_main.Pipeline(
                {
                    "device": {"flick_id": "flick01", "dot_ids": ["dot01"]},
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

    def test_snapshot_reports_queues_timings_and_counters(self, tmp_path):
        pipeline = self._pipeline(tmp_path)
        pipeline.metrics.detection.record(2.0)
        pipeline.metrics.classification.record(0.5)
        pipeline.metrics.unhealthy_results.increment()

        snap = pipeline.health_snapshot()

        assert snap["video_queue"] == 0
        assert snap["classification_queue"] == 0
        assert snap["detection"]["count"] == 1
        assert snap["detection"]["max_seconds"] == pytest.approx(2.0)
        assert snap["classification"]["count"] == 1
        assert snap["unhealthy_results"] == 1
        assert snap["workers"] == {}  # nothing started
        # Process-lifetime uptime: resets on a service restart, unlike the
        # system uptime in the base payload, so crash-loops are visible.
        assert 0.0 <= snap["uptime_seconds"] < 60.0

    def test_snapshot_resets_timing_window(self, tmp_path):
        pipeline = self._pipeline(tmp_path)
        pipeline.metrics.detection.record(2.0)
        pipeline.health_snapshot()
        snap = pipeline.health_snapshot()
        assert snap["detection"]["count"] == 0
        assert snap["detection"]["total_count"] == 1
