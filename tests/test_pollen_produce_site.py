"""The produce-site result hook: a finalized result dir enqueues immediately.

Exercises enqueue_result_dir (what the pipeline's on_result_ready callback calls)
and that the pipeline invokes the callback through its safe wrapper.
"""
import json
from pathlib import Path

from bugcam.pollen.pollen import Pollen, PollenConfig
from bugcam.edge26.result_publish import publish_result_dir


class FakeUploader:
    def upload(self, row):
        pass


def _pollen(tmp_path):
    cfg = PollenConfig(
        db_path=tmp_path / "pollen.db",
        output_root=tmp_path / "out",
        staging_dir=tmp_path / "staging",
    )
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    return Pollen(cfg, uploader=FakeUploader()), cfg.output_root


def _result_dir(out: Path, device: str, name: str, tracks=("t1",)) -> Path:
    rd = out / device / name
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "results.json").write_text(json.dumps({"tracks": [{"track_id": t} for t in tracks]}), encoding="utf-8")
    for t in tracks:
        (rd / "crops" / t).mkdir(parents=True, exist_ok=True)
        (rd / "crops" / t / "frame_000000.jpg").write_bytes(b"img")
    return rd


class TestEnqueueResultDir:
    def test_flik_dir_enqueues_files(self, tmp_path):
        pol, out = _pollen(tmp_path)
        rd = _result_dir(out, "flick1", "20260204_120000")
        n = publish_result_dir(pol, rd, "flick1", [])
        assert n == 2
        assert all(r.metadata.get("device") == "flick1" for r in pol.store.claim_pending())

    def test_dot_dir_enqueues_with_device(self, tmp_path):
        pol, out = _pollen(tmp_path)
        rd = _result_dir(out, "dot1", "20260204")
        publish_result_dir(pol, rd, "flick1", ["dot1"])
        assert all(r.metadata.get("device") == "dot1" for r in pol.store.claim_pending())

    def test_empty_flik_dir_removed(self, tmp_path):
        pol, out = _pollen(tmp_path)
        rd = _result_dir(out, "flick1", "20260204_120000", tracks=())
        assert publish_result_dir(pol, rd, "flick1", []) == 0
        assert not rd.exists()

    def test_unknown_device_ignored(self, tmp_path):
        pol, out = _pollen(tmp_path)
        rd = _result_dir(out, "stranger", "20260204_120000")
        assert publish_result_dir(pol, rd, "flick1", ["dot1"]) == 0


class TestPipelineHook:
    def test_pipeline_invokes_callback_safely(self):
        # The pipeline calls the callback through a guard that swallows errors so a
        # failing upload owner never breaks classification.
        from bugcam.edge26.pipeline import Pipeline

        calls = []
        pipe = Pipeline.__new__(Pipeline)
        pipe._on_result_ready = lambda d: calls.append(d)
        pipe._notify_result_ready(Path("/out/flick1/chunk"))
        assert calls == [Path("/out/flick1/chunk")]

        # a raising callback is contained
        pipe._on_result_ready = lambda d: (_ for _ in ()).throw(RuntimeError("boom"))
        pipe._notify_result_ready(Path("/out/x"))  # must not raise

    def test_no_callback_is_a_noop(self):
        from bugcam.edge26.pipeline import Pipeline

        pipe = Pipeline.__new__(Pipeline)
        pipe._on_result_ready = None
        pipe._notify_result_ready(Path("/out/x"))  # must not raise
