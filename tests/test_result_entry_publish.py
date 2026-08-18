"""Zero-track FLIK result dirs publish via the disk queue (subprocess-safe).

Detection may run in a subprocess with no upload callback, so a finalized
zero-track dir (empty results + optional fallback sample video.mp4) must not be
published from the detection side. Instead detection enqueues an
entry_type="result" task and the main-process classification worker -- which
holds Pollen -- fires the publish. Pins:
  - the worker handler publishes the dir through on_result_ready;
  - a dir with a fallback video.mp4 actually ships (video counts as media);
  - the handler is idempotent when the dir is already published/cleaned.

DOT video behavior (the same disk-queue hop) is covered by
tests/test_dot_per_track.py.
"""
import json
from pathlib import Path

from bugcam.edge26.pipeline import Pipeline
from bugcam.edge26.queue import QueueEntry
from bugcam.edge26.result_publish import publish_result_dir
from bugcam.pollen.pollen import Pollen, PollenConfig


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


def _zero_track_dir(out: Path, flick: str, chunk: str, with_video=True) -> Path:
    """A finalized zero-track FLIK chunk dir: out/<flick>/<chunk_ts>/."""
    rd = out / flick / chunk
    rd.mkdir(parents=True, exist_ok=True)
    rec = {
        "source_device": flick,
        "date": chunk[:8],
        "summary": {"total_detections": 0, "total_tracks": 0,
                    "confirmed_tracks": 0, "unconfirmed_tracks": 0},
        "tracks": [],
    }
    (rd / "results.json").write_text(json.dumps(rec), encoding="utf-8")
    (rd / ".done").write_text("classified=0\nexpected=0\n", encoding="utf-8")
    if with_video:
        (rd / "video.mp4").write_bytes(b"vid")
    return rd


def _result_entry(rd: Path, flick: str) -> QueueEntry:
    return QueueEntry(entry_type="result", source_device=flick, date=rd.name[:8],
                      time=None, track_id=rd.name, track_dir=str(rd),
                      output_dir=str(rd))


def test_result_entry_fires_publish_callback(tmp_path):
    pipe = Pipeline.__new__(Pipeline)
    published = []
    pipe._on_result_ready = published.append
    rd = _zero_track_dir(tmp_path / "out", "FLIK4", "20260710_153602_038477")

    pipe._publish_finalized_result(_result_entry(rd, "FLIK4"))

    assert published == [rd]


def test_result_entry_ships_fallback_video_dir(tmp_path):
    """End-to-end for the stranded case: zero tracks + sample video.mp4."""
    pol, out = _pollen(tmp_path)
    pipe = Pipeline.__new__(Pipeline)
    # The worker runs in the main process, where on_result_ready/Pollen are wired.
    pipe._on_result_ready = lambda d: publish_result_dir(pol, d, "FLIK4", [])
    rd = _zero_track_dir(out, "FLIK4", "20260710_153602_038477")

    pipe._publish_finalized_result(_result_entry(rd, "FLIK4"))

    rows = pol.store.claim_pending()
    keys = {r.s3_key.rsplit("/", 1)[-1] for r in rows}
    assert keys == {"results.json", "video.mp4"}
    assert not rd.exists(), "published FLIK dir must be deleted after enqueue"


def test_result_entry_noop_when_dir_already_published(tmp_path):
    pipe = Pipeline.__new__(Pipeline)
    pipe._on_result_ready = lambda d: (_ for _ in ()).throw(
        AssertionError("must not publish a dir that is already gone"))
    rd = tmp_path / "out" / "FLIK4" / "20260710_153602_038477"  # never created

    pipe._publish_finalized_result(_result_entry(rd, "FLIK4"))
