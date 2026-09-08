"""DOT per-track upload unit (spec: track is the primary DOT unit).

Failing-first contract for the per-track restructure. Pins the produce-site
behavior:
  - a per-track DOT dir nested under the date enqueues with the dot_id (not the
    date dir name) and is deleted after enqueue (terminal unit, like FLIK);
  - an empty/low-confidence per-track DOT dir is deleted with nothing uploaded;
  - a DOT video enqueues as kind="video" under v1/<dot>/<YYYYMMDD>/videos/ with
    its phone-given filename verbatim, not gated on any results .done.

FLIK behavior is covered (unchanged) by tests/test_pollen_produce_site.py.
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


def _dot_track_dir(out: Path, dot: str, date: str, track: str, tracks_present=True) -> Path:
    """A per-track DOT result dir: out/<dot>/<YYYYMMDD>/<track_id>_<HHMMSS>/."""
    rd = out / dot / date / track
    rd.mkdir(parents=True, exist_ok=True)
    rec = {"source_device": dot, "date": date, "tracks": []}
    if tracks_present:
        bare_id, ts = track.rsplit("_", 1)  # dir name is {track_id}_{HHMMSS}; id field is bare
        rec["tracks"] = [{"track_id": bare_id, "timestamp": ts}]
        (rd / "crops" / track).mkdir(parents=True, exist_ok=True)
        (rd / "crops" / track / "frame_000000.jpg").write_bytes(b"img")
    (rd / "results.json").write_text(json.dumps(rec), encoding="utf-8")
    (rd / ".done").write_text("classified=1\nexpected=1\n", encoding="utf-8")
    return rd


# Criterion 1 — per-track result lands correctly, dir deleted after enqueue.
def test_dot_per_track_enqueues_with_dot_id_and_is_deleted(tmp_path):
    pol, out = _pollen(tmp_path)
    rd = _dot_track_dir(out, "dot1", "20260204", "9f2a7c01_120500")

    n = publish_result_dir(pol, rd, "flick1", ["dot1"])

    rows = pol.store.claim_pending()
    assert n == 2  # results.json + 1 crop
    assert all(r.metadata.get("device") == "dot1" for r in rows)
    assert any(r.s3_key.endswith("/dot1/20260204/9f2a7c01_120500/results.json") for r in rows)
    assert not rd.exists(), "terminal per-track DOT dir must be deleted after enqueue"


# Criterion 2 — the device keeps track_id BARE + date + per-track timestamp. The
# tracks table keys on (device_id, timestamp) (identical across the cutover), and the
# backend rebuilds crop/composite keys as {track_id}_{timestamp} -- both need bare.
def test_dot_per_track_results_json_keeps_bare_id_and_time(tmp_path):
    _, out = _pollen(tmp_path)
    rd = _dot_track_dir(out, "dot1", "20260204", "9f2a7c01_120500")
    rec = json.loads((rd / "results.json").read_text())
    assert rec["date"] == "20260204"
    assert rec["tracks"][0]["track_id"] == "9f2a7c01"
    assert rec["tracks"][0]["timestamp"] == "120500"


# Instrumentation — the publish stage emits an INFO stage log (validation visibility).
def test_dot_publish_emits_stage_log(tmp_path, caplog):
    import logging

    pol, out = _pollen(tmp_path)
    rd = _dot_track_dir(out, "dot1", "20260204", "abc12345_120500")
    with caplog.at_level(logging.INFO, logger="bugcam.edge26.result_publish"):
        publish_result_dir(pol, rd, "flick1", ["dot1"])
    assert any("published" in r.message and "dot" in r.message for r in caplog.records)


# Criterion 3 — empty/low-confidence track: nothing enqueued, dir gone.
def test_dot_empty_track_deleted_no_upload(tmp_path):
    pol, out = _pollen(tmp_path)
    rd = _dot_track_dir(out, "dot1", "20260204", "abc12345_120500", tracks_present=False)

    n = publish_result_dir(pol, rd, "flick1", ["dot1"])

    assert n == 0
    assert pol.store.pending_count() == 0
    assert not rd.exists(), "empty per-track DOT dir must be deleted, not left behind"


# Criterion 4 — videos. Detection (subprocess) enqueues a "video" task to the disk
# queue; the main-process worker ships it as kind="video" and drops the local copy.
def test_dot_media_enqueues_video_task(tmp_path):
    from bugcam.edge26.pipeline import Pipeline
    from bugcam.edge26.queue import ClassificationQueue

    _, out = _pollen(tmp_path)
    pipe = Pipeline.__new__(Pipeline)
    pipe.results_dir = out
    pipe.dot_ids = ["dot1"]
    pipe.classification_queue = ClassificationQueue(tmp_path / "pending")
    dot_in = tmp_path / "input" / "dot1_20260204"
    (dot_in / "videos").mkdir(parents=True)
    (dot_in / "videos" / "DOT_20260204_120453.mp4").write_bytes(b"vid")

    pipe._process_dot_media(dot_in)

    entries = pipe.classification_queue.get_pending_entries()
    assert len(entries) == 1
    _, entry = entries[0]
    assert entry.entry_type == "video"
    assert entry.source_device == "dot1"
    # the copied video (in the day dir) is what the worker will ship
    assert Path(entry.track_dir).name == "DOT_20260204_120453.mp4"
    assert Path(entry.track_dir).exists()


def test_publish_dot_video_ships_and_drops_copy(tmp_path):
    from bugcam.edge26.pipeline import Pipeline
    from bugcam.edge26.queue import QueueEntry

    pol, out = _pollen(tmp_path)
    pipe = Pipeline.__new__(Pipeline)
    # The worker runs in the main process, where on_video_ready/Pollen are wired.
    pipe._on_video_ready = lambda v, dev: pol.enqueue_set([v], device=dev, kind="video")
    vpath = out / "dot1" / "20260204" / "videos" / "DOT_20260204_120453.mp4"
    vpath.parent.mkdir(parents=True)
    vpath.write_bytes(b"vid")
    entry = QueueEntry(entry_type="video", source_device="dot1", date="20260204",
                       time=None, track_id=vpath.stem, track_dir=str(vpath))

    pipe._publish_dot_video(entry)

    rows = pol.store.claim_pending()
    assert len(rows) == 1
    assert rows[0].kind == "video"
    assert rows[0].metadata.get("device") == "dot1"
    assert rows[0].s3_key.endswith("/dot1/20260204/videos/DOT_20260204_120453.mp4")
    assert not vpath.exists()                      # producer copy dropped after staging
    assert Path(rows[0].staging_path).exists()     # Pollen holds the staged hardlink


def test_publish_dot_video_kept_and_raises_when_no_owner(tmp_path):
    """No upload owner -> raise (so the queue retries) and never lose the video."""
    import pytest
    from bugcam.edge26.pipeline import Pipeline
    from bugcam.edge26.queue import QueueEntry

    _, out = _pollen(tmp_path)
    pipe = Pipeline.__new__(Pipeline)
    pipe._on_video_ready = None
    vpath = out / "dot1" / "20260204" / "videos" / "v.mp4"
    vpath.parent.mkdir(parents=True)
    vpath.write_bytes(b"vid")
    entry = QueueEntry(entry_type="video", source_device="dot1", date="20260204",
                       time=None, track_id="v", track_dir=str(vpath))

    with pytest.raises(RuntimeError):
        pipe._publish_dot_video(entry)
    assert vpath.exists()  # not lost; will be retried
