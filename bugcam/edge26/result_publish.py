"""Produce-site publishing of a finalized result dir to the upload subsystem.

Lives on the produce side (edge26), not in the upload subsystem: it discovers the
finalized dir's files, decides emptiness, and hands the complete set to the spooler
with the owning device via ``enqueue_set`` (atomic, so the set never splits across
archives), then deletes its own dir. The spooler does no scanning or classification.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from bugcam.edge26.sidecars import RESULTS, UPLOAD_EXCLUDE

logger = logging.getLogger(__name__)


def result_is_empty(results_json: Path) -> bool:
    """True when a result has zero tracks and no crop/composite/video media."""
    results_json = Path(results_json)
    try:
        payload = json.loads(results_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("tracks"):
        return False
    results_dir = results_json.parent
    for sub in ("crops", "composites", "videos"):
        media_dir = results_dir / sub
        if media_dir.is_dir() and any(p.is_file() for p in media_dir.rglob("*")):
            return False
    # The FLIK video sample lands top-level (video.mp4), not under videos/;
    # a zero-track dir holding one is exactly the backdrop footage case.
    if any(p.is_file() and p.suffix == ".mp4" for p in results_dir.iterdir()):
        return False
    return True


def _owning_device(results_dir: Path, flick_id: str, dot_ids: list[str]):
    """Resolve the device that owns a finalized result dir from its path.

    A FLIK chunk sits one level under the device (``<flick>/<chunk_ts>/``); a DOT
    per-track dir sits under the date (``<dot>/<YYYYMMDD>/<track>/``). So the device
    is the dir's parent (FLIK) or grandparent (DOT). Returns (device, is_flik, is_dot)."""
    for seg in (results_dir.parent.name, results_dir.parent.parent.name):
        if seg == flick_id:
            return seg, True, False
        if seg in dot_ids:
            return seg, False, True
    return None, False, False


def publish_result_dir(pollen, results_dir, flick_id: str, dot_ids: list[str]) -> int:
    """Enqueue one finalized result dir's files as a single set, then delete the dir.
    Both FLIK chunks and DOT per-track dirs are terminal units: uploaded once and
    removed (the spooler holds staged hardlinks)."""
    results_dir = Path(results_dir)
    results_json = results_dir / RESULTS
    if not results_json.exists():
        logger.error(
            "publish skipped for %s: no results.json (dir has %d other file(s)) -- "
            "nothing to enqueue and this can never resolve on its own; needs manual "
            "cleanup, not retry",
            results_dir, sum(1 for p in results_dir.iterdir() if p.is_file()) if results_dir.is_dir() else 0,
        )
        return 0
    device, is_flik, is_dot = _owning_device(results_dir, flick_id, dot_ids)
    if not (is_flik or is_dot):
        logger.error(
            "publish skipped for %s: owning device not resolved (parent=%s, "
            "grandparent=%s) against configured flick_id=%s dot_ids=%s",
            results_dir, results_dir.parent.name, results_dir.parent.parent.name,
            flick_id, dot_ids,
        )
        return 0
    if result_is_empty(results_json):
        shutil.rmtree(results_dir, ignore_errors=True)  # nothing to ship: terminal, drop it
        logger.info("result %s empty -> dropped, nothing uploaded", results_dir.name)
        return 0
    files = [p for p in sorted(results_dir.rglob("*")) if p.is_file() and p.name not in UPLOAD_EXCLUDE]
    video_count = sum(1 for p in files if p.suffix == ".mp4")
    try:
        total_mb = sum(p.stat().st_size for p in files) / 1e6
    except OSError:
        total_mb = -1.0
    ids = pollen.enqueue_set(files, device=device, kind="result")
    shutil.rmtree(results_dir, ignore_errors=True)  # enqueued -> staged; producer is done
    logger.info(
        "published %s: %d files, %d video(s), %.1f MB (device=%s, %s) -> enqueued; local dir removed",
        results_dir.name, len(ids), video_count, total_mb, device, "flik" if is_flik else "dot",
    )
    return len(ids)
