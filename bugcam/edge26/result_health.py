"""Sanity checks on a result dir at the moment .done is written.

A .done marker means "classification complete, ready to publish", but several
paths can finalize a directory without a healthy result behind it: tracks that
fail classification gracefully still count toward .completed_tracks, the stale
sweep force-finalizes stuck directories, and a crash can leave results.json
missing or truncated. The audit re-derives what a healthy directory must look
like from the files themselves and reports every mismatch, so bad results show
up as errors in the shipped logs instead of publishing silently.

Stdlib-only on purpose (plus the constants-only sidecars module): auditing
must never depend on the camera/inference stack.
"""
from __future__ import annotations

import json
from pathlib import Path

from bugcam.edge26.sidecars import DONE, RESULTS


def _parse_done(done_path: Path) -> dict[str, str]:
    """Parse the key=value lines of a .done marker; {} if unreadable."""
    try:
        text = done_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _int_field(fields: dict[str, str], key: str) -> int | None:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        return None


def _has_matching_crop_dir(track_id: str, crop_dir_names: list[str]) -> bool:
    # FLIK stores crops under the first UUID segment of the track id
    # (crops/<uuid8>/ for track "<uuid>_<HHMMSS>"); DOT under the full dir name
    # (crops/<track_id>_<HHMMSS>/ for a bare track id). Prefix matching in both
    # directions covers both layouts plus _deduplicate_track_id suffixes.
    base = track_id.split("-")[0]
    return any(
        track_id.startswith(name) or name.startswith(track_id) or name.startswith(base)
        for name in crop_dir_names
    )


def audit_result_dir(output_dir: Path) -> list[str]:
    """Return the ways a just-finalized (.done) result dir looks unhealthy.

    Empty list = healthy. Pure inspection: reads the directory, changes nothing.
    """
    output_dir = Path(output_dir)
    problems: list[str] = []

    done_fields = _parse_done(output_dir / DONE)
    expected = _int_field(done_fields, "expected")
    # The normal path writes classified=, the stale sweep writes completed=.
    classified = _int_field(done_fields, "classified")
    if classified is None:
        classified = _int_field(done_fields, "completed")

    if done_fields.get("swept") == "stale":
        detail = f" ({classified}/{expected} classified)" if expected is not None else ""
        problems.append("finalized by the stale sweep, not by classification" + detail)
    elif expected is not None and classified is not None and classified > expected:
        problems.append(
            f"classified count {classified} exceeds expected {expected} (completion double-counted)"
        )

    results_json = output_dir / RESULTS
    if not results_json.exists():
        problems.append("marked done but results.json is missing (no classification produced output)")
        return problems

    try:
        payload = json.loads(results_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"results.json unreadable: {exc}")
        return problems

    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        problems.append("results.json has no 'tracks' list")
        return problems

    if expected is not None and len(tracks) != expected:
        if len(tracks) < expected:
            problems.append(
                f"results.json holds {len(tracks)} track(s) but {expected} were expected "
                f"({expected - len(tracks)} classification(s) failed or were lost)"
            )
        else:
            problems.append(
                f"results.json holds {len(tracks)} track(s) but only {expected} were expected"
            )

    missing_prediction = sum(
        1 for t in tracks if isinstance(t, dict) and not t.get("final_prediction")
    )
    if missing_prediction:
        problems.append(f"{missing_prediction} of {len(tracks)} track entries lack a final_prediction")

    summary = payload.get("summary") or {}
    confirmed = summary.get("confirmed_tracks")
    if confirmed is not None and confirmed != len(tracks):
        problems.append(
            f"summary claims confirmed_tracks={confirmed} but {len(tracks)} track(s) are recorded"
        )

    if tracks:
        crops_root = output_dir / "crops"
        crop_dir_names = (
            [d.name for d in crops_root.iterdir() if d.is_dir()] if crops_root.is_dir() else []
        )
        if not crop_dir_names:
            problems.append("tracks recorded but no crop folders on disk")
        else:
            orphaned = [
                t["track_id"]
                for t in tracks
                if isinstance(t, dict)
                and t.get("track_id")
                and not _has_matching_crop_dir(t["track_id"], crop_dir_names)
            ]
            if orphaned:
                problems.append(f"no crop folder for track(s): {', '.join(orphaned)}")

    return problems
