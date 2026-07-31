"""Names and semantics of the sidecar files that coordinate a result directory.

A result dir is one terminal unit of pipeline output: a FLIK video chunk
(``<results>/<flick_id>/<YYYYMMDD_HHMMSS...>/``) or a DOT per-track dir
(``<results>/<dot_id>/<YYYYMMDD>/<track>_<HHMMSS>/``). Detection, the
classification worker, the stale sweep, and the publisher hand a dir to each
other through files rather than in-process state, since detection may run in
a subprocess and any participant can die and restart mid-handoff. This module
is the single source of truth for those filenames. It covers only the sidecar
protocol inside a result dir (plus the ``.last_recording`` session marker) --
not the input/pending-queue/output directory trees.

Sidecar files
=============

``results.json`` (RESULTS)
    The payload, written via ``results.json.tmp`` + atomic rename so readers
    never see a torn file. Detection writes it for zero-track dirs; the
    classification worker rewrites it after each classified track.
``.detection.json`` (DETECTION_META)
    Per-video detection metadata, FLIK only. Written by detection, read by
    the classification worker, removed at finalization.
``.expected_tracks`` (EXPECTED_TRACKS)
    Count of classification entries detection will enqueue (always 1 for
    DOT). Must exist before the first entry is enqueued, or a fast
    classification's completion is lost until the stale sweep catches it.
    Removed at finalization.
``.completed_tracks`` (COMPLETED_TRACKS)
    Count of entries resolved so far. Written only by the main-process
    worker, one increment per entry; an unreadable counter is skipped rather
    than reset, so it can't walk backwards. Removed at finalization.
``.done`` (DONE)
    Finalization marker -- existence alone means "ready to publish." The
    ``key=value`` body is diagnostic only (read by the result-health audit);
    nothing branches on it. Never removed; publishing deletes the whole dir.
``.uploaded`` / ``.archived`` / ``.archived-aux`` (LEGACY_UPLOAD_MARKERS)
    Dead names from a retired upload design. Still excluded from publishing
    so old dirs left on disk don't ship them.
``.last_recording`` (LAST_RECORDING)
    Not a result-dir sidecar -- lives in input storage, carries the last
    video of a recording session so the tracker reset survives a restart.
    Written by the recorder owner; cleared after the boundary video only by
    whichever instance runs detection (subprocess mode: the parent must not
    consume it out from under the child).

Lifecycle
=========

Detection creates the dir and fills the payload. Tracked dirs get
``.detection.json`` (FLIK only) then ``.expected_tracks`` before any
classification entry is enqueued; zero-track FLIK dirs skip classification
entirely (empty ``results.json`` -> ``.done`` -> a ``result`` publish entry,
since there's no track entry to trigger it otherwise). The classification
worker resolves each entry, rewrites ``results.json``, and increments
``.completed_tracks``; once completed >= expected it finalizes -- writes
``.done``, drops the progress markers, runs the result-health audit, and
publishes. Publish (result_publish) ships every file not in UPLOAD_EXCLUDE
as one atomic upload set and deletes the dir; empty results delete without
uploading. It doesn't re-check ``.done`` but does require ``results.json``
and a resolvable owning device, so a dir missing either is never shipped --
the orphan sweep retries it indefinitely instead.

Crash recovery
===============

Any participant can die mid-handoff. The stale sweep (run from the
classification worker, since it holds the upload callback in subprocess
mode) re-derives state from markers plus mtimes: a stale ``.done`` gets its
publish retried; ``results.json`` with no ``.expected_tracks``/``.done`` past
the stale threshold gets force-finalized (``swept=stale``); the same with
``.expected_tracks`` also present, unless completions are still short of
expected while entries remain queued; a dir with neither and no real content
past the empty threshold gets deleted. The startup inventory reads the same
markers to rebuild in-memory state after a restart.

With ``detection_in_subprocess``, the child owns everything through
``.expected_tracks``; the parent's classification worker owns everything
after the queue.
"""
from __future__ import annotations

RESULTS = "results.json"
RESULTS_TMP = "results.json.tmp"
DETECTION_META = ".detection.json"
EXPECTED_TRACKS = ".expected_tracks"
COMPLETED_TRACKS = ".completed_tracks"
DONE = ".done"
LAST_RECORDING = ".last_recording"
LEGACY_UPLOAD_MARKERS = frozenset({".uploaded", ".archived", ".archived-aux"})

# Two derived sets, distinct on purpose -- they answer different questions.

# Which names must never ship when a finalized dir is published? Includes the
# legacy markers so dirs left by old versions don't upload them; not
# RESULTS_TMP (normally gone via the atomic rename -- a crash leftover would
# ship alongside the payload, harmlessly).
UPLOAD_EXCLUDE = frozenset({DONE, DETECTION_META, EXPECTED_TRACKS, COMPLETED_TRACKS}) | LEGACY_UPLOAD_MARKERS

# Which names do not make an unfinished dir worth keeping? Used by the sweep
# to decide an abandoned dir is empty enough to delete. Includes RESULTS_TMP
# (a torn write is not content); leaves the legacy markers out, so a dir
# holding only those is conservatively kept.
SWEEP_NON_CONTENT = frozenset({DONE, DETECTION_META, EXPECTED_TRACKS, COMPLETED_TRACKS, RESULTS_TMP})
