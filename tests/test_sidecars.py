"""Pin the on-disk sidecar protocol names.

The constants in bugcam.edge26.sidecars ARE the on-disk protocol: devices in
the field have dirs holding these exact filenames, so a rename in the module
would silently orphan them. These tests pin the literal values so any change
has to be made here too, deliberately.
"""
from bugcam.edge26 import sidecars


def test_sidecar_filenames_are_pinned():
    assert sidecars.RESULTS == "results.json"
    assert sidecars.RESULTS_TMP == "results.json.tmp"
    assert sidecars.DETECTION_META == ".detection.json"
    assert sidecars.EXPECTED_TRACKS == ".expected_tracks"
    assert sidecars.COMPLETED_TRACKS == ".completed_tracks"
    assert sidecars.DONE == ".done"
    assert sidecars.LAST_RECORDING == ".last_recording"
    assert sidecars.LEGACY_UPLOAD_MARKERS == {".uploaded", ".archived", ".archived-aux"}


def test_upload_exclude_pinned():
    # Everything coordination-only plus the legacy markers; never the payload
    # or its temp file (a crash leftover ships harmlessly).
    assert sidecars.UPLOAD_EXCLUDE == {
        ".done", ".detection.json", ".expected_tracks", ".completed_tracks",
        ".uploaded", ".archived", ".archived-aux",
    }
    assert sidecars.RESULTS not in sidecars.UPLOAD_EXCLUDE
    assert sidecars.RESULTS_TMP not in sidecars.UPLOAD_EXCLUDE


def test_sweep_non_content_pinned():
    # Everything coordination-only plus the torn temp write; legacy markers
    # stay out so a dir holding only those is conservatively kept.
    assert sidecars.SWEEP_NON_CONTENT == {
        ".done", ".detection.json", ".expected_tracks", ".completed_tracks",
        "results.json.tmp",
    }
    assert sidecars.RESULTS not in sidecars.SWEEP_NON_CONTENT
