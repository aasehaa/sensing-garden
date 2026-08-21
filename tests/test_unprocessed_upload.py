"""Tests for the raw-file upload helper (presigned-URL / api_key auth, no AWS creds)."""
import time
from pathlib import Path

import pytest

from bugcam.unprocessed_upload import (
    DEFAULT_KEY_PREFIX,
    UPLOADED_SUFFIX,
    build_object_key,
    format_bytes,
    format_upload_summary,
    iter_pending_files,
    parse_size,
    upload_pending_files,
)


class FakePresigner:
    """Stands in for bugcam.pollen.presign.Presigner."""

    def __init__(self, fail_keys: set[str] | None = None):
        self.fail_keys = set(fail_keys or set())
        self.requested_keys: list[str] = []

    def put_url(self, s3_key: str) -> str:
        self.requested_keys.append(s3_key)
        if s3_key in self.fail_keys:
            raise RuntimeError(f"simulated presign failure for {s3_key}")
        return f"https://fake-upload.example/{s3_key}"


def make_put_file(fail_urls: set[str] | None = None):
    """Fake HTTP PUT: records (url, path) pairs instead of hitting the network."""
    calls: list[tuple[str, Path]] = []
    fail_urls = set(fail_urls or set())

    def put_file(url: str, path: Path) -> None:
        if url in fail_urls:
            raise RuntimeError(f"simulated PUT failure for {url}")
        calls.append((url, path))

    put_file.calls = calls  # type: ignore[attr-defined]
    return put_file


def _make_files(tmp_path: Path, names_and_sizes: dict[str, int]) -> None:
    for name, size in names_and_sizes.items():
        (tmp_path / name).write_bytes(b"x" * size)


# --- parse_size / format_bytes (unchanged helpers) -------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("100", 100),
        ("500MB", 500_000_000),
        ("2GB", 2_000_000_000),
        ("1KiB", 1024),
        ("1.5MB", 1_500_000),
        ("10b", 10),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "MB", "-5MB", "5XB"])
def test_parse_size_rejects_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        parse_size(text)


def test_format_bytes_roundish() -> None:
    assert format_bytes(0) == "0B"
    assert format_bytes(1500) == "1.5KB"


# --- build_object_key -------------------------------------------------------------


def test_build_object_key_default_prefix() -> None:
    assert build_object_key("SGSCA11", "a.jpg") == "v1/SGSCA11/raw/a.jpg"


def test_build_object_key_custom_prefix() -> None:
    assert build_object_key("SGSCA11", "a.jpg", key_prefix="raw-test") == "v1/SGSCA11/raw-test/a.jpg"


def test_build_object_key_empty_prefix_omits_segment() -> None:
    assert build_object_key("SGSCA11", "a.jpg", key_prefix="") == "v1/SGSCA11/a.jpg"


# --- iter_pending_files -------------------------------------------------------------


def test_iter_pending_files_skips_already_uploaded_and_dirs(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 1, "b.jpg.uploaded": 1})
    (tmp_path / "subdir").mkdir()

    pending = iter_pending_files(tmp_path)

    assert [p.name for p in pending] == ["a.jpg"]


def test_iter_pending_files_min_age_excludes_recent_files(tmp_path: Path) -> None:
    _make_files(tmp_path, {"old.jpg": 1, "new.jpg": 1})
    old_time = time.time() - 3600
    import os

    os.utime(tmp_path / "old.jpg", (old_time, old_time))

    pending = iter_pending_files(tmp_path, min_age_seconds=300)

    assert [p.name for p in pending] == ["old.jpg"]


def test_iter_pending_files_default_min_age_is_zero_includes_everything(tmp_path: Path) -> None:
    _make_files(tmp_path, {"brand_new.jpg": 1})
    assert [p.name for p in iter_pending_files(tmp_path)] == ["brand_new.jpg"]


# --- upload_pending_files --------------------------------------------------------


def test_upload_pending_files_dry_run_does_not_upload_or_rename(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 20})

    results = upload_pending_files(tmp_path, device_id="SGSCA11", dry_run=True)

    assert {r.status for r in results} == {"planned"}
    assert (tmp_path / "a.jpg").exists()
    assert not (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_uploads_and_renames(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 20})
    presigner = FakePresigner()
    put_file = make_put_file()

    results = upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=put_file)

    assert {r.status for r in results} == {"uploaded"}
    assert len(put_file.calls) == 2  # type: ignore[attr-defined]
    assert presigner.requested_keys == ["v1/SGSCA11/raw/a.jpg", "v1/SGSCA11/raw/b.jpg"]
    assert not (tmp_path / "a.jpg").exists()
    assert (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()
    assert (tmp_path / f"b.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_uses_custom_key_prefix(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    presigner = FakePresigner()

    upload_pending_files(
        tmp_path, presigner=presigner, device_id="SGSCA11", key_prefix="raw-test", put_file=make_put_file()
    )

    assert presigner.requested_keys == ["v1/SGSCA11/raw-test/a.jpg"]


def test_upload_pending_files_reruns_skip_marked_files(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    presigner = FakePresigner()
    put_file = make_put_file()

    upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=put_file)
    second_run = upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=put_file)

    assert second_run == []
    assert len(put_file.calls) == 1  # type: ignore[attr-defined]


def test_upload_pending_files_respects_max_bytes_budget(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 10, "c.jpg": 10})
    presigner = FakePresigner()

    results = upload_pending_files(
        tmp_path, presigner=presigner, device_id="SGSCA11", max_bytes=15, put_file=make_put_file()
    )
    by_name = {r.path.name: r.status for r in results}

    assert by_name["a.jpg"] == "uploaded"
    assert by_name["b.jpg"] == "skipped_budget"
    assert by_name["c.jpg"] == "skipped_budget"
    assert (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()
    assert (tmp_path / "b.jpg").exists()


def test_upload_pending_files_zero_budget_uploads_nothing(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    presigner = FakePresigner()

    results = upload_pending_files(
        tmp_path, presigner=presigner, device_id="SGSCA11", max_bytes=0, put_file=make_put_file()
    )

    assert results[0].status == "skipped_budget"


def test_upload_pending_files_presign_failure_leaves_file_untouched(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 10})
    presigner = FakePresigner(fail_keys={"v1/SGSCA11/raw/a.jpg"})
    put_file = make_put_file()

    results = upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=put_file)
    by_name = {r.path.name: r.status for r in results}

    assert by_name["a.jpg"] == "failed"
    assert by_name["b.jpg"] == "uploaded"
    assert (tmp_path / "a.jpg").exists()  # untouched, left for a retry
    assert (tmp_path / f"b.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_put_failure_leaves_file_untouched(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    presigner = FakePresigner()
    put_file = make_put_file(fail_urls={"https://fake-upload.example/v1/SGSCA11/raw/a.jpg"})

    results = upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=put_file)

    assert results[0].status == "failed"
    assert (tmp_path / "a.jpg").exists()  # nothing deleted or renamed on failure


def test_upload_pending_files_continues_after_failure(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 10})
    presigner = FakePresigner(fail_keys={"v1/SGSCA11/raw/a.jpg"})

    results = upload_pending_files(
        tmp_path, presigner=presigner, device_id="SGSCA11", put_file=make_put_file()
    )
    by_name = {r.path.name: r.status for r in results}

    assert by_name["a.jpg"] == "failed"
    assert by_name["b.jpg"] == "uploaded"


def test_upload_pending_files_requires_presigner_unless_dry_run(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    with pytest.raises(ValueError):
        upload_pending_files(tmp_path, device_id="SGSCA11")


def test_upload_pending_files_min_age_seconds_defers_recent_files(tmp_path: Path) -> None:
    _make_files(tmp_path, {"old.jpg": 10, "new.jpg": 10})
    old_time = time.time() - 3600
    import os

    os.utime(tmp_path / "old.jpg", (old_time, old_time))
    presigner = FakePresigner()

    results = upload_pending_files(
        tmp_path, presigner=presigner, device_id="SGSCA11", min_age_seconds=300, put_file=make_put_file()
    )

    assert [r.path.name for r in results if r.status == "uploaded"] == ["old.jpg"]
    assert (tmp_path / "new.jpg").exists()  # untouched: too fresh, might still be in-flight


# --- format_upload_summary -------------------------------------------------------


def test_format_upload_summary_mentions_device_and_counts(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    presigner = FakePresigner()
    results = upload_pending_files(tmp_path, presigner=presigner, device_id="SGSCA11", put_file=make_put_file())

    summary = format_upload_summary("SGSCA11", DEFAULT_KEY_PREFIX, results)

    assert "SGSCA11" in summary
    assert "raw" in summary
    assert "uploaded: 1 file" in summary
