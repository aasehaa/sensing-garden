"""Tests for the sg-unprocessed upload helper."""
from pathlib import Path

import pytest

from bugcam.unprocessed_upload import (
    UPLOADED_SUFFIX,
    bucket_exists,
    format_bytes,
    format_upload_summary,
    iter_pending_files,
    object_exists,
    parse_size,
    upload_pending_files,
    verify_uploaded_files,
)


class FakeS3Client:
    def __init__(self, buckets: set[str] | None = None, objects: set[str] | None = None):
        self.buckets = set(buckets or set())
        self.objects: set[str] = set(objects or set())
        self.uploads: list[tuple[str, str, str]] = []
        self.fail_keys: set[str] = set()

    def head_bucket(self, Bucket: str):
        if Bucket not in self.buckets:
            error = Exception("Not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {}

    def head_object(self, Bucket: str, Key: str):
        if Key not in self.objects:
            error = Exception("Not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {"Bucket": Bucket, "Key": Key}

    def upload_file(self, Filename: str, Bucket: str, Key: str):
        if Key in self.fail_keys:
            raise RuntimeError(f"simulated upload failure for {Key}")
        self.uploads.append((Filename, Bucket, Key))
        self.objects.add(Key)


def _make_files(tmp_path: Path, names_and_sizes: dict[str, int]) -> None:
    for name, size in names_and_sizes.items():
        (tmp_path / name).write_bytes(b"x" * size)


# --- parse_size / format_bytes -------------------------------------------------


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


# --- iter_pending_files ---------------------------------------------------------


def test_iter_pending_files_skips_already_uploaded_and_dirs(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 1, "b.jpg.uploaded": 1})
    (tmp_path / "subdir").mkdir()

    pending = iter_pending_files(tmp_path)

    assert [p.name for p in pending] == ["a.jpg"]


# --- bucket_exists / object_exists ----------------------------------------------


def test_bucket_exists() -> None:
    client = FakeS3Client(buckets={"sg-unprocessed"})
    assert bucket_exists(client, "sg-unprocessed") is True
    assert bucket_exists(client, "does-not-exist") is False


def test_object_exists() -> None:
    client = FakeS3Client(objects={"a.jpg"})
    assert object_exists(client, "sg-unprocessed", "a.jpg") is True
    assert object_exists(client, "sg-unprocessed", "missing.jpg") is False


# --- upload_pending_files --------------------------------------------------------


def test_upload_pending_files_dry_run_does_not_upload_or_rename(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 20})
    client = FakeS3Client(buckets={"sg-unprocessed"})

    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", dry_run=True, s3_client=client)

    assert {r.status for r in results} == {"planned"}
    assert client.uploads == []
    assert (tmp_path / "a.jpg").exists()
    assert not (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_uploads_and_renames(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 20})
    client = FakeS3Client(buckets={"sg-unprocessed"})

    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", s3_client=client)

    assert {r.status for r in results} == {"uploaded"}
    assert len(client.uploads) == 2
    assert not (tmp_path / "a.jpg").exists()
    assert (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()
    assert (tmp_path / f"b.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_reruns_skip_marked_files(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"})

    upload_pending_files(tmp_path, bucket="sg-unprocessed", s3_client=client)
    second_run = upload_pending_files(tmp_path, bucket="sg-unprocessed", s3_client=client)

    assert second_run == []
    assert len(client.uploads) == 1


def test_upload_pending_files_respects_max_bytes_budget(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 10, "c.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"})

    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", max_bytes=15, s3_client=client)
    by_name = {r.path.name: r.status for r in results}

    # sorted order: a.jpg (10, fits) then b.jpg (would bring total to 20 > 15) -> stop
    assert by_name["a.jpg"] == "uploaded"
    assert by_name["b.jpg"] == "skipped_budget"
    assert by_name["c.jpg"] == "skipped_budget"
    assert len(client.uploads) == 1
    assert (tmp_path / f"a.jpg{UPLOADED_SUFFIX}").exists()
    assert (tmp_path / "b.jpg").exists()


def test_upload_pending_files_zero_budget_uploads_nothing(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"})

    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", max_bytes=0, s3_client=client)

    assert results[0].status == "skipped_budget"
    assert client.uploads == []


def test_upload_pending_files_continues_after_failure(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10, "b.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"})
    client.fail_keys = {"a.jpg"}

    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", s3_client=client)
    by_name = {r.path.name: r.status for r in results}

    assert by_name["a.jpg"] == "failed"
    assert by_name["b.jpg"] == "uploaded"
    assert (tmp_path / "a.jpg").exists()  # left untouched for a retry
    assert (tmp_path / f"b.jpg{UPLOADED_SUFFIX}").exists()


def test_upload_pending_files_requires_client_unless_dry_run(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    with pytest.raises(ValueError):
        upload_pending_files(tmp_path, bucket="sg-unprocessed")


# --- verify_uploaded_files -------------------------------------------------------


def test_verify_uploaded_files_reports_missing(tmp_path: Path) -> None:
    # both are marked .uploaded locally, but only a.jpg actually made it to S3
    _make_files(tmp_path, {f"a.jpg{UPLOADED_SUFFIX}": 10, f"b.jpg{UPLOADED_SUFFIX}": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"}, objects={"a.jpg"})

    missing = verify_uploaded_files(tmp_path, bucket="sg-unprocessed", s3_client=client)

    assert [p.name for p in missing] == [f"b.jpg{UPLOADED_SUFFIX}"]


def test_verify_uploaded_files_ignores_pending_files(tmp_path: Path) -> None:
    # c.jpg was never uploaded (e.g. left behind by a max_bytes budget) --
    # it isn't marked .uploaded, so it's not expected in S3 and isn't an error.
    _make_files(tmp_path, {f"a.jpg{UPLOADED_SUFFIX}": 10, "c.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"}, objects={"a.jpg"})

    missing = verify_uploaded_files(tmp_path, bucket="sg-unprocessed", s3_client=client)

    assert missing == []


def test_verify_uploaded_files_empty_dir_is_fully_synced(tmp_path: Path) -> None:
    client = FakeS3Client(buckets={"sg-unprocessed"})
    assert verify_uploaded_files(tmp_path, bucket="sg-unprocessed", s3_client=client) == []


# --- format_upload_summary -------------------------------------------------------


def test_format_upload_summary_mentions_bucket_and_counts(tmp_path: Path) -> None:
    _make_files(tmp_path, {"a.jpg": 10})
    client = FakeS3Client(buckets={"sg-unprocessed"})
    results = upload_pending_files(tmp_path, bucket="sg-unprocessed", s3_client=client)

    summary = format_upload_summary("sg-unprocessed", results)

    assert "sg-unprocessed" in summary
    assert "uploaded: 1 file" in summary
