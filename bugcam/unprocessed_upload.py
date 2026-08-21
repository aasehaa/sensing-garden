"""Upload local files to the sg-unprocessed S3 bucket, marking each as uploaded.

Dev-only utility. This module is not part of the device runtime path.

Spec
----
Given a source directory, upload every regular file in it (non-recursive) that
isn't already marked uploaded to the target bucket, using the file's own name
as the S3 key. On a successful upload, the local file is renamed by appending
the ``.uploaded`` suffix so a re-run skips it without needing a manifest.

An optional byte budget (``max_bytes``) caps how much data a single run will
push: files are processed in sorted-name order, and as soon as the next file
would push the cumulative total over the budget, that file and everything
after it are left alone (reported as "skipped_budget") rather than uploaded.

A failed upload (S3 error) leaves the local file untouched and is reported as
"failed"; a failed post-upload rename (e.g. name collision) leaves the object
in S3 -- which is harmless, re-running will just re-upload the same key -- and
is reported as "uploaded_unmarked". Either way, one failure does not stop the
rest of the batch.

A separate end-of-run check (``verify_uploaded_files``) confirms that every
file already marked ``.uploaded`` locally -- from this run or a prior one --
really does have a matching S3 object. It only looks at ``.uploaded`` files:
files still pending because a budget cut the run short are expected to be
missing from S3 and are not treated as an error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_UNPROCESSED_BUCKET = "sg-unprocessed"
UPLOADED_SUFFIX = ".uploaded"

_SIZE_UNITS = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def parse_size(value: str) -> int:
    """Parse a human size like ``"500MB"``, ``"2GiB"``, or a plain byte count.

    Raises ValueError for anything that isn't a non-negative number optionally
    followed by one of the units in _SIZE_UNITS.
    """
    text = value.strip().upper()
    if not text:
        raise ValueError("size must not be empty")

    split = 0
    while split < len(text) and (text[split].isdigit() or text[split] == "."):
        split += 1
    number_part, unit_part = text[:split], text[split:].strip()
    if not number_part:
        raise ValueError(f"invalid size: {value!r}")

    number = float(number_part)
    if number < 0:
        raise ValueError(f"size must not be negative: {value!r}")

    unit = unit_part or "B"
    if unit not in _SIZE_UNITS:
        raise ValueError(f"unknown size unit {unit_part!r} in {value!r}")

    return int(number * _SIZE_UNITS[unit])


def format_bytes(n: int) -> str:
    """Render a byte count as a compact human string, e.g. 1536 -> '1.5KB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1000
    return f"{n}B"  # pragma: no cover - unreachable, satisfies type checkers


def _normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().strip("/")
    return f"{prefix}/" if prefix else ""


def iter_pending_files(source_dir: Path) -> list[Path]:
    """Return regular files in source_dir not already marked uploaded, sorted by name."""
    return sorted(
        (p for p in source_dir.iterdir() if p.is_file() and not p.name.endswith(UPLOADED_SUFFIX)),
        key=lambda p: p.name,
    )


def bucket_exists(s3_client: Any, bucket: str) -> bool:
    """Return True if the bucket exists and is accessible."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        return True
    except Exception as exc:
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code in {"404", "403", "NoSuchBucket"}:
            return False
        raise


def object_exists(s3_client: Any, bucket: str, key: str) -> bool:
    """Return True if the object already exists in the bucket."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


@dataclass(frozen=True)
class UploadResult:
    """Outcome of processing one local file."""

    path: Path
    key: str
    size: int
    status: str  # "uploaded" | "uploaded_unmarked" | "skipped_budget" | "failed" | "planned"
    error: str | None = None


def upload_pending_files(
    source_dir: Path,
    *,
    bucket: str = DEFAULT_UNPROCESSED_BUCKET,
    prefix: str = "",
    max_bytes: int | None = None,
    dry_run: bool = False,
    s3_client: Any = None,
) -> list[UploadResult]:
    """Upload pending files from source_dir, renaming each on success.

    Files are processed in sorted-name order. Once the next file would push
    cumulative uploaded bytes past max_bytes (if set), it and every file after
    it are reported as "skipped_budget" and left untouched.
    """
    if not dry_run and s3_client is None:
        raise ValueError("s3_client is required unless dry_run=True")

    key_prefix = _normalize_prefix(prefix)
    results: list[UploadResult] = []
    total = 0
    budget_exhausted = False

    for path in iter_pending_files(source_dir):
        size = path.stat().st_size
        key = f"{key_prefix}{path.name}"

        if budget_exhausted or (max_bytes is not None and total + size > max_bytes):
            budget_exhausted = True
            results.append(UploadResult(path, key, size, "skipped_budget"))
            continue

        if dry_run:
            total += size
            results.append(UploadResult(path, key, size, "planned"))
            continue

        try:
            s3_client.upload_file(str(path), bucket, key)
        except Exception as exc:
            results.append(UploadResult(path, key, size, "failed", error=str(exc)))
            continue

        total += size
        try:
            path.rename(path.with_name(path.name + UPLOADED_SUFFIX))
        except OSError as exc:
            results.append(UploadResult(path, key, size, "uploaded_unmarked", error=str(exc)))
            continue

        results.append(UploadResult(path, key, size, "uploaded"))

    return results


def verify_uploaded_files(
    source_dir: Path,
    *,
    bucket: str,
    prefix: str = "",
    s3_client: Any,
) -> list[Path]:
    """Confirm every file already marked `.uploaded` has a matching S3 object.

    Only files ending in UPLOADED_SUFFIX are checked -- files still pending
    (e.g. left behind by a max_bytes budget) are not expected to be in S3 yet
    and are not an error. Returns the local `.uploaded` paths that have NO
    corresponding object in the bucket -- an empty list means everything this
    run (or a prior run) marked as uploaded is actually present. This always
    re-queries S3 rather than reusing in-memory upload results, since the goal
    is an independent confirmation.
    """
    key_prefix = _normalize_prefix(prefix)
    missing: list[Path] = []
    for path in sorted(source_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or not path.name.endswith(UPLOADED_SUFFIX):
            continue
        original_name = path.name[: -len(UPLOADED_SUFFIX)]
        key = f"{key_prefix}{original_name}"
        if not object_exists(s3_client, bucket, key):
            missing.append(path)
    return missing


def format_upload_summary(bucket: str, results: Iterable[UploadResult]) -> str:
    """Return a human-readable summary of an upload_pending_files() run."""
    results = list(results)
    by_status: dict[str, list[UploadResult]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    lines = [f"Bucket: {bucket}"]
    uploaded = by_status.get("uploaded", []) + by_status.get("uploaded_unmarked", [])
    if uploaded:
        total_bytes = sum(r.size for r in uploaded)
        lines.append(f"  uploaded: {len(uploaded)} file(s), {format_bytes(total_bytes)}")
    if "planned" in by_status:
        planned = by_status["planned"]
        lines.append(f"  planned (dry-run): {len(planned)} file(s), {format_bytes(sum(r.size for r in planned))}")
    if "uploaded_unmarked" in by_status:
        lines.append(
            f"  uploaded but NOT renamed locally (will re-upload next run): "
            f"{len(by_status['uploaded_unmarked'])} file(s)"
        )
    if "skipped_budget" in by_status:
        lines.append(f"  skipped (over budget): {len(by_status['skipped_budget'])} file(s)")
    if "failed" in by_status:
        lines.append(f"  failed: {len(by_status['failed'])} file(s)")
        for r in by_status["failed"]:
            lines.append(f"    - {r.path.name}: {r.error}")
    return "\n".join(lines)
