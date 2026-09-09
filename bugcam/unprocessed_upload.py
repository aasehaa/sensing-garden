"""Upload local raw files via the backend's presigned-URL API, marking each as uploaded.

Dev-only utility. This module is not part of the device runtime path.

Auth
----
Devices in the field carry a sensing-garden ``api_key`` but no AWS credentials, so
this does not talk to S3 directly. It requests a presigned PUT URL from the backend
(the same ``/upload-url`` endpoint and ``Presigner`` Pollen already uses in
production) and PUTs the file to that URL. Uploads land in the production bucket
under ``v1/<device_id>/raw/<filename>`` -- a key shape the backend's existing
per-device scope check already accepts, so no backend changes are needed.

There is no backend endpoint to check whether an object actually exists in S3 via
api_key (no HEAD/GET presign route) -- unlike a raw-boto3 uploader, this cannot
independently verify an upload after the fact. A file counts as uploaded once the
presigned PUT returns success; there is no separate verify step.

Spec
----
Given a source directory, upload every regular file in it (non-recursive) that
isn't already marked uploaded, using the file's own name as the S3 key (under the
device/prefix namespace above). On a successful upload, the local file is renamed
by appending the ``.uploaded`` suffix so a re-run skips it without needing a
manifest.

An optional ``min_age_seconds`` defers any file younger than that -- it may still
be mid-write, or not yet picked up by the live capture/detection pipeline that
also reads this directory; renaming a file out from under that pipeline before it
has had a chance to process it would silently and permanently skip detection for
it. Deferred files are simply skipped this run, not reported as any kind of error.

On a live ``input_dir``, a file discovered as pending can still be consumed
(read + deleted) by the concurrent capture/detection pipeline before we get to
it -- that's expected behavior on a directory we don't own exclusively, not an
error. Such a file is reported as "vanished" and the batch continues; nothing
is lost since the pipeline processed it itself.

An optional byte budget (``max_bytes``) caps how much data a single run will
push: files are processed in sorted-name order, and as soon as the next file
would push the cumulative total over the budget, that file and everything
after it are left alone (reported as "skipped_budget") rather than uploaded.

A failed upload (presign or PUT error) leaves the local file untouched and is
reported as "failed"; a failed post-upload rename (e.g. name collision) leaves
the object already in S3 -- harmless, re-running will just re-upload the same
key -- and is reported as "uploaded_unmarked". Either way, one failure does not
stop the rest of the batch. Nothing is ever deleted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULT_KEY_PREFIX = "raw"
UPLOADED_SUFFIX = ".uploaded"
REQUEST_TIMEOUT_SECONDS = 60

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


def build_object_key(device_id: str, filename: str, *, key_prefix: str = DEFAULT_KEY_PREFIX) -> str:
    """Build the ``v1/<device_id>/...`` S3 key this device is authorized to write to."""
    prefix = key_prefix.strip("/")
    if prefix:
        return f"v1/{device_id}/{prefix}/{filename}"
    return f"v1/{device_id}/{filename}"


def iter_pending_files(source_dir: Path, *, min_age_seconds: float = 0) -> list[Path]:
    """Return regular files in source_dir not already marked uploaded, sorted by name.

    Files younger than ``min_age_seconds`` (by mtime) are excluded -- default 0
    means no age filter, matching plain "everything pending" semantics.
    """
    now = time.time()
    return sorted(
        (
            p
            for p in source_dir.iterdir()
            if p.is_file()
            and not p.name.endswith(UPLOADED_SUFFIX)
            and (min_age_seconds <= 0 or (now - p.stat().st_mtime) >= min_age_seconds)
        ),
        key=lambda p: p.name,
    )


@dataclass(frozen=True)
class UploadResult:
    """Outcome of processing one local file."""

    path: Path
    key: str
    size: int
    status: str  # "uploaded" | "uploaded_unmarked" | "skipped_budget" | "vanished" | "failed" | "planned"
    error: str | None = None


def _default_put_file(url: str, path: Path) -> None:
    import requests

    with path.open("rb") as fh:
        resp = requests.put(url, data=fh, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()


def upload_pending_files(
    source_dir: Path,
    *,
    presigner: Any = None,
    device_id: str,
    key_prefix: str = DEFAULT_KEY_PREFIX,
    max_bytes: Optional[int] = None,
    min_age_seconds: float = 0,
    dry_run: bool = False,
    put_file: Optional[Callable[[str, Path], None]] = None,
    on_result: Optional[Callable[[UploadResult], None]] = None,
    delete_after_upload: bool = False,
) -> list[UploadResult]:
    """Upload pending files from source_dir, renaming each on success.

    Files are processed in sorted-name order. Once the next file would push
    cumulative uploaded bytes past max_bytes (if set), it and every file after
    it are reported as "skipped_budget" and left untouched.

    ``on_result``, if given, is called once per file immediately after each
    result is known -- useful for live progress on a run that may take hours,
    since the return value only arrives once everything is done.

    ``delete_after_upload``, if True, removes the local file on a successful
    upload instead of the default rename-to-``.uploaded``. There is then no
    local record of what's been uploaded, so a re-run cannot skip already-sent
    files by name -- only use this when the source data is expendable once in
    S3 (e.g. deliberately clearing local disk space after independently
    verifying uploads landed). Off by default: renaming is the safe choice.
    """
    if not dry_run and presigner is None:
        raise ValueError("presigner is required unless dry_run=True")
    if not dry_run and put_file is None:
        put_file = _default_put_file

    results: list[UploadResult] = []
    total = 0
    budget_exhausted = False

    def _record(result: UploadResult) -> None:
        results.append(result)
        if on_result is not None:
            on_result(result)

    for path in iter_pending_files(source_dir, min_age_seconds=min_age_seconds):
        # source_dir may be a live pipeline's input_dir: a file listed here can be
        # consumed (and deleted) by that pipeline before we get to it. That's
        # expected on a live directory, not our error -- record it and move on
        # rather than letting the whole batch die on one vanished file.
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            _record(UploadResult(path, "", 0, "vanished"))
            continue
        key = build_object_key(device_id, path.name, key_prefix=key_prefix)

        if budget_exhausted or (max_bytes is not None and total + size > max_bytes):
            budget_exhausted = True
            _record(UploadResult(path, key, size, "skipped_budget"))
            continue

        if dry_run:
            total += size
            _record(UploadResult(path, key, size, "planned"))
            continue

        try:
            upload_url = presigner.put_url(key)
            put_file(upload_url, path)
        except FileNotFoundError:
            # Same race as above, just a later window (vanished between stat()
            # and the PUT itself opening the file).
            _record(UploadResult(path, key, size, "vanished"))
            continue
        except Exception as exc:
            _record(UploadResult(path, key, size, "failed", error=str(exc)))
            continue

        total += size

        if delete_after_upload:
            try:
                path.unlink()
            except FileNotFoundError:
                pass  # already gone (e.g. the live pipeline raced us to it) -- fine, that's the goal
            except OSError as exc:
                _record(UploadResult(path, key, size, "uploaded_unmarked", error=str(exc)))
                continue
            _record(UploadResult(path, key, size, "uploaded"))
            continue

        try:
            path.rename(path.with_name(path.name + UPLOADED_SUFFIX))
        except OSError as exc:
            _record(UploadResult(path, key, size, "uploaded_unmarked", error=str(exc)))
            continue

        _record(UploadResult(path, key, size, "uploaded"))

    return results


def format_upload_summary(device_id: str, key_prefix: str, results: Iterable[UploadResult]) -> str:
    """Return a human-readable summary of an upload_pending_files() run."""
    results = list(results)
    by_status: dict[str, list[UploadResult]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    lines = [f"Device: {device_id}  Destination: {build_object_key(device_id, '', key_prefix=key_prefix)}"]
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
    if "vanished" in by_status:
        lines.append(
            f"  vanished (consumed by the live pipeline before we got to it -- not an error): "
            f"{len(by_status['vanished'])} file(s)"
        )
    if "failed" in by_status:
        lines.append(f"  failed: {len(by_status['failed'])} file(s)")
        for r in by_status["failed"]:
            lines.append(f"    - {r.path.name}: {r.error}")
    lines.append("  (no server-side verification available for this upload path -- success means the PUT returned OK)")
    return "\n".join(lines)
