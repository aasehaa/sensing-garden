#!/usr/bin/env python3
"""Upload local raw files to S3 using this device's own api_key.

Each file is uploaded under its own name, under `v1/<device_id>/raw/...`. On
success the local file is renamed with a `.uploaded` suffix so a re-run skips
it. Authenticates via the backend's presigned-URL API (the same mechanism
Pollen uses in production) with the device's api_key -- no AWS credentials are
read, written, or required.

There is no way for this script to verify an upload landed in S3 after the
fact (the backend has no HEAD/GET presign route for api_key callers) --
success means the presigned PUT returned OK. Nothing is ever deleted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bugcam.config import get_input_storage_dir, load_config
from bugcam.device_config import resolve_flick_id
from bugcam.pollen.presign import Presigner
from bugcam.unprocessed_upload import (
    DEFAULT_KEY_PREFIX,
    UploadResult,
    format_bytes,
    format_upload_summary,
    parse_size,
    upload_pending_files,
)

DEFAULT_MIN_AGE_SECONDS = 300  # margin so we don't race the live capture/detection pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Local directory of files to upload (default: this device's configured input_dir)",
    )
    parser.add_argument(
        "--device-id", default=None, help="Device id used in the S3 key (default: configured flick_id)"
    )
    parser.add_argument("--api-url", default=None, help="Backend API URL (default: from bugcam config)")
    parser.add_argument("--api-key", default=None, help="Backend API key (default: from bugcam config)")
    parser.add_argument(
        "--key-prefix",
        default=DEFAULT_KEY_PREFIX,
        help=f"S3 key prefix under v1/<device-id>/ (default: {DEFAULT_KEY_PREFIX!r})",
    )
    parser.add_argument(
        "--max-bytes",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="Cap on data uploaded this run, e.g. 500MB, 2GiB, or a raw byte count. Unset = no limit.",
    )
    parser.add_argument(
        "--min-age-seconds",
        type=float,
        default=DEFAULT_MIN_AGE_SECONDS,
        help=(
            "Skip files younger than this (mtime), so we don't race the live capture/"
            f"detection pipeline for freshly-written files (default: {DEFAULT_MIN_AGE_SECONDS})"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded, without uploading")
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help=(
            "Delete the local file once uploaded instead of renaming it .uploaded. "
            "No local record is kept, so use only when the source data is expendable "
            "once in S3 -- default is the safer rename-based behavior."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    source_dir = args.source_dir or get_input_storage_dir()
    if not source_dir.is_dir():
        print(f"Error: source dir not found: {source_dir}", file=sys.stderr)
        return 1

    device_id = args.device_id or resolve_flick_id(None)
    if not device_id:
        print("Error: no device id -- pass --device-id or run `bugcam setup`.", file=sys.stderr)
        return 1

    api_url = args.api_url or config.get("api_url")
    api_key = args.api_key or config.get("api_key")
    if not args.dry_run and not (api_url and api_key):
        print("Error: api_url/api_key not set -- pass --api-url/--api-key or run `bugcam setup`.", file=sys.stderr)
        return 1

    presigner = Presigner(api_url, api_key) if (api_url and api_key) else None

    pending_count = len(
        [p for p in source_dir.iterdir() if p.is_file() and not p.name.endswith(".uploaded")]
    )
    counter = {"n": 0}

    def _log_progress(result: UploadResult) -> None:
        counter["n"] += 1
        print(
            f"[{counter['n']}/{pending_count}] {result.status}: {result.path.name} "
            f"({format_bytes(result.size)})" + (f" -- {result.error}" if result.error else ""),
            flush=True,
        )

    results = upload_pending_files(
        source_dir,
        presigner=presigner,
        device_id=device_id,
        key_prefix=args.key_prefix,
        max_bytes=args.max_bytes,
        min_age_seconds=args.min_age_seconds,
        dry_run=args.dry_run,
        on_result=_log_progress,
        delete_after_upload=args.delete_after_upload,
    )

    print(format_upload_summary(device_id, args.key_prefix, results))

    had_failures = any(r.status in {"failed", "uploaded_unmarked"} for r in results)
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
