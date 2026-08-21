#!/usr/bin/env python3
"""Upload local files to the sg-unprocessed S3 bucket.

Each file is uploaded under its own name. On success the local file is
renamed with a `.uploaded` suffix so a re-run skips it. Uses the ambient
AWS credential chain (env vars / ~/.aws/credentials / instance role) --
no credentials are read or written by this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boto3

from bugcam.unprocessed_upload import (
    DEFAULT_UNPROCESSED_BUCKET,
    bucket_exists,
    format_upload_summary,
    parse_size,
    upload_pending_files,
    verify_uploaded_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Local directory of files to upload")
    parser.add_argument("--bucket", default=DEFAULT_UNPROCESSED_BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="Optional S3 key prefix")
    parser.add_argument(
        "--max-bytes",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="Cap on data uploaded this run, e.g. 500MB, 2GiB, or a raw byte count. Unset = no limit.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded, without uploading")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the end-of-run check that every .uploaded-marked file has a matching S3 object",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.source_dir.is_dir():
        print(f"Error: source dir not found: {args.source_dir}", file=sys.stderr)
        return 1

    s3_client = boto3.client("s3")

    if not args.dry_run and not bucket_exists(s3_client, args.bucket):
        print(
            f"Error: bucket '{args.bucket}' does not exist or is not accessible. "
            "This script does not create buckets -- provision it first.",
            file=sys.stderr,
        )
        return 1

    results = upload_pending_files(
        args.source_dir,
        bucket=args.bucket,
        prefix=args.prefix,
        max_bytes=args.max_bytes,
        dry_run=args.dry_run,
        s3_client=s3_client,
    )

    print(format_upload_summary(args.bucket, results))

    had_failures = any(r.status in {"failed", "uploaded_unmarked"} for r in results)

    if args.dry_run:
        return 1 if had_failures else 0

    if not args.skip_verify:
        missing = verify_uploaded_files(
            args.source_dir,
            bucket=args.bucket,
            prefix=args.prefix,
            s3_client=s3_client,
        )
        if missing:
            print(f"\nMarked .uploaded locally but NOT in s3://{args.bucket}: {len(missing)} file(s)")
            for path in missing:
                print(f"  - {path.name}")
            return 1
        print(f"\nVerified: every .uploaded-marked file in {args.source_dir} is present in s3://{args.bucket}")

    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
