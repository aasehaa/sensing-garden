"""
Classification queue manager for interleaving FLIK and DOT classification.

Provides a disk-based FIFO queue that persists across crashes and allows
fair scheduling of classification tasks from both FLIK detection and DOT uploads.
Failed entries are moved to a dead-letter directory with a max count limit.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
MAX_FAILED_ENTRIES = 50


@dataclass
class QueueEntry:
    """Represents a track queued for classification."""
    entry_type: str                          # "flik" or "dot"
    source_device: str
    date: str
    time: Optional[str]
    track_id: str
    track_dir: str                           # Path to crops directory
    labels_path: Optional[str] = None        # Path to labels JSON (for DOT)
    background_path: Optional[str] = None    # Path to background image (for DOT composite)
    num_crops: int = 0
    output_dir: str = ""                     # Where to write final results
    queued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retry_count: int = 0
    last_error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "QueueEntry":
        data = json.loads(json_str)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class ClassificationQueue:
    """
    Disk-based FIFO queue for classification tasks.

    Queues tracks from both FLIK detection and DOT uploads for
    fair interleaved processing. Files are written to disk so
    pending classifications survive crashes.

    Failed entries are moved to a 'failed/' subdirectory after
    MAX_RETRIES attempts. The failed directory is pruned to
    MAX_FAILED_ENTRIES entries on a FIFO (oldest-first) basis.
    """

    def __init__(self, pending_dir: Path):
        self.pending_dir = Path(pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir = self.pending_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def enqueue(
        self,
        entry_type: str,
        source_device: str,
        date: str,
        track_id: str,
        track_dir: Path,
        output_dir: Path,
        time: Optional[str] = None,
        labels_path: Optional[Path] = None,
        background_path: Optional[Path] = None,
        num_crops: int = 0,
    ) -> Path:
        """Queue a track for classification."""
        entry = QueueEntry(
            entry_type=entry_type,
            source_device=source_device,
            date=date,
            time=time,
            track_id=track_id,
            track_dir=str(track_dir),
            labels_path=str(labels_path) if labels_path else None,
            background_path=str(background_path) if background_path else None,
            num_crops=num_crops,
            output_dir=str(output_dir),
        )

        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{source_device}_{date}_{timestamp}_{track_id}.json"
            filepath = self.pending_dir / filename
            # Temp name must not end in .json: get_next()'s *.json glob matches
            # leading-dot names on Python >= 3.11, so a half-written temp file
            # could be picked up (or vanish between glob and stat).
            temp_path = self.pending_dir / f"{filename}.tmp"
            temp_path.write_text(entry.to_json())
            temp_path.rename(filepath)
            logger.debug(f"Queued for classification: {filepath.name}")

        return filepath

    def get_next(self) -> Optional[tuple[Path, QueueEntry]]:
        """Get oldest pending entry (FIFO by mtime)."""
        with self._lock:
            files = sorted(
                self.pending_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime
            )

            if not files:
                return None

            filepath = files[0]
            try:
                entry = QueueEntry.from_json(filepath.read_text())
                logger.debug(f"Retrieved from queue: {filepath.name}")
                return filepath, entry
            except Exception as e:
                logger.error(f"Failed to read queue entry {filepath}: {e}")
                corrupted = filepath.with_suffix(".corrupted")
                filepath.rename(corrupted)
                logger.warning(f"Moved corrupted file to {corrupted}")
                return None

    def remove(self, filepath: Path) -> None:
        """Remove processed entry from queue."""
        with self._lock:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Removed from queue: {filepath.name}")

    def mark_failed(self, filepath: Path, entry: QueueEntry, error: str) -> bool:
        """Handle a failed classification attempt.

        Increments retry count on the entry. If MAX_RETRIES is exceeded,
        moves the entry to the failed/ directory. Otherwise, rewrites it
        with the updated retry count so it will be retried.

        Returns True if entry should be retried, False if moved to failed/.
        """
        entry.retry_count += 1
        entry.last_error = error[:500]

        if entry.retry_count >= MAX_RETRIES:
            with self._lock:
                failed_path = self.failed_dir / filepath.name
                try:
                    filepath.rename(failed_path)
                except FileNotFoundError:
                    pass
                logger.error(f"Entry {filepath.name} exceeded {MAX_RETRIES} retries, "
                            f"moved to failed/: {error}")
            self._prune_failed()
            return False

        with self._lock:
            if filepath.exists():
                # Rewrite atomically: an in-place write torn by a crash leaves a
                # truncated entry that get_next() can only discard as corrupted,
                # permanently losing the track's completion count.
                temp_path = filepath.with_name(f"{filepath.name}.tmp")
                temp_path.write_text(entry.to_json())
                temp_path.rename(filepath)
        logger.warning(f"Entry {filepath.name} failed "
                      f"(retry {entry.retry_count}/{MAX_RETRIES}): {error}")
        return True

    def _prune_failed(self) -> None:
        """Remove oldest failed entries beyond MAX_FAILED_ENTRIES."""
        failed_files = sorted(
            self.failed_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_file in failed_files[MAX_FAILED_ENTRIES:]:
            old_file.unlink()
            logger.debug(f"Pruned old failed entry: {old_file.name}")

    def count(self) -> int:
        """Count pending entries."""
        with self._lock:
            return len(list(self.pending_dir.glob("*.json")))

    def recover(self) -> int:
        """Recover pending entries after crash. Returns count."""
        # An orphaned temp file is an enqueue that died before the rename: that
        # track was already counted in .expected_tracks but will never be
        # classified, so its directory relies on the stale sweep to finalize.
        with self._lock:
            for temp_path in list(self.pending_dir.glob("*.tmp")) + list(self.pending_dir.glob(".tmp_*.json")):
                logger.warning(f"Removing orphaned queue temp file (lost enqueue): {temp_path.name}")
                temp_path.unlink(missing_ok=True)
        count = self.count()
        if count > 0:
            logger.info(f"Recovered {count} pending classification(s) from {self.pending_dir}")
        self._prune_failed()
        return count

    def get_pending_entries(self) -> list[tuple[Path, QueueEntry]]:
        """Get all pending entries sorted by mtime (for debugging/inspection)."""
        entries = []
        with self._lock:
            files = sorted(
                self.pending_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime
            )
            for filepath in files:
                try:
                    entry = QueueEntry.from_json(filepath.read_text())
                    entries.append((filepath, entry))
                except Exception:
                    pass
        return entries


def get_pending_dir() -> Path:
    """Get the pending classification queue directory."""
    from bugcam.settings import get_state_dir
    return get_state_dir() / "pending"