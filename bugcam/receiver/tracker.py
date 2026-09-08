"""Track finalization for DOT receiver.

Tracks which track directories have recently received data.
After FINALIZATION_DELAY seconds of inactivity, creates done.txt
as a fallback for when the iOS app doesn't send /upload_done.
Thread-safe for use with Flask's threaded mode.
"""

import time
import threading
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PendingTrackTracker:
    """Tracks which track directories have recently received data.

    After FINALIZATION_DELAY seconds of inactivity, creates done.txt
    as a fallback for when the iOS app doesn't send /upload_done.
    Thread-safe for use with Flask's threaded mode.
    """

    FINALIZATION_DELAY = 60.0
    STALE_AGE = 600
    CHECK_INTERVAL = 2.0

    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self._tracks = {}
        self._lock = threading.Lock()

    def update(self, dot_dir: str, track_dir: str, has_crops: bool = False):
        """Record activity on a track directory. Called on every upload."""
        key = (dot_dir, track_dir)
        with self._lock:
            entry = self._tracks.setdefault(key, {"last_activity": time.time(), "has_crops": False})
            entry["last_activity"] = time.time()
            if has_crops:
                entry["has_crops"] = True

    def mark_done(self, dot_dir: str, track_dir: str):
        """Remove a track from tracking (done.txt created or confirmed)."""
        key = (dot_dir, track_dir)
        with self._lock:
            self._tracks.pop(key, None)

    def check_pending(self):
        """Check all pending tracks and finalize those idle > FINALIZATION_DELAY
        or stale > STALE_AGE. Returns list of finalized track keys."""
        now = time.time()
        finalized = []
        stale_keys = []
        with self._lock:
            for key, entry in list(self._tracks.items()):
                age = now - entry["last_activity"]
                should_finalize = False

                if age > self.FINALIZATION_DELAY and entry["has_crops"]:
                    should_finalize = True
                elif age > self.STALE_AGE:
                    should_finalize = True

                if should_finalize:
                    dot_dir, track_dir = key
                    track_path = self.storage_root / dot_dir / "crops" / track_dir
                    done_path = track_path / "done.txt"
                    if track_path.exists() and not done_path.exists():
                        try:
                            done_path.write_text(
                                f"Track completed at {datetime.now().isoformat()}\n"
                            )
                            logger.info(f"Finalized track {track_dir} "
                                        f"(idle {age:.0f}s)")
                            finalized.append(key)
                        except Exception as e:
                            logger.error(f"Failed to create done.txt for "
                                         f"{track_dir}: {e}")
                    stale_keys.append(key)

            for key in stale_keys:
                self._tracks.pop(key, None)

        return finalized

    def recover_orphaned_tracks(self, max_age: float = 30.0):
        """Scan for track directories that have files but no done.txt.

        Finalizes those whose oldest file is older than max_age seconds.
        Called at startup to handle receiver restarts.
        """
        now = time.time()
        recovered = 0
        for dot_dir in self.storage_root.iterdir():
            if not dot_dir.is_dir():
                continue
            crops_dir = dot_dir / "crops"
            if not crops_dir.exists():
                continue
            for track_dir in crops_dir.iterdir():
                if not track_dir.is_dir():
                    continue
                done_path = track_dir / "done.txt"
                if done_path.exists():
                    continue
                files = [f for f in track_dir.iterdir()
                          if f.is_file() and f.name != "done.txt"]
                if not files:
                    continue
                oldest = min(f.stat().st_mtime for f in files)
                if now - oldest > max_age:
                    try:
                        done_path.write_text(
                            f"Track completed at {datetime.now().isoformat()} "
                            f"(recovered)\n"
                        )
                        logger.info(f"Recovered orphaned track: "
                                    f"{dot_dir.name}/{track_dir.name}")
                        recovered += 1
                    except Exception as e:
                        logger.error(f"Failed to recover {track_dir.name}: {e}")
        if recovered:
            logger.info(f"Recovered {recovered} orphaned track(s)")


def finalization_loop(tracker: PendingTrackTracker, stop_event: threading.Event) -> None:
    """Background thread that checks for idle tracks to finalize.

    Shared by both `bugcam run` (receiver embedded alongside the pipeline)
    and `bugcam receive start` (standalone receiver server).
    """
    while not stop_event.is_set():
        try:
            tracker.check_pending()
        except Exception as e:
            logger.error(f"Finalization loop error: {e}")
        stop_event.wait(PendingTrackTracker.CHECK_INTERVAL)
