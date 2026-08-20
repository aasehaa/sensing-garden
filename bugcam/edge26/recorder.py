"""
Video recorder with continuous and interval modes.

Supports two recording modes:
    - Continuous: gapless chunk saving, no interruption between chunks.
    - Interval: record one chunk, release camera, wait N minutes, repeat.

Architecture:
    - Camera streams directly to picamera2's H264Encoder (software encoding
      -- there is no HW H.264 encoder on Pi 5)
    - Encoder writes raw H.264, remuxed to MP4 via ffmpeg -c copy
    - No frame queue or grabber thread
"""

import os
import shutil
import subprocess
import time
import queue
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple

from bugcam.record_window import RecordingWindow

logger = logging.getLogger(__name__)
MIN_FREE_DISK_BYTES = 500 * 1024 * 1024

# Failure recovery: after a recording error the camera and encoder are torn
# down and rebuilt from scratch. If that fails this many times in a row, the
# picamera2/V4L2 state is assumed unrecoverable and the process exits nonzero
# so systemd (Restart=on-failure) recreates it with a clean slate.
MAX_CONSECUTIVE_FAILURES = 3
RECOVERY_BACKOFF_SECONDS = 5.0
# Wait between attempts when a chunk is skipped without an error (e.g. low
# disk) so the continuous loop never retries in a tight spin.
NO_CHUNK_RETRY_SECONDS = 30.0
FATAL_EXIT_CODE = 1


class VideoRecorder:
    """
    Video recorder supporting continuous and interval modes.

    Saves fixed-duration video chunks to disk. Camera resolution is
    auto-detected. Completed chunk paths are pushed to a queue for
    downstream processing.
    """

    def __init__(
        self,
        output_dir: str,
        fps: int,
        chunk_duration: int,
        resolution: Tuple[int, int],
        device_id: str,
        video_queue: Optional[queue.Queue] = None,
        recording_mode: str = "continuous",
        interval_minutes: float = 5,
        bitrate: int = 20_000_000,
        record_window: Optional[RecordingWindow] = None,
        on_chunk_complete=None,
    ):
        """
        Initialize the video recorder.

        Args:
            output_dir: Directory to save video chunks (e.g., external USB path)
            fps: Frames per second for recording
            chunk_duration: Duration of each chunk in seconds
            resolution: Requested recording resolution as (width, height)
            device_id: Identifier for filename prefix
            video_queue: Queue to put completed video paths for downstream processing
            recording_mode: "continuous" (no gaps) or "interval" (record every N minutes)
            interval_minutes: Minutes between start of recordings (interval mode only)
            bitrate: H.264 encoder bitrate in bps
            record_window: Daily local-time window outside which no chunks are
                recorded (camera released, loop idles); None records around the clock
            on_chunk_complete: Callback (path, duration_seconds) invoked for every
                completed chunk, e.g. to log sampling effort
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.chunk_duration = chunk_duration
        self.requested_resolution = resolution
        self.device_id = device_id
        self.video_queue = video_queue
        self.recording_mode = recording_mode
        self.interval_minutes = interval_minutes
        self.bitrate = bitrate
        self.record_window = record_window
        self.on_chunk_complete = on_chunk_complete

        # Resolution is requested from config and confirmed during init.
        self.resolution: Tuple[int, int] = (0, 0)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Threading controls
        self.stop_event = threading.Event()
        self.camera = None
        self.encoder = None

        # Last completed chunk (read by Pipeline after stop)
        self.last_chunk_path: Optional[Path] = None

        # Calculate expected frames per chunk
        self.frames_per_chunk = fps * chunk_duration

        logger.info("VideoRecorder initialized:")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Target FPS: {fps}, Chunk duration: {chunk_duration}s")
        logger.info(f"  Requested resolution: {resolution[0]}x{resolution[1]}")
        logger.info(f"  Recording mode: {recording_mode}"
                    + (f", interval: {interval_minutes} min" if recording_mode == "interval" else ""))
        if record_window is not None:
            logger.info(f"  Recording window: {record_window.describe()}")

    def _init_camera(self) -> None:
        """Initialize the camera (picamera2) and read its resolution."""
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder, Quality

        self.camera = Picamera2()

        # Create requested video config and read final applied resolution
        # Picamera2 "RGB888" outputs BGR order [B,G,R] which matches OpenCV expectation.
        config = self.camera.create_video_configuration(
            main={"size": self.requested_resolution, "format": "RGB888"}
        )
        self.camera.configure(config)

        # Read resolution from the configured camera
        width = config["main"]["size"][0]
        height = config["main"]["size"][1]
        self.resolution = (width, height)

        # Set frame rate
        self.camera.set_controls({
            "FrameRate": float(self.fps),
        })

        # Create H.264 encoder (software -- see module docstring)
        self.encoder = H264Encoder(bitrate=self.bitrate, framerate=self.fps)
        self.encoder_quality = Quality.HIGH

        self.camera.start()

        # Allow camera to warm up
        time.sleep(2)
        logger.info(f"PiCamera started: {self.resolution} @ {self.fps}fps")

        # Recalculate frames per chunk with actual fps
        self.frames_per_chunk = self.fps * self.chunk_duration

        logger.info(f"Recording: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
        logger.info(f"Chunk: {self.chunk_duration}s = {self.frames_per_chunk} frames")

    def _release_camera(self) -> None:
        """Release camera resources."""
        if self.camera is None:
            return

        try:
            self.camera.stop()
        finally:
            self.camera.close()

        self.camera = None
        logger.info("Camera released")

    def _generate_chunk_path(self) -> Path:
        """
        Generate a unique path for a new video chunk.

        Filename format: {device_id}_{YYYYMMDD}_{HHMMSS}_{ffffff}.mp4
        Timestamp reflects when recording started, in UTC: the stem is the
        pipeline's source of observation time (result dir names, S3 keys,
        video_timestamp), so it must be unambiguous across deployment sites.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.device_id}_{timestamp}.mp4"
        return self.output_dir / filename

    @staticmethod
    def _check_ffmpeg_available() -> bool:
        """Check if ffmpeg is available on the system."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except Exception:
            return False

    def _remux_chunk(self, src: Path, dst: Path) -> bool:
        """Remux raw H.264 to MP4 container. Returns True on success."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(src),
                    "-c", "copy",
                    "-r", str(self.fps),
                    str(dst),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                src.unlink(missing_ok=True)
                return True
            logger.error(f"Remux failed: {result.stderr}")
            return False
        except FileNotFoundError:
            logger.warning("ffmpeg not found, using raw H.264 file")
            return False
        except Exception as e:
            logger.error(f"Remux error: {e}")
            return False

    def _record_chunk(self) -> Optional[Path]:
        """
        Record a single chunk using picamera2's H.264 encoder.

        The encoder writes raw H.264 to a temp file, then remuxes to MP4.
        Camera stays running between chunks so there is no re-init delay.

        Raises on camera/encoder failure — the caller owns recovery (teardown,
        backoff, re-init) and escalation. Returns None only for benign skips
        (low disk, stop requested before any data was written).
        """
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < MIN_FREE_DISK_BYTES:
            logger.warning(
                "Skipping chunk because free space is low: %.1fMB available",
                free_bytes / (1024 * 1024),
            )
            return None

        chunk_path = self._generate_chunk_path()
        temp_h264 = chunk_path.with_suffix(".h264")
        logger.info(f"Recording chunk: {chunk_path.name}")

        try:
            recording_started = time.monotonic()
            self.camera.start_recording(
                self.encoder,
                str(temp_h264),
                quality=self.encoder_quality,
            )

            chunk_end = time.time() + self.chunk_duration
            while time.time() < chunk_end and not self.stop_event.is_set():
                time.sleep(0.1)

            self.camera.stop_recording()
            # Measured, not nominal: an early stop shortens the chunk.
            recorded_seconds = time.monotonic() - recording_started
        except Exception:
            # picamera2 opens the output file before starting the encoder, so
            # a failed start (e.g. encoder still attached from an earlier
            # failure) litters an empty .h264 on every attempt.
            try:
                if temp_h264.exists() and temp_h264.stat().st_size == 0:
                    temp_h264.unlink()
            except OSError:
                pass
            raise

        if self.stop_event.is_set() and not temp_h264.exists():
            return None

        # A good recording already happened above; a remux/rename/stat error
        # here is a filesystem hiccup (e.g. a flaky SD-card write), not a
        # camera/encoder fault. Keep it out of the caller's failure counter --
        # it must not tear down or rebuild a perfectly healthy camera.
        try:
            if temp_h264.exists():
                has_ffmpeg = self._check_ffmpeg_available()
                if has_ffmpeg:
                    remuxed = self._remux_chunk(temp_h264, chunk_path)
                    if not remuxed:
                        temp_h264.rename(chunk_path)
                else:
                    temp_h264.rename(chunk_path)
                    logger.warning(
                        f"Chunk saved as raw H.264 (no ffmpeg): {chunk_path.name}"
                    )

            if chunk_path.exists():
                size_mb = chunk_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Chunk complete: {chunk_path.name} ({size_mb:.1f}MB)"
                )
                self._notify_chunk_complete(chunk_path, recorded_seconds)
                return chunk_path
        except OSError:
            logger.error(
                f"Chunk finalize failed for {chunk_path.name} (disk error, not a camera fault)",
                exc_info=True,
            )
            return None

        return None

    def _notify_chunk_complete(self, chunk_path: Path, duration_seconds: float) -> None:
        """Report a completed chunk to the consumer; its failures must never
        break the recording loop."""
        if self.on_chunk_complete is None:
            return
        try:
            self.on_chunk_complete(chunk_path, duration_seconds)
        except Exception:
            logger.error(f"on_chunk_complete callback failed for {chunk_path.name}", exc_info=True)

    def start(self) -> None:
        """
        Start video recording (continuous or interval).

        This method blocks and runs the recording loop.
        Call from a thread if you need non-blocking behavior.
        """
        if self.recording_mode == "interval":
            self._start_interval()
        else:
            self._start_continuous()

    def _window_open(self) -> bool:
        """Whether the recording window currently allows recording."""
        return self.record_window is None or self.record_window.is_open()

    def _wait_for_window(self) -> bool:
        """Block until the recording window opens or a stop is requested.

        Returns True when recording may proceed, False when stopping.
        """
        if self._window_open():
            return not self.stop_event.is_set()
        logger.info(
            f"Outside recording window ({self.record_window.describe()}); "
            "recording paused"
        )
        while not self.stop_event.is_set():
            if self._window_open():
                logger.info("Recording window open; resuming recording")
                return True
            time.sleep(1.0)
        return False

    def _start_continuous(self) -> None:
        """Record non-stop, chunk after chunk with no gaps.

        Outside the recording window the camera is released and the loop
        idles; it re-initializes the camera when the window reopens. Camera
        errors are recovered in place (teardown, backoff, retry) rather than
        ending the loop: a failed stop_recording() otherwise leaves the
        encoder attached and every retry would throw while littering empty
        files. After MAX_CONSECUTIVE_FAILURES the process exits nonzero so
        systemd restarts it -- the library state itself is bad at that point.
        """
        logger.info("Starting continuous recording...")

        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                if not self._window_open():
                    self._cleanup(final=False)
                    if not self._wait_for_window():
                        break
                try:
                    if self.camera is None:
                        self._init_camera()
                    chunk_path = self._record_chunk()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    consecutive_failures = self._handle_recording_failure(
                        consecutive_failures, "Recording"
                    )
                    continue

                if chunk_path:
                    consecutive_failures = 0
                    self.last_chunk_path = chunk_path
                    if self.video_queue:
                        self.video_queue.put(chunk_path)
                elif not self.stop_event.is_set():
                    self.stop_event.wait(NO_CHUNK_RETRY_SECONDS)
        finally:
            self._cleanup(final=True)

    def _start_interval(self) -> None:
        """Record one chunk, wait, repeat."""
        interval_seconds = self.interval_minutes * 60

        logger.info(f"Starting interval recording "
                    f"({self.chunk_duration}s every {self.interval_minutes} min)...")

        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                if not self._wait_for_window():
                    break

                chunk_start = time.time()

                try:
                    # Init camera, record one chunk, release camera
                    self._init_camera()
                    chunk_path = self._record_chunk()
                except Exception:
                    if self.stop_event.is_set():
                        break
                    consecutive_failures = self._handle_recording_failure(
                        consecutive_failures, "Recording iteration"
                    )
                else:
                    if chunk_path:
                        consecutive_failures = 0
                        self.last_chunk_path = chunk_path
                        if self.video_queue:
                            self.video_queue.put(chunk_path)

                    self._cleanup()

                if self.stop_event.is_set():
                    break

                # Wait until next interval
                elapsed = time.time() - chunk_start
                wait_time = max(0, interval_seconds - elapsed)

                if wait_time > 0:
                    logger.info(f"Next recording in {wait_time:.0f}s")
                    # Sleep in small increments so stop_event is responsive
                    wait_end = time.time() + wait_time
                    while time.time() < wait_end and not self.stop_event.is_set():
                        time.sleep(min(1.0, wait_end - time.time()))

        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
        finally:
            self._cleanup(final=True)

    def _handle_recording_failure(self, consecutive_failures: int, context: str) -> int:
        """Log a recording failure, escalate to a fatal process exit past the
        threshold, and tear down for a fresh retry. Shared by every recording
        loop so the counter/backoff/escalation policy lives in one place.

        Must be called from within the except block for the failure so
        exc_info=True captures it. Returns the updated failure count.
        """
        consecutive_failures += 1
        logger.error(
            f"{context} failed (consecutive failure "
            f"{consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})",
            exc_info=True,
        )
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self._fatal_exit()
        self._teardown_for_recovery()
        return consecutive_failures

    def _teardown_for_recovery(self) -> None:
        """Best-effort camera/encoder teardown after a failure. Never raises.

        stop_recording() detaches an encoder a failed chunk may have left
        running — without this, every later start_recording() throws. The
        camera object is then discarded so the next attempt builds a fresh
        Picamera2 and H264Encoder instead of reusing possibly-bad state.
        """
        if self.camera is not None:
            try:
                self.camera.stop_recording()
                logger.info("Released encoder left attached by the failed chunk")
            except Exception:
                pass
        try:
            self._cleanup()
        except Exception as e:
            logger.warning(f"Camera release during recovery failed: {e}")
            self.camera = None
        self.encoder = None
        self.stop_event.wait(RECOVERY_BACKOFF_SECONDS)

    def _fatal_exit(self) -> None:
        """Exit the whole process so the service manager restarts it clean."""
        logger.critical(
            "Camera/encoder unrecoverable after %d consecutive failures; "
            "exiting so systemd can restart the process",
            MAX_CONSECUTIVE_FAILURES,
        )
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        os._exit(FATAL_EXIT_CODE)

    def stop(self) -> None:
        """
        Signal the recorder to stop.

        The current chunk will be finalized before stopping.
        """
        logger.info("Stopping recorder...")
        self.stop_event.set()

    def _cleanup(self, final: bool = False) -> None:
        """
        Clean up camera resources.

        Args:
            final: If True, also sets stop_event (full shutdown).
                   If False, only tears down the camera (interval pause).
        """
        if final:
            self.stop_event.set()

        self._release_camera()

        logger.debug("Recorder resources released")
