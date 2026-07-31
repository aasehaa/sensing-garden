import json
import logging
import multiprocessing as mp
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

import cv2

from bugcam.edge26.capture import VideoRecorder
from bugcam.edge26.processing import VideoProcessor, HailoClassifier
from bugcam.edge26.output import ResultsWriter
from bugcam.edge26.metrics import PipelineMetrics
from bugcam.edge26.queue import ClassificationQueue, QueueEntry
from bugcam.edge26.result_health import audit_result_dir
from bugcam.edge26.sidecars import (
    COMPLETED_TRACKS,
    DETECTION_META,
    DONE,
    EXPECTED_TRACKS,
    LAST_RECORDING,
    RESULTS,
    SWEEP_NON_CONTENT,
)
from bugcam.log_shipping import DailyLogHandler, ship_existing_logs
from bugcam.record_window import RecordingWindow, local_video_date, resolve_zone, video_stem_utc_iso
from bugcam.capture_report import CAPTURES_SUBDIR

# Producer-owned utility dirs under a device dir, not per-timestamp result
# directories -- the sweep and inventory must never treat them as results
# (rmtree'ing one races its live writer, e.g. the heartbeat loop).
NON_RESULT_SUBDIRS = {"heartbeats", "environment", "logs", CAPTURES_SUBDIR}


def _video_sample_decision(
    *, sample_saved: bool, is_last_in_batch: bool, confirmed_count: int, random_sampling: bool,
) -> tuple[bool, str | None]:
    """Decide whether to save this FLIK video as the batch's uploaded sample.

    Default mode prioritizes the first video in the batch with confirmed
    tracks over a later plain one, falling back to the batch's last video if
    none had confirmed tracks -- the sample is "best of N". ``random_sampling``
    disables that bias: the saved sample is always the batch's actual Nth
    (last) video, regardless of what any video in the batch contained.

    Returns (should_save, reason) where reason is one of "detections"
    (default mode, a confirmed track triggered the save), "fallback"
    (default mode, batch ended with no confirmed tracks), "interval"
    (random-sampling mode, batch simply ended), or None if not saving.
    """
    if sample_saved:
        return False, None
    if random_sampling:
        return (True, "interval") if is_last_in_batch else (False, None)
    if confirmed_count > 0:
        return True, "detections"
    if is_last_in_batch:
        return True, "fallback"
    return False, None


def setup_logging(log_dir: Path, *, on_log_complete=None) -> None:
    """Configure logging to console and a daily-rotating file.

    When ``on_log_complete`` is given, the log mechanism owns shipping: a completed
    (rolled-over) file is pushed to it, and any non-today logs left by a prior run are
    shipped now. The upload subsystem never scans for logs. When it is ``None``
    (uploads disabled), nothing ships logs -- they accumulate on disk locally."""
    log_dir.mkdir(parents=True, exist_ok=True)

    #TODO I think this is handling logging for the application broadly,
    # so it should be declared outside of edge26
    file_handler = DailyLogHandler(log_dir)
    file_handler.on_complete = on_log_complete

    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"

    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ]
    )

    if on_log_complete is not None:
        ship_existing_logs(on_log_complete, log_dir)

    # Reduce noise from libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("hailo_platform").setLevel(logging.WARNING)


logger = logging.getLogger("edge26")


class Pipeline:
    """
    Main pipeline orchestrating capture and processing.
    
    Architecture:
        - Detection thread: Runs BugSpot detection/tracking (maintains tracker state)
        - Classification thread: Runs Hailo classification (shared resource)
        - Classification queue: Disk-based FIFO queue for both FLIK and DOT tracks
    """
    
    def __init__(
        self,
        config: dict,
        *,
        detection_child: bool = False,
        shared_video_queue=None,
        shared_stop_event=None,
        shared_recording_stopped=None,
        shared_metrics=None,
        on_result_ready=None,
        on_video_ready=None,
        on_chunk_recorded=None,
    ):
        self.config = config
        self._on_result_ready = on_result_ready
        self._on_video_ready = on_video_ready
        self._on_chunk_recorded = on_chunk_recorded

        # --- Pipeline mode (resolved early; queue/event types depend on it) ---
        pipeline_config = config.get("pipeline", {})
        self.enable_recording = pipeline_config.get("enable_recording", True)
        self.enable_processing = pipeline_config.get("enable_processing", True)
        self.enable_classification = pipeline_config.get("enable_classification", True)
        self.continuous_tracking = pipeline_config.get("continuous_tracking", False)
        # With recording disabled, treat the input dir as a drop folder: keep
        # the detection loop alive and poll for injected FLIK videos instead of
        # exiting once the startup scan drains.
        self.watch_input = pipeline_config.get("watch_input", False)

        # Run the GIL-heavy detection loop in its own subprocess so it cannot
        # starve the recorder threads (dropped-frame fix). The child is a
        # detection-only Pipeline; the parent keeps recording + classification.
        self.detection_in_subprocess = pipeline_config.get("detection_in_subprocess", False)
        self._detection_child = detection_child
        if self._detection_child:
            # The detection child never records or classifies.
            self.enable_recording = False
            self.enable_classification = False
            self.enable_processing = True
        # This instance runs the detection loop (and owns the tracker-reset
        # markers) when monolithic, or when it is the detection child.
        self._owns_detection = (not self.detection_in_subprocess) or self._detection_child

        # Coordination primitives. In subprocess mode the recorder (parent) and
        # detection loop (child) live in different processes, so the queue and
        # events must be multiprocessing-backed and shared between them.
        if self.detection_in_subprocess:
            self._mp_ctx = mp.get_context("spawn")
            self.video_queue = shared_video_queue or self._mp_ctx.JoinableQueue()
            self.stop_event = shared_stop_event or self._mp_ctx.Event()
            self.recording_stopped = shared_recording_stopped or self._mp_ctx.Event()
        else:
            self._mp_ctx = None
            self.video_queue = queue.Queue()
            self.stop_event = threading.Event()
            self.recording_stopped = threading.Event()

        # Stage timings + event counters surfaced through the heartbeat. Shared
        # with the detection child (mp-backed) so its timings land in the
        # parent's snapshots.
        self.metrics = shared_metrics or PipelineMetrics(ctx=self._mp_ctx)
        # Process-lifetime marker: the heartbeat's system uptime survives a
        # systemd restart of the service, so a crash + quick restart would be
        # invisible without a clock that resets with the process.
        self._started_monotonic = time.monotonic()

        self.recorder_thread = None
        self.detection_thread = None
        self.detection_process = None
        self.classification_thread = None
        
        # Device config
        device_config = config.get("device", {})
        self.flick_id = device_config.get("flick_id", "edge26")
        self.dot_ids = device_config.get("dot_ids", [])
        # Filenames/timestamps are UTC; the configured zone localizes the
        # recording window and day boundaries (None: UTC days, no window).
        self.timezone_name = device_config.get("timezone") or None
        self._local_zone = None
        if self.timezone_name:
            try:
                self._local_zone = resolve_zone(self.timezone_name)
            except ValueError:
                logger.error(
                    f"Unknown timezone {self.timezone_name!r}; "
                    "day boundaries fall back to UTC and no recording window applies"
                )
                self.timezone_name = None
        self.input_storage = Path(config["paths"]["input_storage"])
        
        # Output paths
        self.results_dir = Path(config["output"]["results_dir"])
        
        # Pending queue for classification
        pending_dir = Path(config["paths"].get("pending_dir", 
                         Path(config["paths"]["input_storage"]).parent / "pending"))
        self.classification_queue = ClassificationQueue(pending_dir)
        
        # --- Video sampling (save 1 video per N to output) ---
        self._video_batch_count = 0
        self._video_sample_saved = False
        self._video_sample_interval = pipeline_config.get("video_sample_interval", 10)
        # --random-sampling: disable the confirmed-track priority above so the
        # saved sample is the batch's actual Nth video, not "best of N".
        self._random_sampling = pipeline_config.get("random_sampling", False)
        
        # --- Tracker reset signals (continuous_tracking mode) ---
        self._last_sweep_monotonic = time.monotonic()
        self._sweep_interval_seconds = 300.0
        self._status_interval_seconds = 600.0
        self._status_thread = None
        # 1. Day-change: reset when the date in the filename changes
        self._last_video_date: str = ""
        # 2. Recording-stop: reset after the last recorded video is processed
        #    Persisted via .last_recording marker file so it survives restarts.
        self._reset_after_video: str = ""
        self._pending_tracker_reset = False
        # Only the instance that runs the detection loop touches the tracker-reset
        # markers. In subprocess mode that's the child (at its own startup), so the
        # parent must not consume/unlink the marker out from under it.
        if self.continuous_tracking and self._owns_detection:
            self._load_last_recording_marker()
        
        # Initialize components based on mode
        self.recorder = self._init_recorder() if self.enable_recording else None
        self.processor = VideoProcessor(config) if self.enable_processing else None
        self.writer = ResultsWriter(config["output"]) if self.enable_processing else None
        
        # Eagerly initialize classifier for the classification thread
        if self.enable_classification and self.processor:
            self.processor._classifier = HailoClassifier(self.processor.classification_config)
            logger.info("Hailo classifier initialized")
        
        logger.info("=" * 60)
        logger.info("EDGE26 PIPELINE INITIALIZED")
        logger.info("=" * 60)
        
        # Mode info
        mode = "RECORD + PROCESS" if (self.enable_recording and self.enable_processing) else \
               "RECORD ONLY" if self.enable_recording else \
               "PROCESS ONLY" if self.enable_processing else "NONE"
        logger.info(f"Mode:          {mode}")
        logger.info(f"Device:        {self.flick_id}")
        logger.info(f"Input storage: {config['paths']['input_storage']}")
        logger.info(f"Pending dir:   {pending_dir}")
        if self.enable_processing:
            logger.info(f"Results dir:   {config['output']['results_dir']}")
            classify = pipeline_config.get("enable_classification", True)
            logger.info(f"Classification: {'enabled' if classify else 'disabled (detection only)'}")
            cont_track = pipeline_config.get("continuous_tracking", True)
            logger.info(f"Tracking:      {'continuous (across videos)' if cont_track else 'per-video (reset each)'}")
        if self.dot_ids:
            logger.info(f"DOT devices:   {', '.join(self.dot_ids)}")
        if self.enable_recording:
            rec_mode = pipeline_config.get("recording_mode", "continuous")
            logger.info(f"Chunk duration: {config['capture']['chunk_duration_seconds']}s")
            logger.info(f"Recording mode: {rec_mode}"
                       + (f" (every {pipeline_config.get('recording_interval_minutes', 5)} min)"
                          if rec_mode == "interval" else ""))
    
    def _init_recorder(self) -> VideoRecorder:
        """Initialize video recorder from config."""
        paths = self.config["paths"]
        capture = self.config["capture"]
        pipeline_cfg = self.config.get("pipeline", {})
        
        return VideoRecorder(
            output_dir=paths["input_storage"],
            fps=capture["fps"],
            chunk_duration=capture["chunk_duration_seconds"],
            resolution=tuple(capture.get("resolution", [1080, 1080])),
            device_id=self.flick_id,
            video_queue=self.video_queue,
            camera_index=capture["camera_index"],
            use_picamera=capture["use_picamera"],
            recording_mode=pipeline_cfg.get("recording_mode", "continuous"),
            interval_minutes=pipeline_cfg.get("recording_interval_minutes", 5),
            bitrate=capture.get("bitrate", 20_000_000),
            record_window=RecordingWindow.from_config(
                pipeline_cfg.get("record_window"), self.timezone_name
            ),
            on_chunk_complete=self._on_chunk_recorded,
        )
    
    def _audit_finalized_dir(self, output_dir: Path) -> None:
        """Sanity-check a dir at the moment .done lands; one error line if unhealthy.

        A .done marker only proves the completion counter reached the expected
        count -- not that classification produced a sane result. Single
        greppable prefix so the log ingestion side can alert on it.
        """
        try:
            problems = audit_result_dir(output_dir)
        except Exception:
            logger.warning("result health audit failed for %s", output_dir, exc_info=True)
            return
        if problems:
            self.metrics.unhealthy_results.increment()
            logger.error("unhealthy result %s: %s", output_dir, "; ".join(problems))
        else:
            logger.debug(f"result health OK: {output_dir.name}")

    def _notify_result_ready(self, output_dir: Path) -> None:
        # Tell the upload owner (Pollen) a result dir is finalized, if wired.
        if self._on_result_ready is None:
            logger.warning(
                "no upload owner wired; %s is finalized (.done) but will NOT be shipped or retried",
                output_dir,
            )
            return
        try:
            self._on_result_ready(output_dir)
        except Exception:
            logger.error("result-ready callback failed for %s", output_dir, exc_info=True)

    def _notify_video_ready(self, video_path: Path, device: str) -> bool:
        """Tell the upload owner a DOT video is ready. DOT videos are not tied to a
        track, so they ship as their own unit keyed under the device/day. Returns True
        if an upload owner took it (staged it), so the producer can drop its copy."""
        if self._on_video_ready is None:
            logger.warning(
                "no upload owner wired; %s (device=%s) will NOT be shipped", video_path, device,
            )
            return False
        try:
            self._on_video_ready(video_path, device)
            return True
        except Exception:
            logger.error("video-ready callback failed for %s (device=%s)", video_path, device, exc_info=True)
            return False

    def _publish_dot_video(self, entry: QueueEntry) -> None:
        """Ship a queued DOT video to the upload owner, then drop the local copy. Runs in
        the main-process classification worker (which holds Pollen); detection only
        enqueues the task. Raises on failure so the queue retries rather than lose it."""
        video_path = Path(entry.track_dir)
        if not video_path.exists():
            return  # already shipped/cleaned: idempotent
        if not self._notify_video_ready(video_path, entry.source_device):
            raise RuntimeError(f"video enqueue failed (no upload owner?): {video_path.name}")
        video_path.unlink()
        logger.info("DOT video staged for upload: %s (dropped local copy)", video_path.name)

    def _publish_finalized_result(self, entry: QueueEntry) -> None:
        """Ship a finalized (.done) result dir to the upload owner. Runs in the
        main-process classification worker (which holds Pollen); detection only
        enqueues -- in subprocess mode it has no upload callback of its own.
        Publish failures are not retried through the queue: the directory stays
        on disk and the orphan sweep retries it, unlike DOT videos where the
        queue entry is the only pointer to the file."""
        output_dir = Path(entry.output_dir)
        if not output_dir.exists():
            return  # already published/cleaned: idempotent
        self._audit_finalized_dir(output_dir)
        self._notify_result_ready(output_dir)

    def _is_flick_video(self, path: Path) -> bool:
        """Check if a path is a FLICK video (matches flick_id prefix)."""
        return (path.is_file()
                and path.suffix == ".mp4"
                and path.name.startswith(f"{self.flick_id}_"))
    
    def _is_dot_directory(self, path: Path) -> bool:
        """Check if a path is a DOT device directory (matches a dot_id prefix)."""
        if not path.is_dir():
            return False
        return any(path.name.startswith(f"{dot_id}_") for dot_id in self.dot_ids)
    
    def _find_existing_items(self) -> list:
        """
        Find existing videos and DOT directories in input_storage.
        
        Returns a sorted list of (path, type) tuples where type is
        "video" or "dot". Only items matching configured device IDs
        are included. Sorted by name gives chronological order since
        filenames and directory names both contain timestamps.
        """
        if not self.input_storage.exists():
            return []
        
        items = []
        for entry in sorted(self.input_storage.iterdir()):
            if self._is_flick_video(entry):
                items.append((entry, "video"))
            elif self.dot_ids and self._is_dot_directory(entry):
                items.append((entry, "dot"))
        
        if items:
            n_videos = sum(1 for _, t in items if t == "video")
            n_dots = sum(1 for _, t in items if t == "dot")
            parts = []
            if n_videos:
                parts.append(f"{n_videos} video(s)")
            if n_dots:
                parts.append(f"{n_dots} DOT dir(s)")
            logger.info(f"Found {', '.join(parts)} to process")
        
        return items
    
    def _find_flick_videos(self) -> list:
        """Find unprocessed FLIK videos in input_storage (chronological)."""
        if not self.input_storage.exists():
            return []

        return [f for f in sorted(self.input_storage.iterdir())
                if self._is_flick_video(f)]

    def _find_dot_directories(self) -> list:
        """Find unprocessed DOT directories in input_storage."""
        if not self.input_storage.exists() or not self.dot_ids:
            return []
        
        return [d for d in sorted(self.input_storage.iterdir())
                if self._is_dot_directory(d)]
    
    def _parse_dot_dir_name(self, dir_name: str):
        """
        Parse a DOT directory name into (dot_id, date_str).
        
        Directory name format: {dot_id}_{YYYYMMDD}
        Returns (dot_id, "YYYYMMDD") or (None, None).
        """
        for dot_id in self.dot_ids:
            if dir_name.startswith(f"{dot_id}_"):
                date_str = dir_name[len(dot_id) + 1:]
                return dot_id, date_str
        return None, None
    
    def _compute_output_dir(self, device_id: str, date_time: str) -> Path:
        """Compute the output directory for a device and timestamp."""
        return self.results_dir / device_id / date_time
    
    def _find_ready_dot_tracks(self, dot_dir: Path) -> list:
        """Find tracks within a DOT directory that have a done.txt signal."""
        crops_dir = dot_dir / "crops"
        if not crops_dir.exists():
            return []
        return [d for d in sorted(crops_dir.iterdir())
                if d.is_dir() and (d / "done.txt").exists()]
    
    def _find_latest_background(self, dot_dir: Path):
        """Find the most recent background image in a DOT directory."""
        backgrounds = sorted(dot_dir.glob("*_background.jpg"))
        if backgrounds:
            return backgrounds[-1]
        fallback = dot_dir / "current_background.jpg"
        return fallback if fallback.exists() else None
    
    def _process_dot_media(self, dot_dir: Path) -> None:
        """Copy videos and backgrounds to output regardless of track readiness.
        
        Ensures media files reach S3 quickly even when no insect tracks
        have been detected yet. Called on every detection worker poll.
        """
        try:
            dot_id, date_str = self._parse_dot_dir_name(dot_dir.name)
            if not dot_id:
                return
            
            output_dir = self._compute_output_dir(dot_id, date_str)
            output_dir.mkdir(parents=True, exist_ok=True)
            copied_something = False
            
            videos_dir = dot_dir / "videos"
            if videos_dir.exists():
                dst_videos = output_dir / "videos"
                dst_videos.mkdir(parents=True, exist_ok=True)
                for vid in sorted(videos_dir.iterdir()):
                    if vid.is_file() and vid.suffix == ".mp4":
                        dst = dst_videos / vid.name
                        if not dst.exists():
                            shutil.copy2(vid, dst)
                            logger.info(f"  Video copied: {vid.name}")
                            copied_something = True
                            # Detection runs in a subprocess with no upload callback, so
                            # hand the video to the main-process classification worker
                            # (which owns Pollen) via the disk queue, same path as tracks.
                            self.classification_queue.enqueue(
                                entry_type="video",
                                source_device=dot_id,
                                date=date_str,
                                track_id=dst.stem,
                                track_dir=dst,
                                output_dir=dst.parent,
                            )
                        vid.unlink()
            
            background = self._find_latest_background(dot_dir)
            if background:
                dst_background = output_dir / background.name
                if not dst_background.exists():
                    shutil.copy2(background, dst_background)
                    logger.info(f"  Background copied: {background.name}")
            
            if copied_something:
                logger.info(f"MEDIA: Copied new files from {dot_dir.name} to {output_dir.name}")
        
        except Exception as e:
            logger.error(f"Failed to process media from {dot_dir.name}: {e}", exc_info=True)
    
    @staticmethod
    def _deduplicate_track_id(track_id: str, results: dict) -> str:
        """If track_id already exists in results, append a suffix to make it unique."""
        existing_ids = {t.get("track_id") for t in results.get("tracks", [])}
        if track_id not in existing_ids:
            return track_id
        n = 1
        while f"{track_id}_{n}" in existing_ids:
            n += 1
        deduped = f"{track_id}_{n}"
        logger.warning(f"Track {track_id} already in results, saving as {deduped}")
        return deduped

    def _load_existing_results(self, results_path: Path) -> dict:
        """Load existing results.json for incremental updates, or create a fresh structure."""
        if results_path.exists():
            try:
                with open(results_path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupt results.json, starting fresh: {results_path}")
            except Exception as e:
                logger.error(f"Cannot read results.json ({e}), starting fresh: {results_path}")
        return {
            "source_device": None,
            "processing_timestamp": None,
            "summary": {
                "total_detections": 0,
                "total_tracks": 0,
                "confirmed_tracks": 0,
                "unconfirmed_tracks": 0,
            },
            "tracks": [],
        }
    
    # ------------------------------------------------------------------
    # Last-recording marker (persists across restarts)
    # ------------------------------------------------------------------
    
    @property
    def _marker_path(self) -> Path:
        return self.input_storage / LAST_RECORDING
    
    def _load_last_recording_marker(self) -> None:
        """Read the .last_recording marker on startup."""
        if not self._marker_path.exists():
            return
        
        marker_video = self._marker_path.read_text().strip()
        if not marker_video:
            self._marker_path.unlink(missing_ok=True)
            return
        
        if (self.input_storage / marker_video).exists():
            # Video still waiting to be processed
            self._reset_after_video = marker_video
            logger.info(f"Previous session marker: will reset tracker after {marker_video}")
        else:
            # Already processed (deleted) — reset before next video
            self._pending_tracker_reset = True
            self._marker_path.unlink(missing_ok=True)
            logger.info(f"Previous session ended ({marker_video} already processed), "
                       f"tracker will reset on next video")
    
    def _save_last_recording_marker(self) -> None:
        """Write the .last_recording marker when recording stops."""
        if not (self.continuous_tracking and self.recorder
                and self.recorder.last_chunk_path):
            return
        
        filename = self.recorder.last_chunk_path.name
        self._marker_path.write_text(filename)
        self._reset_after_video = filename
        logger.info(f"Marked last recording: {filename}")
    
    def _clear_last_recording_marker(self) -> None:
        """Delete the marker after the boundary video is processed."""
        self._marker_path.unlink(missing_ok=True)
        self._reset_after_video = ""
    
    # ------------------------------------------------------------------
    # Detection Thread - Runs BugSpot detection/tracking
    # ------------------------------------------------------------------
    
    def _detection_worker(self) -> None:
        """
        Worker that runs detection/tracking for videos and queues DOT tracks.
        
        Maintains continuous tracker state for FLIK videos.
        Queues both FLIK and DOT tracks for classification.
        """
        logger.info("Detection worker started")
        
        # Process existing items in chronological order
        for path, item_type in self._find_existing_items():
            if self.stop_event.is_set():
                break
            if item_type == "video":
                self._process_video_detection(path)
            else:
                self._process_dot_media(path)
                self._process_dot_directory_detection(path)
        
        # Process new videos from queue + poll for DOT directories
        while not self.stop_event.is_set():
            try:
                video_path = self.video_queue.get(timeout=1.0)
                self._process_video_detection(video_path)
                self.video_queue.task_done()
                
                # Check for DOT directories after each video (interleaved processing)
                for dot_dir in self._find_dot_directories():
                    if self.stop_event.is_set():
                        break
                    self._process_dot_media(dot_dir)
                    self._process_dot_directory_detection(dot_dir)

            except queue.Empty:
                # Drop-folder mode: no recorder feeds the video queue, so pick
                # up injected FLIK videos ourselves. Processing deletes each
                # video, so anything found here is unprocessed. Inject with mv
                # (atomic) -- a file mid-copy could be picked up truncated.
                if self.watch_input:
                    for video in self._find_flick_videos():
                        if self.stop_event.is_set():
                            break
                        self._process_video_detection(video)

                # Check for new DOT directories while waiting
                for dot_dir in self._find_dot_directories():
                    if self.stop_event.is_set():
                        break
                    self._process_dot_media(dot_dir)
                    self._process_dot_directory_detection(dot_dir)

                # If recording stopped, check if we're done
                if self.recording_stopped.is_set():
                    remaining = self.video_queue.qsize()
                    has_ready_tracks = any(
                        self._find_ready_dot_tracks(d)
                        for d in self._find_dot_directories()
                    )
                    pending_count = self.classification_queue.count()
                    if remaining == 0 and not has_ready_tracks and pending_count == 0:
                        logger.info("Queue empty - processing complete")
                        # Nothing sets stop_event on the drain path otherwise,
                        # so the classification worker would keep the process
                        # joined forever. All queues are provably empty here
                        # (an in-flight entry still counts in pending_count).
                        self.stop_event.set()
                        break
                continue
            except Exception as e:
                logger.error(f"Detection error: {e}", exc_info=True)
        
        logger.info("Detection worker stopped")
    
    def _process_video_detection(self, video_path: Path) -> None:
        """
        Process a FLIK video: detection/tracking only, queue crops for classification.
        
        Maintains tracker state for continuous tracking across videos.
        """
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return
        
        logger.info("-" * 50)
        logger.info(f"DETECTION: {video_path.name}")
        logger.info("-" * 50)
        
        try:
            # Compute output directory: results_dir/flick_id/date_time/
            date_time = video_path.stem[len(self.flick_id) + 1:]
            output_dir = self._compute_output_dir(self.flick_id, date_time)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # --- Pre-process tracker resets (continuous_tracking only) ---
            if self.continuous_tracking:
                # Pending reset from a previous session whose marker video
                # was already processed before we started
                if self._pending_tracker_reset:
                    logger.info("Resetting tracker (previous recording session ended)")
                    self.processor.reset_tracker()
                    self._pending_tracker_reset = False
                
                # Day-change detection: stems are UTC; compare local calendar
                # dates so the reset fires overnight, not at UTC midnight
                # (mid-evening at western sites)
                video_date = local_video_date(date_time, self._local_zone)
                if self._last_video_date and video_date != self._last_video_date:
                    logger.info(f"Day changed ({self._last_video_date} → {video_date}), resetting tracker")
                    self.processor.reset_tracker()
                self._last_video_date = video_date
            
            # Run BugSpot detection/tracking (Phases 1-4)
            detection_started = time.monotonic()
            result = self.processor._pipeline.process_video(
                str(video_path),
                extract_crops=True,
                render_composites=self.processor.output_config.get("save_composites", True),
                save_crops_dir=str(output_dir / "crops"),
                save_composites_dir=str(output_dir / "composites") if self.processor.output_config.get("save_composites", True) else None,
            )
            detection_seconds = time.monotonic() - detection_started
            self.metrics.detection.record(detection_seconds)

            logger.info(f"  BugSpot: {len(result.confirmed_tracks)} confirmed / "
                       f"{len(result.track_paths)} total tracks "
                       f"({detection_seconds:.1f}s)")
            
            # Collect confirmed tracks whose crops made it to disk. Enqueueing
            # is deferred until .expected_tracks is on disk (below): the
            # classification worker can finish a fast track before detection
            # gets around to writing the count, in which case
            # _check_classification_complete silently no-ops and the directory
            # never reaches .done.
            track_timestamp = date_time.split('_')[-1] if '_' in date_time else None
            queueable_tracks = []
            for track_id, track in result.confirmed_tracks.items():
                # BugSpot saves crops using first 8 chars of track UUID
                # track_id format: {uuid}_{timestamp} -> use first 8 chars for directory
                base_track_id = track_id.split('-')[0]
                track_dir = output_dir / "crops" / base_track_id

                if not track_dir.exists():
                    logger.warning(f"Track directory not found: {track_dir}")
                    continue

                queueable_tracks.append((track_id, track, track_dir))
            confirmed_count = len(queueable_tracks)
            
            # Sample video: save 1 per N to output (0 = disabled)
            if self._video_sample_interval > 0:
                self._video_batch_count += 1
                is_last_in_batch = self._video_batch_count >= self._video_sample_interval

                should_save, reason = _video_sample_decision(
                    sample_saved=self._video_sample_saved,
                    is_last_in_batch=is_last_in_batch,
                    confirmed_count=confirmed_count,
                    random_sampling=self._random_sampling,
                )
                if should_save:
                    shutil.copy2(video_path, output_dir / "video.mp4")
                    self._video_sample_saved = True
                    logger.info(f"  Sample video saved ({reason})")

                if is_last_in_batch:
                    self._video_batch_count = 0
                    self._video_sample_saved = False
            
            # Clear detections but KEEP tracker state (continuous tracking)
            self.processor.clear_video_detections()
            
            # Delete processed video
            self._delete_video(video_path)
            
            # Recording-stop boundary: reset tracker after the last
            # video from the previous recording session
            if self._reset_after_video and video_path.name == self._reset_after_video:
                logger.info("Last recorded video processed, resetting tracker")
                self.processor.reset_tracker()
                self._clear_last_recording_marker()
            
            # Save detection metadata for classification thread to merge into results
            if confirmed_count > 0:
                # Backend parses video_timestamp as ISO-8601; date_time is the
                # compact YYYYMMDD_HHMMSS_micros video stem (matches processor.py),
                # stamped in UTC by the recorder, so carry the +00:00 offset.
                video_timestamp_iso = video_stem_utc_iso(date_time)
                if video_timestamp_iso is None:
                    logger.warning(
                        f"Video stem {date_time!r} did not parse as YYYYMMDD_HHMMSS; "
                        f"shipping video_timestamp: null for {video_path.name}"
                    )
                detection_meta = {
                    "source_device": self.flick_id,
                    "date": date_time[:8],
                    "video_file": video_path.name,
                    "video_timestamp": video_timestamp_iso,
                    "model_id": self.config.get("model", {}).get("model_id"),
                    "video_info": {
                        "fps": result.video_info.get("fps"),
                        "total_frames": result.video_info.get("total_frames"),
                        "duration_seconds": result.video_info.get("duration"),
                    } if hasattr(result, "video_info") and result.video_info else None,
                    "summary": {
                        "total_detections": len(result.all_detections) if hasattr(result, "all_detections") else 0,
                        "total_tracks": len(result.track_paths) if hasattr(result, "track_paths") else 0,
                        "confirmed_tracks": len(result.confirmed_tracks),
                        "unconfirmed_tracks": (len(result.track_paths) - len(result.confirmed_tracks)) if hasattr(result, "track_paths") else 0,
                    },
                    "tracks": {
                        tid: {
                            "num_detections": track.num_detections if hasattr(track, "num_detections") else None,
                            "first_seen_seconds": track.first_frame_time if hasattr(track, "first_frame_time") else None,
                            "last_seen_seconds": track.last_frame_time if hasattr(track, "last_frame_time") else None,
                            "duration_seconds": track.duration if hasattr(track, "duration") else None,
                            "topology_metrics": track.topology_metrics if hasattr(track, "topology_metrics") else None,
                        }
                        for tid, track in result.confirmed_tracks.items()
                    },
                    "frame_detections": {
                        track_id: [
                            {
                                "frame_number": det.get("frame_number"),
                                "timestamp_seconds": det.get("frame_time_seconds"),
                                "bbox": det.get("bbox"),
                            }
                            for det in result.all_detections
                            if det.get("track_id") == track_id
                        ]
                        for track_id in result.confirmed_tracks
                    } if hasattr(result, "all_detections") else {},
                }
                meta_path = output_dir / DETECTION_META
                meta_path.write_text(json.dumps(detection_meta, indent=2, default=str))
                
                # Write expected track count for completeness check -- must be
                # on disk before the first track is enqueued
                (output_dir / EXPECTED_TRACKS).write_text(str(confirmed_count))
                logger.info(f"  Detection metadata saved ({confirmed_count} tracks)")

                for track_id, track, track_dir in queueable_tracks:
                    self.classification_queue.enqueue(
                        entry_type="flik",
                        source_device=self.flick_id,
                        date=date_time[:8],  # YYYYMMDD
                        time=track_timestamp,
                        track_id=track_id,
                        track_dir=track_dir,
                        output_dir=output_dir,
                        num_crops=len(track.crops),
                    )
                logger.info(f"QUEUED: {confirmed_count} tracks for classification")
            else:
                # No confirmed tracks — write empty results and mark done so
                # the upload thread can discover and clean up this directory
                empty_results = {
                    "source_device": self.flick_id,
                    "date": date_time[:8],
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "summary": {
                        "total_detections": 0,
                        "total_tracks": 0,
                        "confirmed_tracks": 0,
                        "unconfirmed_tracks": 0,
                    },
                    "tracks": [],
                }
                self.writer.write_results(results=empty_results, output_dir=output_dir)
                (output_dir / DONE).write_text("classified=0\nexpected=0\n")
                logger.info("  No confirmed tracks, marked directory done")
                # Detection runs in a subprocess with no upload callback, so
                # hand the finalized dir to the main-process classification
                # worker (which owns Pollen) via the disk queue, same path as
                # tracks. Notifying from here would strand the directory --
                # zero-track dirs have no track entries, so nothing else ever
                # triggers a main-process publish for them. The health audit
                # runs at the consume site (_publish_finalized_result).
                self.classification_queue.enqueue(
                    entry_type="result",
                    source_device=self.flick_id,
                    date=date_time[:8],
                    time=track_timestamp,
                    track_id=output_dir.name,
                    track_dir=output_dir,
                    output_dir=output_dir,
                )

        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
    
    def _process_dot_directory_detection(self, dot_dir: Path) -> None:
        """
        Process DOT directory: copy crops/labels, queue for classification.
        
        Does NOT touch the tracker - DOT processing is independent.
        """
        try:
            dot_id, date_str = self._parse_dot_dir_name(dot_dir.name)
            if not dot_id:
                logger.warning(f"Could not parse DOT directory: {dot_dir.name}")
                return
            
            ready_tracks = self._find_ready_dot_tracks(dot_dir)
            if not ready_tracks:
                return
            
            logger.info("-" * 50)
            logger.info(f"DOT DETECTION: {dot_dir.name} ({len(ready_tracks)} track(s) ready)")
            logger.info("-" * 50)
            
            # Videos are handled separately by _process_dot_media (a standalone unit
            # under <dot>/<YYYYMMDD>/videos/); detection only owns the tracks.
            background = self._find_latest_background(dot_dir)

            # Each ready track becomes its own terminal result dir,
            # <dot>/<YYYYMMDD>/<track_id>_<HHMMSS>/: one results.json + .done,
            # uploaded once and deleted (no day-bucket accumulation).
            queued_count = 0
            for track_dir in ready_tracks:
                if self.stop_event.is_set():
                    break

                track_dir_name = track_dir.name
                track_id = track_dir_name.rsplit("_", 1)[0]
                track_timestamp = track_dir_name.rsplit("_", 1)[-1] if "_" in track_dir_name else None

                track_output_dir = self._compute_output_dir(dot_id, f"{date_str}/{track_dir_name}")
                track_output_dir.mkdir(parents=True, exist_ok=True)

                # Background lives in the track dir so the composite step has it after
                # the incoming DOT dir is cleaned up; the dir is terminal so it's local.
                track_background = None
                if background:
                    track_background = track_output_dir / background.name
                    shutil.copy2(background, track_background)

                # Copy crops to output
                dst_crops = track_output_dir / "crops" / track_dir_name
                dst_crops.mkdir(parents=True, exist_ok=True)

                crop_count = 0
                for f in track_dir.iterdir():
                    if f.name != "done.txt" and f.is_file():
                        shutil.copy2(f, dst_crops / f.name)
                        crop_count += 1

                # Copy label file to output
                label_src = dot_dir / "labels" / f"{track_id}.json"
                dst_labels = track_output_dir / "labels"
                dst_labels.mkdir(parents=True, exist_ok=True)
                if label_src.exists():
                    shutil.copy2(label_src, dst_labels / f"{track_id}.json")

                # One track per dir: complete-on-single, so .done fires immediately.
                # Must be on disk before the enqueue, or a fast classification
                # no-ops the completion check and the dir never reaches .done.
                (track_output_dir / EXPECTED_TRACKS).write_text("1")

                # Queue for classification. track_id stays bare: the backend
                # reconstructs crop/composite keys as {track_id}_{timestamp} and keys
                # the tracks table on (device_id, timestamp), not track_id.
                self.classification_queue.enqueue(
                    entry_type="dot",
                    source_device=dot_id,
                    date=date_str,
                    time=track_timestamp,
                    track_id=track_id,
                    track_dir=dst_crops,
                    output_dir=track_output_dir,
                    labels_path=dst_labels / f"{track_id}.json" if label_src.exists() else None,
                    background_path=track_background,
                    num_crops=crop_count,
                )
                queued_count += 1

                # Delete processed track from input
                shutil.rmtree(track_dir)
                logger.info(
                    "DOT track -> %s/%s/%s (%d crops, bare id=%s)",
                    dot_id, date_str, track_dir_name, crop_count, track_id,
                )

            logger.info(f"QUEUED: {queued_count} DOT tracks for classification")
            
            # Clean up DOT directory if empty after processing
            try:
                remaining = list(dot_dir.iterdir())
                if not remaining:
                    dot_dir.rmdir()
                    logger.info(f"Removed empty DOT directory: {dot_dir.name}")
            except OSError:
                pass
        
        except Exception as e:
            logger.error(f"Failed to process DOT {dot_dir.name}: {e}", exc_info=True)
    
    # ------------------------------------------------------------------
    # Classification Thread - Runs Hailo classification
    # ------------------------------------------------------------------
    
    def _classification_worker(self) -> None:
        """
        Worker that processes classification queue (FIFO).
        
        Classifies tracks from both FLIK and DOT sources.
        """
        logger.info("Classification worker started")
        
        # Recover any pending from crash
        self.classification_queue.recover()
        
        while not self.stop_event.is_set():
            # Nothing in this loop may propagate: an uncaught exception ends the
            # thread with the traceback going to stderr only (never the daily
            # log), and classification silently stops for the rest of the run.

            # The stale sweep lives here, not in the detection worker: in
            # subprocess mode detection has no upload callback, so its orphan
            # publish retries could never actually publish.
            try:
                self._maybe_sweep_stale_directories()
            except Exception:
                logger.error("Stale directory sweep failed", exc_info=True)

            try:
                result = self.classification_queue.get_next()
            except Exception:
                logger.error("Failed to read classification queue", exc_info=True)
                time.sleep(1.0)
                continue

            if result is None:
                time.sleep(0.5)
                continue

            filepath, entry = result

            try:
                if entry.entry_type == "video":
                    self._publish_dot_video(entry)
                elif entry.entry_type == "result":
                    self._publish_finalized_result(entry)
                else:
                    classify_started = time.monotonic()
                    if entry.entry_type == "flik":
                        self._classify_flik_track(entry)
                    else:
                        self._classify_dot_track(entry)
                    self.metrics.classification.record(time.monotonic() - classify_started)

                self.classification_queue.remove(filepath)

            except Exception as e:
                logger.error(f"Classification failed for {filepath.name}: {e}", exc_info=True)
                try:
                    should_retry = self.classification_queue.mark_failed(filepath, entry, str(e))
                    if should_retry:
                        time.sleep(1.0)
                    else:
                        # Permanently failed — still count as completed for .done check
                        self._check_classification_complete(Path(entry.output_dir))
                except Exception:
                    logger.error(f"Failed to record classification failure for {filepath.name}", exc_info=True)
                    time.sleep(1.0)
        
        logger.info("Classification worker stopped")
    
    def _classify_flik_track(self, entry: QueueEntry) -> None:
        """Classify a FLIK track from queue entry."""
        track_dir = Path(entry.track_dir)
        output_dir = Path(entry.output_dir)
        
        if not track_dir.exists():
            logger.warning(f"Track directory not found: {track_dir}")
            self._check_classification_complete(output_dir)
            return
        
        logger.info(f"CLASSIFY FLIK: {entry.track_id} ({entry.num_crops} crops)")
        
        # Load crops
        crop_files = sorted(track_dir.glob("frame_*.jpg"))
        if not crop_files:
            logger.warning(f"No crops found in {track_dir}")
            self._check_classification_complete(output_dir)
            return
        
        # Ensure classifier is initialized
        if self.processor._classifier is None:
            self.processor._classifier = HailoClassifier(self.processor.classification_config)
        
        # Classify
        classifications = []
        frames = []
        
        for crop_path in crop_files:
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            
            frame_num = int(crop_path.stem.split("_")[1])
            classification = self.processor._classifier.classify(crop)
            classifications.append(classification)
            
            frames.append({
                "frame_number": frame_num,
                "prediction": {
                    "family": classification.family,
                    "genus": classification.genus,
                    "species": classification.species,
                    "family_confidence": classification.family_confidence,
                    "genus_confidence": classification.genus_confidence,
                    "species_confidence": classification.species_confidence,
                }
            })
        
        if not classifications:
            self._check_classification_complete(output_dir)
            return
        
        # Hierarchical aggregation
        final_pred = self.processor._classifier.hierarchical_aggregate(classifications)
        if not final_pred:
            self._check_classification_complete(output_dir)
            return
        
        logger.info(f"  {final_pred['family']} / {final_pred['genus']} / {final_pred['species']} "
                   f"({final_pred['species_confidence']:.1%})")
        
        # Load existing results
        results_path = output_dir / RESULTS
        results = self._load_existing_results(results_path)
        
        # Load detection metadata to enrich results
        detection_meta = self._load_detection_meta(output_dir)
        
        # Deduplicate track_id if this is a retry after crash
        track_id = self._deduplicate_track_id(entry.track_id, results)
        
        # Enrich results with detection metadata (first track writes top-level fields)
        if detection_meta and not results.get("video_file"):
            results["video_file"] = detection_meta.get("video_file")
            results["video_timestamp"] = detection_meta.get("video_timestamp")
            results["model_id"] = detection_meta.get("model_id")
            if detection_meta.get("video_info"):
                results["video_info"] = detection_meta["video_info"]
            results["date"] = detection_meta.get("date", entry.date)
        results["source_device"] = entry.source_device
        results["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Build per-track frame data, enriched with detection metadata
        track_frames = frames
        track_meta = detection_meta.get("tracks", {}).get(entry.track_id, {}) if detection_meta else {}
        frame_dets = detection_meta.get("frame_detections", {}).get(entry.track_id, []) if detection_meta else []
        
        if detection_meta and not track_meta and not frame_dets:
            logger.warning(f"Track {entry.track_id} not found in detection metadata, enrichment skipped")
        
        if frame_dets or track_meta:
            frame_det_map = {fd["frame_number"]: fd for fd in frame_dets if fd.get("frame_number") is not None}
            enriched_frames = []
            for f in frames:
                fd = frame_det_map.get(f.get("frame_number"))
                enriched = dict(f)
                if fd:
                    if fd.get("timestamp_seconds") is not None:
                        enriched["timestamp_seconds"] = fd["timestamp_seconds"]
                    if fd.get("bbox") is not None:
                        enriched["bbox"] = fd["bbox"]
                enriched_frames.append(enriched)
            track_frames = enriched_frames
        
        # Update results
        track_result = {
            "track_id": track_id,
            "timestamp": entry.time,
            "final_prediction": final_pred,
            "num_detections": len(track_frames),
            "frames": track_frames,
        }
        if track_meta.get("first_seen_seconds") is not None:
            track_result["first_seen_seconds"] = track_meta["first_seen_seconds"]
        if track_meta.get("last_seen_seconds") is not None:
            track_result["last_seen_seconds"] = track_meta["last_seen_seconds"]
        if track_meta.get("duration_seconds") is not None:
            track_result["duration_seconds"] = track_meta["duration_seconds"]
        if track_meta.get("topology_metrics") is not None:
            track_result["topology_metrics"] = track_meta["topology_metrics"]
        
        results["tracks"].append(track_result)
        
        # Update summary: total counts from detection metadata, confirmed from actual classified tracks
        if detection_meta and "summary" in detection_meta:
            results["summary"]["total_detections"] = detection_meta["summary"].get("total_detections", 0)
            results["summary"]["total_tracks"] = detection_meta["summary"].get("total_tracks", 0)
            results["summary"]["unconfirmed_tracks"] = detection_meta["summary"].get("unconfirmed_tracks", 0)
        else:
            results["summary"]["total_detections"] = sum(t.get("num_detections", 0) for t in results["tracks"])
            results["summary"]["total_tracks"] = len(results["tracks"])
        results["summary"]["confirmed_tracks"] = len(results["tracks"])
        
        # Write results
        self.writer.write_results(results=results, output_dir=output_dir)
        
        # Check if all tracks for this output directory are done
        self._check_classification_complete(output_dir)
    
    def _classify_dot_track(self, entry: QueueEntry) -> None:
        """Classify a DOT track from queue entry."""
        track_dir = Path(entry.track_dir)
        output_dir = Path(entry.output_dir)
        
        if not track_dir.exists():
            logger.warning(f"Track directory not found: {track_dir}")
            self._check_classification_complete(output_dir)
            return
        
        logger.info(f"CLASSIFY DOT: {entry.track_id} ({entry.num_crops} crops)")
        
        # Classify using existing method
        track_result = self.processor.classify_dot_track(
            track_dir, entry.track_id, entry.time
        )
        
        if not track_result:
            self._check_classification_complete(output_dir)
            return
        
        final = track_result.get("final_prediction", {})
        logger.info(f"  {final.get('family', 'N/A')} / {final.get('genus', 'N/A')} / "
                   f"{final.get('species', 'N/A')} ({final.get('species_confidence', 0):.1%})")
        
        # Create composite if background available
        if entry.background_path:
            background_path = Path(entry.background_path)
            labels_path = Path(entry.labels_path) if entry.labels_path else None
            composite_dir = output_dir / "composites"
            composite_dir.mkdir(parents=True, exist_ok=True)
            
            track_dir_name = f"{entry.track_id}_{entry.time}" if entry.time else entry.track_id
            composite_path = composite_dir / f"{track_dir_name}.jpg"
            
            try:
                if labels_path and labels_path.exists():
                    self.processor.create_dot_composite(
                        track_dir, background_path, labels_path, composite_path
                    )
                    logger.debug("  Composite saved")
            except Exception as e:
                logger.warning(f"  Could not create composite: {e}")
        
        # Load existing results
        results_path = output_dir / RESULTS
        results = self._load_existing_results(results_path)
        
        # Deduplicate track_id if this is a retry after crash
        track_id = self._deduplicate_track_id(track_result["track_id"], results)
        track_result["track_id"] = track_id
        
        # Update results
        results["tracks"].append(track_result)
        results["source_device"] = entry.source_device
        results["date"] = entry.date
        results["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Update summary
        results["summary"]["total_tracks"] = len(results["tracks"])
        results["summary"]["confirmed_tracks"] = len(results["tracks"])
        results["summary"]["total_detections"] = sum(t.get("num_detections", 0) for t in results["tracks"])
        
        # Write results
        self.writer.write_results(results=results, output_dir=output_dir)
        
        # Check if all tracks for this output directory are done
        self._check_classification_complete(output_dir)
    
    def _delete_video(self, video_path: Path) -> None:
        """Delete processed video."""
        try:
            video_path.unlink()
            logger.debug(f"Deleted: {video_path.name}")
        except Exception as e:
            logger.error(f"Could not delete {video_path.name}: {e}")
    
    @staticmethod
    def _load_detection_meta(output_dir: Path) -> dict:
        """Load detection metadata sidecar if available."""
        meta_path = output_dir / DETECTION_META
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read detection metadata: {e}")
        return {}
    
    def _check_classification_complete(self, output_dir: Path) -> None:
        """
        Increment completed count and check if all tracks for this dir are done.

        When detection enqueues tracks, it writes .expected_tracks with the count.
        Each call to this method increments .completed_tracks. When
        completed >= expected, writes .done to signal the upload thread.
        Also called on graceful failures (missing crops, empty dirs) and
        permanent queue failures to ensure .done is always written eventually.
        """
        expected_path = output_dir / EXPECTED_TRACKS
        if not expected_path.exists():
            if (output_dir / DONE).exists():
                # Re-check after finalization (e.g. a retried entry): benign.
                logger.debug(f"Completion check on already-finalized {output_dir.name}")
            else:
                logger.warning(
                    "completion for %s cannot be tracked: no .expected_tracks and not "
                    "finalized -- this track's completion is lost (stale sweep must rescue)",
                    output_dir,
                )
            return

        try:
            expected = int(expected_path.read_text().strip())
        except (ValueError, OSError):
            logger.warning(f"Unreadable {expected_path}; completion cannot be tracked", exc_info=True)
            return

        # Atomically increment completed count
        completed_path = output_dir / COMPLETED_TRACKS
        try:
            completed = int(completed_path.read_text().strip()) + 1
        except FileNotFoundError:
            completed = 1
        except (ValueError, OSError):
            # An unreadable existing counter must not fall back to 1 -- that
            # walks the count backwards and the directory can then never reach
            # .expected_tracks. Skip the increment; the stale sweep finalizes
            # the directory once the queue drains.
            logger.warning(f"Unreadable {completed_path}; deferring to stale sweep", exc_info=True)
            return
        completed_path.write_text(str(completed))
        logger.info(f"  Classification progress: {completed}/{expected} in {output_dir.name}")

        if completed >= expected:
            done_path = output_dir / DONE
            done_path.write_text(f"classified={completed}\nexpected={expected}\n")
            logger.info(f"Classification complete: {completed}/{expected} tracks in {output_dir.name}")
            expected_path.unlink(missing_ok=True)
            completed_path.unlink(missing_ok=True)
            detection_meta_path = output_dir / DETECTION_META
            detection_meta_path.unlink(missing_ok=True)
            self._audit_finalized_dir(output_dir)
            self._notify_result_ready(output_dir)
    
    def _maybe_sweep_stale_directories(self) -> None:
        """Run the stale sweep on a wall-clock cadence, busy or idle.

        Previously gated on a count of dequeued videos, so the sweep starved
        whenever few videos arrived -- exactly the low-throughput conditions
        where stuck directories accumulate.
        """
        now = time.monotonic()
        if now - self._last_sweep_monotonic < self._sweep_interval_seconds:
            return
        self._last_sweep_monotonic = now
        self._sweep_stale_directories()

    def _sweep_stale_directories(self) -> None:
        """Clean up FLIK output directories that are stuck without .done markers.
        
        Handles two cases:
        1. Directories with results.json but no .done and no pending classification
           entries — likely a crash left them incomplete. If older than 30 minutes,
           write .done so the upload thread can pick them up.
        2. Empty directories with no results.json and no .done — created by detection
           but never populated. Remove them if older than 10 minutes.
        """

        stale_threshold_seconds = 30 * 60
        empty_threshold_seconds = 10 * 60
        orphan_threshold_seconds = 180 * 60

        # Per-pass tallies: one summary line per sweep makes stuck-state growth
        # visible in the daily log without scanning the disk by hand.
        scanned = 0
        orphans_retried = 0
        rescued = 0
        removed_empty = 0
        awaiting = 0

        try:
            for device_dir in self.results_dir.iterdir():
                if not device_dir.is_dir():
                    continue
                for output_dir in device_dir.iterdir():
                    if not output_dir.is_dir() or output_dir.name in NON_RESULT_SUBDIRS:
                        continue
                    scanned += 1

                    done_path = output_dir / DONE
                    if done_path.exists():
                        # A finalized directory that is still fully present past the
                        # threshold was never actually published (a successful
                        # enqueue_set is always followed by shutil.rmtree of this
                        # exact directory) -- e.g. no upload owner was wired at the
                        # moment it finished, or the callback raised. Otherwise
                        # invisible until someone manually finds these on disk.
                        age_seconds = (datetime.now().timestamp() - done_path.stat().st_mtime)
                        if age_seconds > orphan_threshold_seconds:
                            logger.warning(
                                "orphaned result: %s has been .done for %.0fs but was never "
                                "published (directory still on disk); retrying publish",
                                output_dir, age_seconds,
                            )
                            self._notify_result_ready(output_dir)
                            orphans_retried += 1
                        continue

                    results_path = output_dir / RESULTS
                    expected_path = output_dir / EXPECTED_TRACKS
                    
                    # Case 1: Has results.json but not marked done
                    if results_path.exists() and not expected_path.exists():
                        # No pending classification — mark done
                        age_seconds = (datetime.now().timestamp() - output_dir.stat().st_mtime)
                        if age_seconds > stale_threshold_seconds:
                            done_path.write_text("swept=stale\n")
                            logger.info(f"Swept stale directory: {output_dir.name} (no .expected_tracks, marked done)")
                            self._audit_finalized_dir(output_dir)
                            # Nothing scans for .done markers -- publishing only
                            # happens through this callback, so fire it or the
                            # rescue is a dead letter.
                            self._notify_result_ready(output_dir)
                            rescued += 1
                    
                    elif results_path.exists() and expected_path.exists():
                        # Has expected tracks but not all completed
                        # Check if all tracks are already classified
                        try:
                            expected = int(expected_path.read_text().strip())
                        except (ValueError, OSError):
                            expected = 0
                        completed_path = output_dir / COMPLETED_TRACKS
                        try:
                            completed = int(completed_path.read_text().strip())
                        except (ValueError, OSError):
                            completed = 0
                        
                        age_seconds = (datetime.now().timestamp() - output_dir.stat().st_mtime)
                        if age_seconds > stale_threshold_seconds:
                            if completed < expected:
                                # No pending or in-flight entries anywhere means the
                                # missing completions can never arrive (entry lost or
                                # corrupted); ship what exists instead of stranding it.
                                # With entries still queued, completions may yet come.
                                if self.classification_queue.count() > 0:
                                    awaiting += 1
                                    continue
                                logger.warning(
                                    "Sweeping %s with %d/%d tracks classified; "
                                    "remaining completions can no longer arrive",
                                    output_dir.name, completed, expected,
                                )
                            done_path.write_text(f"swept=stale\ncompleted={completed}\nexpected={expected}\n")
                            logger.info(f"Swept stale directory: {output_dir.name} ({completed}/{expected} completed, no .done)")
                            self._audit_finalized_dir(output_dir)
                            self._notify_result_ready(output_dir)
                            rescued += 1
                        else:
                            awaiting += 1
                    
                    # Case 2: Empty directory (no results.json, no classification activity)
                    elif not results_path.exists() and not expected_path.exists():
                        has_content = False
                        for f in output_dir.rglob("*"):
                            if f.is_file() and f.name not in SWEEP_NON_CONTENT:
                                has_content = True
                                break
                        if not has_content:
                            age_seconds = (datetime.now().timestamp() - output_dir.stat().st_mtime)
                            if age_seconds > empty_threshold_seconds:
                                shutil.rmtree(output_dir)
                                logger.info(f"Removed empty stale directory: {output_dir.name}")
                                removed_empty += 1
        except Exception as e:
            logger.warning(f"Error during stale directory sweep: {e}", exc_info=True)

        summary = (
            f"Sweep pass: {scanned} result dir(s) on disk, {awaiting} awaiting classification, "
            f"{rescued} rescued, {orphans_retried} orphan publish retries, {removed_empty} empty removed"
        )
        # Quiet at debug when there is nothing on disk and nothing happened;
        # anything else is worth a line in the daily log.
        if scanned == 0:
            logger.debug(summary)
        else:
            logger.info(summary)
    
    def _log_results_inventory(self) -> None:
        """One-shot startup scan of leftover result dirs from prior runs.

        Every dir counted here is work a previous run did not finish shipping;
        logging the split at boot dates when stuck state appeared without
        needing shell access to the device.
        """
        try:
            total = finalized_unpublished = awaiting = other = 0
            if self.results_dir.is_dir():
                for device_dir in self.results_dir.iterdir():
                    if not device_dir.is_dir():
                        continue
                    for output_dir in device_dir.iterdir():
                        if not output_dir.is_dir() or output_dir.name in NON_RESULT_SUBDIRS:
                            continue
                        total += 1
                        if (output_dir / DONE).exists():
                            finalized_unpublished += 1
                        elif (output_dir / EXPECTED_TRACKS).exists():
                            awaiting += 1
                        else:
                            other += 1
            if total == 0:
                logger.info("Startup inventory: no leftover result dirs")
            else:
                log = logger.warning if finalized_unpublished else logger.info
                log(
                    "Startup inventory: %d leftover result dir(s): %d finalized but never "
                    "published (.done), %d awaiting classification (.expected_tracks), %d other",
                    total, finalized_unpublished, awaiting, other,
                )
        except Exception:
            logger.warning("Startup inventory scan failed", exc_info=True)

    def _worker_states(self) -> list[tuple[str, object, bool]]:
        """(name, worker, required-to-be-alive) for every started worker.

        Detection exits legitimately once recording has stopped and its queues
        drain; classification only exits on stop_event, so while the pipeline
        runs it must be alive."""
        recording_active = not self.recording_stopped.is_set()
        return [
            (name, worker, required)
            for name, worker, required in (
                ("recorder", self.recorder_thread, recording_active),
                ("detection", self.detection_process or self.detection_thread, recording_active),
                ("classification", self.classification_thread, True),
            )
            if worker is not None
        ]

    def _video_queue_depth(self):
        try:
            return self.video_queue.qsize()
        except NotImplementedError:  # e.g. macOS mp queues
            return None

    def health_snapshot(self, *, reset: bool = True) -> dict:
        """Raw pipeline health for the heartbeat: worker liveness, backlog
        depths, and per-stage timing windows since the last snapshot. The
        device only measures -- thresholds and alerting live in the backend."""
        return {
            "uptime_seconds": round(time.monotonic() - self._started_monotonic, 1),
            "workers": {name: worker.is_alive() for name, worker, _ in self._worker_states()},
            "video_queue": self._video_queue_depth(),
            "classification_queue": self.classification_queue.count(),
            "detection": self.metrics.detection.snapshot(reset=reset),
            "classification": self.metrics.classification.snapshot(reset=reset),
            "unhealthy_results": self.metrics.unhealthy_results.value,
        }

    def _status_loop(self) -> None:
        """Periodic liveness/backlog line in the daily log.

        A worker that dies from an uncaught exception reports its traceback to
        stderr only -- the daily log just goes quiet. This line turns that
        silence into an explicit DEAD marker, plus queue depths for backlog
        trend analysis after the fact.
        """
        while not self.stop_event.wait(self._status_interval_seconds):
            try:
                parts = []
                dead_unexpectedly = []
                for name, worker, required in self._worker_states():
                    alive = worker.is_alive()
                    parts.append(f"{name}={'alive' if alive else 'DEAD'}")
                    if not alive and required:
                        dead_unexpectedly.append(name)
                video_backlog = self._video_queue_depth()
                line = (
                    f"STATUS: {' '.join(parts)} | video_queue={video_backlog if video_backlog is not None else -1} "
                    f"classify_queue={self.classification_queue.count()}"
                )
                if dead_unexpectedly:
                    logger.error(f"{line} -- {', '.join(dead_unexpectedly)} died; "
                                 "check stderr/journal for the traceback")
                else:
                    logger.info(line)
            except Exception:
                logger.warning("Status loop error", exc_info=True)

    # ------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE")
        logger.info("=" * 60)

        self._log_results_inventory()

        # Start recorder (if enabled)
        if self.enable_recording and self.recorder:
            self.recorder_thread = threading.Thread(
                target=self.recorder.start,
                daemon=True,
                name="Recorder"
            )
            self.recorder_thread.start()
            logger.info("Recorder thread started")
        elif self.watch_input:
            # Drop-folder mode: leave recording_stopped unset so the detection
            # loop keeps polling for injected videos instead of draining and
            # exiting; stop_recording() (e.g. Ctrl+C) ends the watch.
            logger.info("Recording disabled; watching %s for injected input", self.input_storage)
        else:
            self.recording_stopped.set()  # No recording

        # Start detection worker — an in-process thread, or a dedicated
        # subprocess (separate interpreter/GIL) when detection_in_subprocess is
        # enabled, so detection can't starve the recorder threads.
        if self.enable_processing and self.processor:
            if self.detection_in_subprocess and not self._detection_child:
                self.detection_process = self._mp_ctx.Process(
                    target=_detection_subprocess_entry,
                    args=(self.config, self.video_queue,
                          self.stop_event, self.recording_stopped, self.metrics),
                    name="DetectionProcess",
                    daemon=False,
                )
                self.detection_process.start()
                logger.info(f"Detection subprocess started (pid={self.detection_process.pid})")
            else:
                self.detection_thread = threading.Thread(
                    target=self._detection_worker,
                    daemon=False,
                    name="Detection"
                )
                self.detection_thread.start()
                logger.info("Detection thread started")
            
            # Start classification worker
            if self.enable_classification:
                self.classification_thread = threading.Thread(
                    target=self._classification_worker,
                    daemon=False,
                    name="Classification"
                )
                self.classification_thread.start()
                logger.info("Classification thread started")
        
        self._status_thread = threading.Thread(
            target=self._status_loop,
            daemon=True,
            name="PipelineStatus",
        )
        self._status_thread.start()

        if self.enable_recording and self.enable_processing:
            logger.info("Pipeline running - Ctrl+C to stop recording (processing continues)")
        elif self.enable_recording:
            logger.info("Recording - Ctrl+C to stop")
        else:
            logger.info("Processing existing videos...")
    
    def stop_recording(self) -> None:
        """Stop recording only, processing continues."""
        if not self.recording_stopped.is_set():
            logger.info("=" * 60)
            logger.info("STOPPING RECORDING")
            logger.info("=" * 60)
            
            if self.recorder:
                self.recorder.stop()
            if self.recorder_thread:
                self.recorder_thread.join(timeout=10.0)
            
            # Mark the last recorded video so tracker resets after it
            self._save_last_recording_marker()
            
            self.recording_stopped.set()
            logger.info("Recording stopped - processing remaining videos...")
            
            remaining = self.video_queue.qsize()
            if remaining > 0:
                logger.info(f"Videos in queue: {remaining}")
            
            pending = self.classification_queue.count()
            if pending > 0:
                logger.info(f"Pending classifications: {pending}")
    
    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        logger.info("=" * 60)
        logger.info("STOPPING PIPELINE")
        logger.info("=" * 60)
        
        # Stop recorder first
        self.stop_recording()
        
        # Stop threads / detection subprocess
        self.stop_event.set()
        
        if self.detection_process:
            self.detection_process.join(timeout=30.0)
            if self.detection_process.is_alive():
                logger.warning("Detection subprocess did not exit in time; terminating")
                self.detection_process.terminate()
                self.detection_process.join(timeout=5.0)
            logger.info("Detection subprocess stopped")
        
        if self.detection_thread:
            self.detection_thread.join(timeout=30.0)
            logger.info("Detection thread stopped")
        
        if self.classification_thread:
            self.classification_thread.join(timeout=30.0)
            logger.info("Classification thread stopped")
        
        logger.info("Pipeline stopped cleanly")
    
    def wait(self) -> None:
        """Wait for pipeline (blocks until stopped)."""
        # Wait for recorder to finish (if running)
        if self.recorder_thread:
            self.recorder_thread.join()
        
        # Wait for detection subprocess to finish (if running)
        if self.detection_process:
            self.detection_process.join()
        
        # Wait for detection thread to finish (if running)
        if self.detection_thread:
            self.detection_thread.join()
        
        # Wait for classification thread to finish (if running)
        if self.classification_thread:
            self.classification_thread.join()


def _detection_subprocess_entry(config, video_queue, stop_event, recording_stopped, metrics=None):
    """Spawned-subprocess entrypoint for the detection loop.

    Runs in its own interpreter (own GIL) so the GIL-heavy detection work cannot
    starve the recorder threads in the parent process. Builds a detection-only
    ``Pipeline`` that shares the recorder's video queue and stop/recording
    events, then runs the detection worker loop. All outputs (crops, composites,
    and the disk-based classification queue) are written to disk exactly as in
    the in-process path, so detection results are unchanged.
    """
    try:
        setup_logging(Path(config["paths"]["logs_dir"]))
    except Exception:
        logging.basicConfig(level=logging.INFO)
    logger.info("Detection subprocess starting")
    detector = Pipeline(
        config,
        detection_child=True,
        shared_video_queue=video_queue,
        shared_stop_event=stop_event,
        shared_recording_stopped=recording_stopped,
        shared_metrics=metrics,
    )
    try:
        detector._detection_worker()
    finally:
        logger.info("Detection subprocess exiting")
