"""All-in-one record, process, upload, and heartbeat command."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from bugcam import __version__
from bugcam.capture_report import CAPTURES_SUBDIR, DEFAULT_REPORT_INTERVAL_SECONDS, CaptureLog
from bugcam.commands.heartbeat import write_heartbeat_snapshot
from bugcam.pollen.integration import build_pollen
from bugcam.edge26.result_publish import publish_result_dir
from bugcam.pollen.transport import DEFAULT_MULTIPART_THRESHOLD, DEFAULT_PART_SIZE
from bugcam.settings import (
    DEFAULT_API_URL,
    DEFAULT_S3_BUCKET,
    get_input_storage_dir,
    get_output_storage_dir,
    get_state_dir,
    load_config,
    log_startup_config,
    parse_dot_ids,
    resolve_flick_id,
)
from bugcam.commands.status import _check_time_sync
from bugcam.environment_sensor import collect_environment_reading
from bugcam.processing import parse_capture_resolution
from bugcam.record_window import RecordingWindow
from bugcam.runtime import build_pipeline, resolve_bundle_provenance, select_model_reference
from bugcam.receiver import create_app
from bugcam.receiver.config import RECEIVER_DEFAULT_PORT, RECEIVER_DEFAULT_HOST
from bugcam.receiver.tracker import PendingTrackTracker

app = typer.Typer(help="Record, process, upload, and emit heartbeats", invoke_without_command=True, no_args_is_help=False)
console = Console()
HEARTBEAT_INTERVAL_SECONDS = 300
DEFAULT_VIDEO_SAMPLE_INTERVAL = 10
ENVIRONMENT_INTERVAL_SECONDS = 60
PID_FILE_PATH = get_state_dir() / "bugcam.pid"
logger = logging.getLogger(__name__)


def _emit_heartbeat(
    flick_id: str,
    input_dir: Path,
    output_dir: Path,
    dot_ids: list[str],
    pollen: Any = None,
    pipeline: Any = None,
    timezone_name: str | None = None,
) -> None:
    """Write one enriched heartbeat and ship it.

    Enrichment (pipeline health, upload stats) is best-effort -- a broken
    section must never cost the heartbeat itself. Delivery is an immediate
    PUT so heartbeats never wait on the spool/batch cadence, falling back to
    the durable queue (priority kind, never archived) when the PUT fails."""
    pipeline_status = upload_status = None
    if pipeline is not None:
        try:
            pipeline_status = pipeline.health_snapshot()
        except Exception:
            logger.exception("pipeline health snapshot failed; heartbeat continues without it")
    if pollen is not None:
        try:
            upload_status = pollen.stats()
        except Exception:
            logger.exception("upload stats failed; heartbeat continues without them")
    path = write_heartbeat_snapshot(
        output_dir, flick_id, input_dir, dot_ids,
        timezone_name=timezone_name,
        pipeline_status=pipeline_status, upload_status=upload_status,
    )
    if pollen is None:
        return
    try:
        pollen.upload_path_now(path)
        path.unlink(missing_ok=True)
        return
    except Exception as exc:
        logger.warning(
            "immediate heartbeat upload failed (%s: %s); spooling %s",
            type(exc).__name__, exc, path.name,
        )
    try:
        pollen.enqueue_set([path], device=flick_id, kind="heartbeat")  # staged; our copy is done
        path.unlink(missing_ok=True)
    except Exception:
        logger.exception("heartbeat enqueue failed (will retry next interval): %s", path.name)


def _heartbeat_loop(
    flick_id: str,
    input_dir: Path,
    output_dir: Path,
    dot_ids: list[str],
    stop_event: threading.Event,
    pollen: Any = None,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    timezone_name: str | None = None,
    pipeline: Any = None,
) -> None:
    while not stop_event.is_set():
        try:
            _emit_heartbeat(
                flick_id, input_dir, output_dir, dot_ids,
                pollen=pollen, pipeline=pipeline, timezone_name=timezone_name,
            )
        except Exception:
            logger.exception("heartbeat emission failed (will retry next interval)")
        stop_event.wait(interval)


def _capture_report_loop(
    capture_log: CaptureLog,
    flick_id: str,
    stop_event: threading.Event,
    pollen: Any = None,
    interval: float = DEFAULT_REPORT_INTERVAL_SECONDS,
) -> None:
    """Seal and ship sampling-effort reports every ``interval`` seconds.

    Reports ride the normal upload path (kind="capture" under the device's data
    tree), so with --archive-batch each one lands in the per-device tar next to
    the results it describes. Without pollen, reports accumulate locally, like
    heartbeats."""

    def _ship(report_path: Path) -> None:
        if pollen is None:
            return
        try:
            pollen.enqueue_set([report_path], device=flick_id, kind="capture")  # staged; copy done
            report_path.unlink(missing_ok=True)
        except Exception:
            logger.exception(
                "capture report enqueue failed (kept; recovered next pass): %s", report_path.name
            )

    # A prior run's unsealed samples and unshipped reports go out first.
    for leftover in capture_log.recover():
        _ship(leftover)
    while not stop_event.wait(interval):
        _ship(capture_log.rotate())


def _environment_loop(
    flick_id: str,
    output_dir: Path,
    stop_event: threading.Event,
    pollen: Any = None,
) -> None:
    warning_emitted = False
    while not stop_event.is_set():
        try:
            path, _payload = collect_environment_reading(output_dir=output_dir, flick_id=flick_id)
            if pollen is not None:
                pollen.enqueue_set([path], device=flick_id, kind="environment")  # staged; copy done
                path.unlink(missing_ok=True)
            warning_emitted = False
        except Exception as exc:
            if not warning_emitted:
                console.print(f"[yellow]Environment sensor warning[/yellow] {exc}")
                logger.warning("environment sensor/enqueue warning: %s: %s", type(exc).__name__, exc)
                warning_emitted = True
        stop_event.wait(ENVIRONMENT_INTERVAL_SECONDS)


def _receiver_loop(
    host: str,
    port: int,
    stop_event: threading.Event,
) -> None:
    """Run the Flask receiver server in a thread."""
    flask_app = create_app(config={"host": host, "port": port})
    tracker = flask_app.config.get("TRACKER")

    if tracker:
        logger.info("Scanning for orphaned tracks...")
        tracker.recover_orphaned_tracks()

        finalization_stop = threading.Event()
        finalization_thread = threading.Thread(
            target=_finalization_loop,
            args=(tracker, finalization_stop),
            daemon=True
        )
        finalization_thread.start()
        logger.info("Track finalization thread started for receiver")

    logger.info(f"Receiver starting on {host}:{port}")
    flask_app.run(host=host, port=port, threaded=True, debug=False)

    if tracker and finalization_thread:
        finalization_stop.set()
        finalization_thread.join(timeout=5)


def _finalization_loop(tracker: PendingTrackTracker, stop_event: threading.Event):
    """Background thread that checks for idle tracks to finalize."""
    while not stop_event.is_set():
        try:
            tracker.check_pending()
        except Exception as e:
            logger.error(f"Finalization loop error: {e}")
        stop_event.wait(PendingTrackTracker.CHECK_INTERVAL)


def _parse_resolution_option(value: str) -> tuple[int, int]:
    try:
        return parse_capture_resolution(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_runtime_settings(
    api_url: str | None,
    api_key: str | None,
    flick_id: str | None,
    dot_ids: str | None,
    bucket: str | None,
    *,
    enable_upload: bool = True,
) -> dict[str, Any]:
    config = load_config()
    resolved_api_url = api_url or str(config.get("api_url") or DEFAULT_API_URL)
    resolved_api_key = api_key or str(config.get("api_key") or "")
    resolved_flick_id = resolve_flick_id(flick_id)
    resolved_dot_ids = parse_dot_ids(dot_ids) if dot_ids is not None else parse_dot_ids(config.get("dot_ids"))
    resolved_bucket = bucket or str(config.get("s3_bucket") or DEFAULT_S3_BUCKET)

    missing_fields = [
        field_name
        for field_name, value in (
            ("api_key", resolved_api_key),
            ("s3_bucket", resolved_bucket),
        )
        if not value
    ]
    if missing_fields and enable_upload:
        joined = ", ".join(missing_fields)
        raise typer.BadParameter(f"Missing required config values: {joined}. Run `bugcam setup` or pass CLI flags.")
    if not resolved_flick_id:
        raise typer.BadParameter("Missing flick_id. Run `bugcam setup` or pass --flick-id.")

    return {
        "api_url": resolved_api_url.rstrip("/"),
        "api_key": resolved_api_key,
        "flick_id": resolved_flick_id,
        "dot_ids": resolved_dot_ids,
        "s3_bucket": resolved_bucket,
    }


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_bugcam_process(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return "bugcam" in result.stdout.lower()


def _acquire_pid_file() -> Path:
    PID_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PID_FILE_PATH.exists():
        raw_pid = PID_FILE_PATH.read_text(encoding="utf-8").strip()
        if raw_pid.isdigit():
            pid = int(raw_pid)
            if _process_is_running(pid) and _is_bugcam_process(pid):
                raise RuntimeError(f"BugCam is already running (PID {raw_pid}). Use `kill {raw_pid}` to stop it first.")
        PID_FILE_PATH.unlink(missing_ok=True)

    PID_FILE_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return PID_FILE_PATH


def _release_pid_file(pid_path: Path) -> None:
    if pid_path.exists():
        pid_path.unlink(missing_ok=True)


def _resolve_pollen_settings(archive_batch: bool | None, upload_poll: int) -> dict[str, Any]:
    """Resolve upload settings: CLI flag wins, then the config file, then default.

    Config-file keys: archive_batch, upload_poll_interval,
    upload_multipart_threshold, upload_part_size.
    """
    cfg = load_config()
    return {
        "batch": archive_batch if archive_batch is not None else bool(cfg.get("archive_batch", False)),
        "poll_interval": float(cfg.get("upload_poll_interval", upload_poll)),
        "multipart_threshold": int(cfg.get("upload_multipart_threshold", DEFAULT_MULTIPART_THRESHOLD)),
        "part_size": int(cfg.get("upload_part_size", DEFAULT_PART_SIZE)),
    }


def _resolve_heartbeat_interval(heartbeat_interval: float | None) -> float:
    """Resolve the heartbeat cadence: CLI flag wins, then config, then default."""
    if heartbeat_interval is not None:
        return float(heartbeat_interval)
    return float(load_config().get("heartbeat_interval", HEARTBEAT_INTERVAL_SECONDS))


def _resolve_recording_window(
    timezone_name: str | None, record_window: str | None
) -> tuple[str | None, str | None]:
    """Resolve timezone and record window: CLI flag wins, then config.

    Config-file keys: timezone (IANA name), record_window ("HH:MM-HH:MM" local
    or "always").
    """
    cfg = load_config()
    resolved_tz = timezone_name or str(cfg.get("timezone") or "") or None
    resolved_window = record_window or str(cfg.get("record_window") or "") or None
    return resolved_tz, resolved_window


def _resolve_capture_report_interval() -> float:
    """Sampling-report cadence (config: capture_report_interval). The default
    matches the default --upload-poll, so one report per archive batch."""
    return float(load_config().get("capture_report_interval", DEFAULT_REPORT_INTERVAL_SECONDS))


def _resolve_video_sample_interval(video_sample_interval: int | None) -> int:
    """Resolve the FLIK sample-video cadence: CLI flag wins, then config, then
    default (config: video_sample_interval, default 10). Saves 1 processed
    video per N to the upload queue; 0 disables sample-video saving entirely."""
    resolved = video_sample_interval if video_sample_interval is not None else int(
        load_config().get("video_sample_interval", DEFAULT_VIDEO_SAMPLE_INTERVAL)
    )
    if resolved < 0:
        raise typer.BadParameter("video_sample_interval must be >= 0")
    return resolved


@app.callback()
def run(
    api_url: str | None = typer.Option(None, "--api-url", help="Backend API URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="Per-device API key"),
    flick_id: str | None = typer.Option(None, "--flick-id", help="FLICK device ID"),
    dot_ids: str | None = typer.Option(None, "--dot-ids", help="Comma-separated DOT IDs"),
    input_dir: Path = typer.Option(get_input_storage_dir(), "--input-dir", help="Directory for recorded input"),
    output_dir: Path = typer.Option(get_output_storage_dir(), "--output-dir", help="Directory for processed output"),
    model: str | None = typer.Option(None, "--model", help="Model bundle name or model.hef path"),
    mode: str = typer.Option("continuous", "--mode", help="'continuous' (always recording) or 'interval' (record periodically)"),
    interval: int = typer.Option(5, "--interval", help="Minutes between recordings in interval mode"),
    chunk_duration: int = typer.Option(60, "--chunk-duration", help="Length of each recorded chunk in seconds"),
    fps: int = typer.Option(30, "--fps", help="Recording frame rate"),
    resolution: str = typer.Option("1080x1080", "--resolution", help="Recording resolution in WxH format"),
    bitrate: int = typer.Option(20_000_000, "--bitrate", help="H.264 encoder bitrate in bps (hardware encoding only)"),
    bucket: str | None = typer.Option(None, "--bucket", help="Configured output bucket"),
    upload_poll: int = typer.Option(3600, "--upload-poll", help="Seconds between upload polls; with --archive-batch this is also the batch cadence (one tar per device per poll) (config: upload_poll_interval)"),
    heartbeat_interval: float | None = typer.Option(None, "--heartbeat-interval", help="Seconds between heartbeat snapshots (config: heartbeat_interval, default 300)"),
    video_sample_interval: int | None = typer.Option(None, "--video-sample-interval", help="Save 1 processed video per N to the upload queue; 0 disables sample-video saving (config: video_sample_interval, default 10)"),
    random_sampling: bool = typer.Option(
        False,
        "--random-sampling",
        help="Disable the confirmed-track priority in sample-video selection: always upload the "
             "batch's actual Nth video (per --video-sample-interval), not whichever one in the "
             "batch had confirmed tracks",
    ),
    with_receiver: bool = typer.Option(
        True,
        "--with-receiver/--no-receiver",
        help="Start DOT receiver server alongside pipeline",
    ),
    receiver_port: int = typer.Option(RECEIVER_DEFAULT_PORT, "--receiver-port", help="DOT receiver HTTP port"),
    receiver_host: str = typer.Option(RECEIVER_DEFAULT_HOST, "--receiver-host", help="DOT receiver bind address"),
    detection_config: Path | None = typer.Option(None, "--detection-config", help="Path to detection config YAML file"),
    detection_in_subprocess: bool = typer.Option(
        True,
        "--detection-in-subprocess/--detection-in-thread",
        help="Run detection in a separate process (own GIL) so it can't starve the recorder threads (default: on)",
    ),
    delete_after_upload: bool = typer.Option(
        True,
        "--delete-after-upload/--no-delete-after-upload",
        help="Delete local files after they are uploaded (default: on)",
    ),
    enable_upload: bool = typer.Option(
        True,
        "--upload/--no-upload",
        help="Upload results to S3; --no-upload disables all uploads (default: on)",
    ),
    archive_batch: bool | None = typer.Option(
        None,
        "--archive-batch/--no-archive-batch",
        help="Bundle outputs into per-device tar archives instead of per-object uploads; cadence follows --upload-poll (config: archive_batch)",
    ),
    record: bool = typer.Option(
        True,
        "--record/--no-record",
        help="Record from the camera; --no-record watches --input-dir as a drop "
             "folder instead, processing injected videos/DOT dirs (inject with mv, "
             "not cp: a file mid-copy can be picked up truncated)",
    ),
    timezone_name: str | None = typer.Option(
        None,
        "--timezone",
        help="IANA timezone of the deployment site, e.g. Europe/London or "
             "America/Toronto (config: timezone); localizes the record window "
             "and day boundaries. Timestamps stay UTC.",
    ),
    record_window: str | None = typer.Option(
        None,
        "--record-window",
        help="Daily local-time recording window 'HH:MM-HH:MM', or 'always' "
             "(config: record_window). With a timezone configured the default "
             "is 05:00-22:00; processing/uploading continues outside the window.",
    ),
) -> None:
    """Run recording, processing, uploading, and periodic heartbeat emission."""
    if mode not in {"continuous", "interval"}:
        raise typer.BadParameter("mode must be 'continuous' or 'interval'")
    resolved_timezone, resolved_window = _resolve_recording_window(timezone_name, record_window)
    try:
        # Validate early so a config typo fails at launch, not mid-run.
        window = RecordingWindow.from_config(resolved_window, resolved_timezone)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    resolved_video_sample_interval = _resolve_video_sample_interval(video_sample_interval)
    try:
        pid_path = _acquire_pid_file()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    parsed_resolution = _parse_resolution_option(resolution)

    try:
        ntp_ok, ntp_detail = _check_time_sync()
        if not ntp_ok:
            console.print(f"[yellow]Warning[/yellow] {ntp_detail}")

        settings = _resolve_runtime_settings(api_url, api_key, flick_id, dot_ids, bucket, enable_upload=enable_upload)
        resolved_heartbeat_interval = _resolve_heartbeat_interval(heartbeat_interval)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pollen is the upload subsystem; it owns all device -> S3 traffic.
        # --no-upload disables it entirely (no device -> S3 traffic at all).
        pollen_instance = None
        if enable_upload:
            pollen_settings = _resolve_pollen_settings(archive_batch, upload_poll)
            pollen_instance = build_pollen(
                output_dir,
                settings["api_url"],
                settings["api_key"],
                state_dir=get_state_dir(),
                poll_interval=pollen_settings["poll_interval"],
                multipart_threshold=pollen_settings["multipart_threshold"],
                part_size=pollen_settings["part_size"],
                batch=pollen_settings["batch"],
                delete_after_upload=delete_after_upload,
            )
            console.print(f"[dim]Upload[/dim] enabled (batch={pollen_settings['batch']})")
            logger.info(
                "upload enabled (batch=%s, poll_interval=%s)",
                pollen_settings["batch"], pollen_settings["poll_interval"],
            )

            # The manifest is a fixed-key object the queue does not own; ship it once
            # at startup. A missing manifest is non-fatal -- the backend resolves
            # devices from each result's source_device.
            # TODO: this writes to a shared v1/manifest.json (matches master); confirm a
            # single shared key is right vs a device-specific one -- multiple FLIK devices
            # on one bucket would overwrite each other.
            try:
                manifest = json.dumps(
                    {
                        "flick_id": settings["flick_id"],
                        "dot_ids": settings["dot_ids"],
                        "timezone": resolved_timezone,
                    },
                    indent=2,
                ).encode("utf-8")
                pollen_instance.upload_now(manifest, "v1/manifest.json", "application/json")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Manifest upload failed[/yellow] {exc} (non-fatal)")
                logger.warning("manifest upload failed (non-fatal): %s: %s", type(exc).__name__, exc)
        else:
            console.print("[yellow]Uploads disabled[/yellow] (--no-upload)")
            logger.warning("uploads disabled for this run (--no-upload) -- no artifact will reach S3")

        selected_model = select_model_reference(model)
        provenance = resolve_bundle_provenance(selected_model)
        if model is None:
            console.print(f"[dim]Using model[/dim] {provenance['model_id']}")
        console.print(f"[cyan]Running[/cyan] flick={settings['flick_id']} dots={settings['dot_ids'] or '[]'}")
        console.print(f"[dim]Model[/dim] {provenance['model_id']}")
        if not record:
            console.print(f"[yellow]Recording disabled[/yellow] (--no-record); watching {input_dir} for injected input")
            logger.info("recording disabled for this run (--no-record); processing injected input only")
        else:
            console.print(f"[dim]Recording window[/dim] {window.describe()}")
            logger.info("recording window: %s", window.describe())

        on_result_ready = None
        on_log_complete = None
        on_video_ready = None
        if pollen_instance is not None:
            on_result_ready = lambda d: publish_result_dir(  # noqa: E731 - produce-site adapter
                pollen_instance, d, settings["flick_id"], settings["dot_ids"]
            )
            on_video_ready = lambda v, dev: pollen_instance.enqueue_set(  # noqa: E731 - produce-site adapter
                [v], device=dev, kind="video"
            )

            def on_log_complete(path: Path) -> None:  # ship a rolled-over log, then drop our copy
                pollen_instance.enqueue_set([path], device=settings["flick_id"], kind="log")
                path.unlink(missing_ok=True)

        # Sampling-effort tracking only makes sense when the camera runs;
        # injected input (--no-record) is not sampling.
        capture_log = None
        if record:
            capture_log = CaptureLog(
                output_dir / settings["flick_id"] / CAPTURES_SUBDIR, settings["flick_id"]
            )
        pipeline = build_pipeline(
            flick_id=settings["flick_id"],
            dot_ids=settings["dot_ids"],
            input_dir=input_dir,
            output_dir=output_dir,
            model_reference=selected_model,
            recording_mode=mode,
            recording_interval=interval,
            chunk_duration=chunk_duration,
            fps=fps,
            resolution=parsed_resolution,
            bitrate=bitrate,
            detection_in_subprocess=detection_in_subprocess,
            enable_recording=record,
            watch_input=not record,
            detection_config_path=detection_config,
            timezone_name=resolved_timezone,
            record_window=resolved_window,
            video_sample_interval=resolved_video_sample_interval,
            random_sampling=random_sampling,
            on_result_ready=on_result_ready,
            on_log_complete=on_log_complete,
            on_video_ready=on_video_ready,
            on_chunk_recorded=capture_log.record_chunk if capture_log else None,
        )
        # setup_logging() ran inside build_pipeline(), so file+console handlers
        # are live here -- this is the earliest point the full startup config
        # (device config + detection.yaml/defaults + commit) can reach the log.
        log_startup_config(
            version=__version__,
            device_config=settings,
            detection_config=pipeline.config.get("detection", {}),
            tracking_config=pipeline.config.get("tracking", {}),
            detection_config_source=pipeline.config.get("detection_config_source", "unknown"),
        )
        heartbeat_stop_event = threading.Event()
        environment_stop_event = threading.Event()
        capture_report_stop_event = threading.Event()
        receiver_stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(
                settings["flick_id"],
                input_dir,
                output_dir,
                settings["dot_ids"],
                heartbeat_stop_event,
                pollen_instance,
                resolved_heartbeat_interval,
                resolved_timezone,
                pipeline,
            ),
            daemon=True,
            name="BugCamHeartbeat",
        )
        environment_thread = threading.Thread(
            target=_environment_loop,
            args=(
                settings["flick_id"],
                output_dir,
                environment_stop_event,
                pollen_instance,
            ),
            daemon=True,
            name="BugCamEnvironment",
        )
        capture_report_thread = None
        if capture_log is not None:
            capture_report_thread = threading.Thread(
                target=_capture_report_loop,
                args=(
                    capture_log,
                    settings["flick_id"],
                    capture_report_stop_event,
                    pollen_instance,
                    _resolve_capture_report_interval(),
                ),
                daemon=True,
                name="BugCamCaptureReport",
            )

        receiver_thread = None
        if with_receiver:
            receiver_thread = threading.Thread(
                target=_receiver_loop,
                args=(receiver_host, receiver_port, receiver_stop_event),
                daemon=True,
                name="BugCamReceiver",
            )

        if pollen_instance is not None:
            pollen_instance.start()
        pipeline.start()
        heartbeat_thread.start()
        environment_thread.start()
        if capture_report_thread:
            capture_report_thread.start()
        if receiver_thread:
            receiver_thread.start()
            console.print(f"[dim]Receiver[/dim] http://{receiver_host}:{receiver_port}")

        pipeline.wait()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping recording and draining remaining work[/yellow]")
        pipeline.stop_recording()
        pipeline.wait()
    finally:
        if "heartbeat_stop_event" in locals():
            heartbeat_stop_event.set()
        if "environment_stop_event" in locals():
            environment_stop_event.set()
        if "capture_report_stop_event" in locals():
            capture_report_stop_event.set()
        if "receiver_stop_event" in locals() and receiver_thread:
            receiver_stop_event.set()
        if "heartbeat_thread" in locals():
            heartbeat_thread.join(timeout=1)
        if "environment_thread" in locals():
            environment_thread.join(timeout=1)
        if "capture_report_thread" in locals() and capture_report_thread:
            capture_report_thread.join(timeout=1)
        if "receiver_thread" in locals() and receiver_thread:
            receiver_thread.join(timeout=5)
        # Stop Pollen's loop after finishing the current tick; anything still
        # queued is durable in SQLite and resumes on the next launch.
        if "pollen_instance" in locals() and pollen_instance is not None:
            pollen_instance.stop()
        _release_pid_file(pid_path)
