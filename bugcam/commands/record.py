"""Video recording command for bugcam."""
import typer
import time
import platform
import subprocess
import os
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console
from typing import Optional
from ..settings import get_input_storage_dir, resolve_flick_id
from . import parse_resolution_option

app = typer.Typer(help="Record videos from camera")
console = Console()

# Default output directory
DEFAULT_OUTPUT_DIR = get_input_storage_dir()


def _build_recording_path(output_dir: Path, flick_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{flick_id}_{timestamp}.mp4"


def _check_disk_space(output_dir: Path, min_free_mb: int = 300) -> tuple[bool, int]:
    """Check if output directory has sufficient free disk space.

    Returns tuple of (has_space, free_mb).
    """
    try:
        usage = shutil.disk_usage(output_dir)
        free_mb = usage.free // (1024 * 1024)
        return free_mb >= min_free_mb, free_mb
    except Exception:
        # If we can't check, assume it's OK
        return True, -1


def _check_camera_available() -> bool:
    if platform.system() != "Linux":
        return True  # Can't check on non-Linux
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-c", "from picamera2 import Picamera2; Picamera2()"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def _remux_video(path: Path) -> bool:
    if not _check_ffmpeg_available():
        console.print("[yellow]ffmpeg not found, skipping remux[/yellow]")
        return True

    tmp_path = path.with_suffix('.tmp.mp4')
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c", "copy", str(tmp_path)],
            check=True,
            capture_output=True
        )
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        console.print(f"[yellow]Remux failed: {e}[/yellow]")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def _record_single_video(output_path: Path, length: int, quiet: bool, resolution: tuple[int, int]) -> bool:
    # Import here to avoid import errors on non-Pi systems
    try:
        from picamera2 import Picamera2
        from picamera2.encoders import H264Encoder
    except ImportError:
        console.print("[red]picamera2 not available. Run on Raspberry Pi.[/red]")
        return False

    try:
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"format": 'RGB888', "size": resolution}
        )
        picam2.configure(camera_config)

        # Try to set autofocus controls (only available on Camera Module 3)
        try:
            picam2.set_controls({"AfMode": 0, "LensPosition": 0.5})
        except Exception:
            # Camera doesn't support autofocus (e.g., Camera Module 2, HQ Camera)
            pass

        picam2.start()

        encoder = H264Encoder(bitrate=10000000)  # 10 Mbps

        if not quiet:
            console.print(f"[cyan]Recording {length}s video...[/cyan]", end=" ")

        picam2.start_recording(encoder, str(output_path))
        time.sleep(length)
        picam2.stop_recording()
        picam2.close()

        if not quiet:
            console.print("[green]done[/green]")

        return True

    except Exception as e:
        console.print(f"[red]Recording failed: {e}[/red]")
        return False


@app.command()
def single(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    length: int = typer.Option(60, "--length", "-l", help="Length of video in seconds"),
    flick_id: Optional[str] = typer.Option(None, "--flick-id", help="FLICK device ID for generated filenames"),
    resolution: str = typer.Option("1080x1080", "--resolution", help="Recording resolution in WxH format"),
) -> None:
    """Record a single video.

    Records one video and exits. Useful for testing.
    """
    if platform.system() != "Linux":
        console.print("[red]Recording only works on Raspberry Pi (Linux)[/red]")
        raise typer.Exit(1)

    if not _check_camera_available():
        console.print("[red]Camera not accessible[/red]")
        raise typer.Exit(1)
    parsed_resolution = parse_resolution_option(resolution)
    resolved_flick_id = resolve_flick_id(flick_id)

    # Generate output path if not specified
    if output is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = _build_recording_path(DEFAULT_OUTPUT_DIR, resolved_flick_id)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    # Check disk space before recording
    has_space, free_mb = _check_disk_space(output.parent)
    if not has_space:
        console.print(f"[red]Insufficient disk space. Need at least 300MB free, have {free_mb}MB.[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Recording {length}s video to {output}[/cyan]")

    if _record_single_video(output, length, quiet=False, resolution=parsed_resolution):
        _remux_video(output)
        console.print(f"[green]Video saved: {output}[/green]")
    else:
        raise typer.Exit(1)
