"""System status and diagnostics for bugcam."""
import importlib.util
import typer
import subprocess
import platform
from pathlib import Path
from rich.console import Console
from ..settings import (
    get_edge26_labels_path,
    get_edge26_model_path,
    get_input_storage_dir,
    get_output_storage_dir,
    get_python_for_detection,
    load_config,
    parse_dot_ids,
    is_edge26_classification_enabled,
)
from ..model_bundles import get_installed_bundles

app = typer.Typer(help="Check system status and dependencies")
console = Console()
I2C_SENSOR_MAP = {
    "61": "SCD30",
    "62": "SCD40",
    "69": "SEN55",
    "76": "BME280",
    "77": "BME280",
}


# --- Check Functions ---

def _check_python_import(import_name: str) -> bool:
    """Check if a package is importable in the detection Python interpreter."""
    if platform.system() != "Linux":
        return False
    try:
        python_exe = get_python_for_detection()
        result = subprocess.run(
            [python_exe, "-c", f"import {import_name}"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_hailo_device() -> tuple[bool, str]:
    """Test Hailo AI accelerator connection."""
    try:
        result = subprocess.run(
            ["hailortcli", "scan"],
            capture_output=True,
            timeout=10
        )
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        if "not found" in stdout.lower() or "not found" in stderr.lower():
            return False, "No device found"

        if result.returncode == 0 and "Hailo" in stdout:
            for line in stdout.strip().split("\n"):
                if "Hailo" in line and "not found" not in line.lower():
                    return True, line.strip()
            return True, "Detected"

        return False, "Check failed"
    except FileNotFoundError:
        return False, "hailortcli not installed"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:50]


def _check_camera() -> tuple[bool, str]:
    """Test camera connection."""
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-c", "from picamera2 import Picamera2; Picamera2()"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "Accessible"

        stderr = result.stderr.decode()
        if "dtype size changed" in stderr or "binary incompatibility" in stderr:
            return False, "NumPy incompatibility"
        return False, "Not accessible"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:50]


def _check_sensor() -> tuple[bool, str]:
    """Test I2C sensor connection."""
    i2c_device = Path("/dev/i2c-1")
    if not i2c_device.exists():
        return False, "Sensor connection not enabled. Enable: sudo raspi-config → Interface Options → I2C"

    i2cdetect_paths = ["i2cdetect", "/usr/sbin/i2cdetect"]
    for i2cdetect_bin in i2cdetect_paths:
        try:
            result = subprocess.run(
                [i2cdetect_bin, "-y", "1"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                continue

            output = result.stdout.decode()
            sensors: list[str] = []
            for address, sensor_name in I2C_SENSOR_MAP.items():
                if address in output and sensor_name not in sensors:
                    sensors.append(sensor_name)

            if sensors:
                return True, ", ".join(sensors)
            return False, "I2C enabled, no sensors detected"
        except FileNotFoundError:
            continue
        except Exception as e:
            return False, str(e)[:50]

    return False, "Sensor detection tool missing. Install: sudo apt install i2c-tools"


def _check_time_sync() -> tuple[bool, str]:
    """Check whether the system clock is synchronized."""
    if platform.system() != "Linux":
        return False, "Skipped (Linux only)"
    try:
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        synced = "NTPSynchronized=yes" in result.stdout
        if synced:
            return True, "Clock synchronized"
        return False, "System clock not synced — check internet connection"
    except FileNotFoundError:
        return False, "timedatectl not available"
    except subprocess.TimeoutExpired:
        return False, "timedatectl timeout"
    except Exception as exc:
        return False, str(exc)[:50]


def _check_models() -> tuple[bool, str]:
    """Check for installed models."""
    bundles = get_installed_bundles(require_labels=True)
    if bundles:
        return True, f"{len(bundles)} installed"
    return False, "None installed"


def _check_storage_paths() -> tuple[bool, str]:
    """Check the configured input and output storage directories."""
    input_dir = get_input_storage_dir()
    output_dir = get_output_storage_dir()
    paths_ok = input_dir.exists() and output_dir.exists()
    detail = f"input={input_dir}, output={output_dir}"
    return paths_ok, detail


def _check_edge26_processor() -> tuple[bool, str]:
    """Check pipeline readiness."""
    classification_enabled = is_edge26_classification_enabled()
    details = [f"classification={'on' if classification_enabled else 'off'}"]
    ok = True
    bugspot_ok = importlib.util.find_spec("bugspot") is not None
    details.append(f"bugspot={'ok' if bugspot_ok else 'missing'}")
    ok = ok and bugspot_ok

    if classification_enabled:
        model_path = get_edge26_model_path()
        model_ok = model_path.exists()
        details.append(f"model={'ok' if model_ok else 'missing'}")
        ok = ok and model_ok

        labels_path = get_edge26_labels_path(model_path)
        labels_ok = labels_path.exists()
        details.append(f"labels={'ok' if labels_ok else 'missing'}")
        ok = ok and labels_ok

        for module_name in ("requests", "hailo_platform"):
            module_ok = importlib.util.find_spec(module_name) is not None
            details.append(f"{module_name}={'ok' if module_ok else 'missing'}")
            ok = ok and module_ok

    return ok, ", ".join(details)


def _print_status(name: str, ok: bool, detail: str) -> None:
    """Print a status row."""
    status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {name:.<20} {status}  {detail}")


def _display_api_url(api_url: str) -> str:
    return api_url.removeprefix("https://").removeprefix("http://")


def _print_device_row(name: str, value: str) -> None:
    console.print(f"  {name:.<20} {value}")


def _print_device_section() -> bool:
    config = load_config()
    flick_id = str(config.get("flick_id") or config.get("device_id") or "").strip()
    dot_ids = parse_dot_ids(config.get("dot_ids"))
    api_url = str(config.get("api_url") or "").strip()
    registered = bool(flick_id and api_url and str(config.get("api_key") or "").strip())

    console.print("\n[cyan]Device[/cyan]")
    if not registered:
        console.print("  Not registered. Run bugcam setup")
        return False

    _print_device_row("ID", flick_id)
    _print_device_row("DOTs", ", ".join(dot_ids) if dot_ids else "none")
    _print_device_row("API", _display_api_url(api_url))
    _print_device_row("Registered", "yes")
    return True


# --- Subcommands ---

@app.command()
def deps() -> None:
    """Check software dependencies."""
    console.print("\n[bold cyan]Dependencies[/bold cyan]")

    if platform.system() != "Linux":
        console.print("[yellow]  Skipped (Linux only)[/yellow]\n")
        return

    all_ok = True
    packages = [
        ("gi", "python3-gi"),
        ("hailo", "hailo-all"),
        ("hailo_apps", "bugcam setup"),
        ("numpy", "python3-numpy"),
        ("cv2", "python3-opencv"),
    ]

    for import_name, install_hint in packages:
        ok = _check_python_import(import_name)
        if not ok:
            all_ok = False
        detail = "" if ok else f"Install: {install_hint}"
        _print_status(import_name, ok, detail)

    console.print()
    raise typer.Exit(0 if all_ok else 1)


@app.command()
def devices() -> None:
    """Check hardware device connections."""
    console.print("\n[bold cyan]Devices[/bold cyan]")

    if platform.system() != "Linux":
        console.print("[yellow]  Skipped (Linux only)[/yellow]\n")
        return

    all_ok = True

    hailo_ok, hailo_detail = _check_hailo_device()
    if not hailo_ok:
        all_ok = False
    _print_status("Hailo", hailo_ok, hailo_detail)

    camera_ok, camera_detail = _check_camera()
    if not camera_ok:
        all_ok = False
    _print_status("Camera", camera_ok, camera_detail)

    sensor_ok, sensor_detail = _check_sensor()
    if not sensor_ok:
        all_ok = False
    _print_status("Sensors", sensor_ok, sensor_detail)

    console.print()
    raise typer.Exit(0 if all_ok else 1)


@app.command()
def hailo() -> None:
    """Check Hailo AI accelerator."""
    console.print("\n[bold cyan]Hailo Check[/bold cyan]")

    if platform.system() != "Linux":
        console.print("[yellow]  Skipped (Linux only)[/yellow]\n")
        return

    ok, detail = _check_hailo_device()
    if ok:
        console.print(f"[green]  ✓ {detail}[/green]")
    else:
        console.print(f"[red]  ✗ {detail}[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Check AI HAT+ is properly seated")
        console.print("  2. Enable PCIe Gen 3: [cyan]sudo raspi-config[/cyan] → Advanced → PCIe")
        console.print("  3. Reinstall: [cyan]sudo apt install --reinstall hailo-all[/cyan]")
        console.print("  4. Reboot: [cyan]sudo reboot[/cyan]")

    console.print()
    raise typer.Exit(0 if ok else 1)


@app.command()
def camera() -> None:
    """Check camera connection."""
    console.print("\n[bold cyan]Camera Check[/bold cyan]")

    if platform.system() != "Linux":
        console.print("[yellow]  Skipped (Linux only)[/yellow]\n")
        return

    ok, detail = _check_camera()
    if ok:
        console.print(f"[green]  ✓ {detail}[/green]")
    else:
        console.print(f"[red]  ✗ {detail}[/red]")
        if "NumPy" in detail:
            console.print("\n[yellow]Fix:[/yellow]")
            console.print("  [cyan]sudo apt install --reinstall python3-numpy python3-picamera2 python3-libcamera[/cyan]")

    console.print()
    raise typer.Exit(0 if ok else 1)


@app.command()
def sensor() -> None:
    """Check I2C sensor connection."""
    console.print("\n[bold cyan]Sensor Check[/bold cyan]")

    if platform.system() != "Linux":
        console.print("[yellow]  Skipped (Linux only)[/yellow]\n")
        return

    ok, detail = _check_sensor()
    if ok:
        console.print(f"[green]  ✓ {detail}[/green]")
    else:
        console.print(f"[red]  ✗ {detail}[/red]")

    console.print()
    raise typer.Exit(0 if ok else 1)


@app.command()
def models() -> None:
    """Check installed models."""
    console.print("\n[bold cyan]Models[/bold cyan]")

    ok, detail = _check_models()
    if ok:
        console.print(f"[green]  ✓ {detail}[/green]")
    else:
        console.print(f"[red]  ✗ {detail}[/red]")
        console.print("\n[yellow]Download:[/yellow] [cyan]bugcam models download yolov8s[/cyan]")

    console.print()
    raise typer.Exit(0 if ok else 1)


@app.command()
def storage() -> None:
    """Check storage directories and pipeline readiness."""
    console.print("\n[bold cyan]Storage[/bold cyan]")
    queue_ok, queue_detail = _check_storage_paths()
    runtime_ok, runtime_detail = _check_edge26_processor()
    _print_status("Paths", queue_ok, queue_detail)
    _print_status("Pipeline", runtime_ok, runtime_detail)
    console.print()
    raise typer.Exit(0 if queue_ok and runtime_ok else 1)


@app.command()
def time() -> None:
    """Check system clock synchronization."""
    console.print("\n[bold cyan]Clock[/bold cyan]")
    ok, detail = _check_time_sync()
    _print_status("Clock", ok, detail)
    console.print()
    raise typer.Exit(0 if ok else 1)


@app.callback(invoke_without_command=True)
def status(ctx: typer.Context) -> None:
    """Run all system checks."""
    if ctx.invoked_subcommand is not None:
        return

    console.print("\n[bold]bugcam system status[/bold]")

    all_ok = True

    device_ok = _print_device_section()
    if not device_ok:
        all_ok = False

    # Dependencies
    console.print("\n[cyan]Dependencies[/cyan]")
    if platform.system() != "Linux":
        console.print("  [dim]Skipped (Linux only)[/dim]")
    else:
        for import_name, _ in [("gi", ""), ("hailo", ""), ("hailo_apps", ""), ("numpy", ""), ("cv2", "")]:
            ok = _check_python_import(import_name)
            if not ok:
                all_ok = False
            _print_status(import_name, ok, "")

    # Models
    console.print("\n[cyan]Models[/cyan]")
    models_ok, models_detail = _check_models()
    if not models_ok:
        all_ok = False
    _print_status("Bundles", models_ok, models_detail)

    # Storage
    console.print("\n[cyan]Storage[/cyan]")
    storage_ok, storage_detail = _check_storage_paths()
    if not storage_ok:
        all_ok = False
    _print_status("Paths", storage_ok, storage_detail)
    runtime_ok, runtime_detail = _check_edge26_processor()
    if not runtime_ok:
        all_ok = False
    _print_status("Pipeline", runtime_ok, runtime_detail)

    console.print("\n[cyan]Clock[/cyan]")
    time_ok, time_detail = _check_time_sync()
    if not time_ok:
        all_ok = False
    _print_status("Clock", time_ok, time_detail)

    # Devices
    console.print("\n[cyan]Devices[/cyan]")
    if platform.system() != "Linux":
        console.print("  [dim]Skipped (Linux only)[/dim]")
    else:
        hailo_ok, hailo_detail = _check_hailo_device()
        if not hailo_ok:
            all_ok = False
        _print_status("Hailo", hailo_ok, hailo_detail)

        camera_ok, camera_detail = _check_camera()
        if not camera_ok:
            all_ok = False
        _print_status("Camera", camera_ok, camera_detail)

        sensor_ok, sensor_detail = _check_sensor()
        if not sensor_ok:
            all_ok = False
        _print_status("Sensors", sensor_ok, sensor_detail)

    console.print()
    if all_ok:
        console.print("[green]All checks passed![/green]\n")
    else:
        console.print("[yellow]Some checks failed. Run subcommands for details:[/yellow]")
        console.print("  [cyan]bugcam status deps[/cyan]    - Software dependencies")
        console.print("  [cyan]bugcam status devices[/cyan] - Hardware connections")
        console.print("  [cyan]bugcam status models[/cyan]  - Installed models")
        console.print("  [cyan]bugcam status storage[/cyan] - Input/output paths\n")
        console.print("  [cyan]bugcam status time[/cyan]    - Clock synchronization\n")

    raise typer.Exit(0 if all_ok else 1)
