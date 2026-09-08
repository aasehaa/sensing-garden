"""Setup command for BugCam."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
import typer
from rich.console import Console

from ..settings import (
    DEFAULT_API_URL,
    DEFAULT_S3_BUCKET,
    build_dot_ids,
    get_hailo_venv_dir,
    get_python_for_detection,
    get_state_dir,
    load_config,
    load_device_config,
    save_config,
)

app = typer.Typer(help="Install dependencies")
console = Console()

# Updated for Hailo AI v5
HAILO_RPI5_EXAMPLES_URL = "https://github.com/hailo-ai/hailo-apps.git"
HAILO_APPS_INFRA_URL = "git+https://github.com/hailo-ai/hailo-apps.git"
SEN55_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sensors" / "sen55"


def check_import(python_exe: str, module: str) -> bool:
    """Check if a module can be imported."""
    try:
        result = subprocess.run(
            [python_exe, "-c", f"import {module}"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_hailo_platform_available() -> bool:
    """Check if hailo_platform is importable via system Python."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import hailo_platform"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _create_hailo_venv_with_system_packages(venv_dir: Path) -> None:
    """Create a venv that can access system site-packages via .pth file."""
    if venv_dir.exists():
        console.print(f"[dim]Removing existing {venv_dir}...[/dim]")
        shutil.rmtree(venv_dir)

    console.print(f"[cyan]Creating Hailo venv at {venv_dir}...[/cyan]")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_command([sys.executable, "-m", "venv", str(venv_dir)], timeout=60)

    # Add .pth file to access system packages (where hailo_platform is installed)
    site_packages = list(venv_dir.glob("lib/python*/site-packages"))[0]
    pth_file = site_packages / "system-packages.pth"
    system_packages_path = "/usr/lib/python3/dist-packages"
    pth_file.write_text(system_packages_path + "\n")
    console.print(f"[green]Created {pth_file} pointing to {system_packages_path}[/green]")


def _install_hailo_apps(python_exe: str) -> None:
    """Install hailo-apps in the venv."""
    console.print("[cyan]Installing hailo-apps...[/cyan]")
    cmd = [python_exe, "-m", "pip", "install", HAILO_APPS_INFRA_URL]
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    _run_command(cmd, timeout=300)


def _run_command(cmd: list[str], *, cwd: str | None = None, timeout: int) -> None:
    result = subprocess.run(cmd, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _install_hailo_environment() -> None:
    """Set up Hailo environment using system packages (TAPPAS 5.x compatible)."""
    hailo_venv_dir = get_hailo_venv_dir()
    python_exe = get_python_for_detection()

    # Check if already properly set up
    if check_import(python_exe, "hailo_platform") and check_import(python_exe, "hailo_apps"):
        console.print("[green]Hailo setup already complete[/green]\n")
        return

    # If system has hailo_platform, create venv with system packages access
    if _is_hailo_platform_available():
        console.print("[green]hailo_platform found in system Python.[/green]")
        console.print("[cyan]Creating Hailo venv with system packages access...[/cyan]")

        _create_hailo_venv_with_system_packages(hailo_venv_dir)
        python_exe = get_python_for_detection()

        # Install hailo_apps in the venv
        if not check_import(python_exe, "hailo_apps"):
            _install_hailo_apps(python_exe)

        # Verify both packages are importable
        if check_import(python_exe, "hailo_platform") and check_import(python_exe, "hailo_apps"):
            console.print("[green]Hailo setup complete[/green]\n")
            return
        else:
            console.print("[yellow]Warning: hailo_platform or hailo_apps still not importable[/yellow]")

    # Fallback: Try the old method with install script (for TAPPAS 3.x systems)
    console.print("[yellow]System packages method failed. Trying install script...[/yellow]")

    if not hailo_venv_dir.exists():
        temp_clone_dir = Path("/tmp/hailo-rpi5-examples-setup")
        if temp_clone_dir.exists():
            console.print("[yellow]Removing existing temp directory...[/yellow]")
            shutil.rmtree(temp_clone_dir)

        console.print("[cyan]Cloning hailo-apps (shallow clone)...[/cyan]")
        console.print(f"[dim]$ git clone --depth 1 {HAILO_RPI5_EXAMPLES_URL} {temp_clone_dir}[/dim]\n")
        _run_command(["git", "clone", "--depth", "1", HAILO_RPI5_EXAMPLES_URL, str(temp_clone_dir)], timeout=120)
        console.print("[green]Clone complete.[/green]\n")

        install_script = temp_clone_dir / "install.sh"
        if not install_script.exists():
            raise FileNotFoundError(f"install.sh not found at {install_script}")

        console.print("[cyan]Running install script with --no-tappas-required...[/cyan]")
        console.print(f"[dim]$ cd {temp_clone_dir} && ./install.sh --no-tappas-required[/dim]\n")
        try:
            _run_command(["sudo", "./install.sh", "--no-tappas-required"], cwd=str(temp_clone_dir), timeout=600)
        except RuntimeError:
            # Try without sudo as fallback
            _run_command(["./install.sh", "--no-tappas-required"], cwd=str(temp_clone_dir), timeout=600)
        console.print("[green]Install script complete.[/green]\n")

        temp_venv_dir = temp_clone_dir / "venv_hailo_apps"
        if not temp_venv_dir.exists():
            temp_venv_dir = temp_clone_dir / "venv_hailo_rpi_examples"
            if not temp_venv_dir.exists():
                raise FileNotFoundError(f"venv not found at {temp_clone_dir}/venv_hailo_*")

        console.print(f"[cyan]Moving Hailo environment to {hailo_venv_dir}...[/cyan]")
        hailo_venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if hailo_venv_dir.exists():
            shutil.rmtree(hailo_venv_dir)
        shutil.move(str(temp_venv_dir), str(hailo_venv_dir))
        console.print("[green]Hailo environment moved.[/green]\n")

        if temp_clone_dir.exists():
            shutil.rmtree(temp_clone_dir)

    python_exe = get_python_for_detection()
    if not check_import(python_exe, "hailo_apps"):
        raise RuntimeError("hailo_apps installation verification failed")
    console.print("[green]Hailo setup complete[/green]\n")


def _existing_flick_id(existing_config: dict[str, Any]) -> str:
    # existing_config is always the same dict load_config() would return here
    # (see call sites), so this matches load_device_config().flick_id exactly.
    return load_device_config().flick_id


def _existing_dot_count(existing_config: dict[str, Any]) -> int:
    dot_ids = existing_config.get("dot_ids")
    if isinstance(dot_ids, list):
        return len(dot_ids)
    return 0


def _detect_external_drives() -> list[str]:
    """Detect mounted external drives under /media/pi/."""
    drives = []
    try:
        result = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "/media/pi/" in line:
                parts = line.split()
                if len(parts) >= 3:
                    mount_point = parts[2]
                    # Check for common external drive filesystems
                    if any(fs in line.lower() for fs in ["vfat", "exfat", "ntfs", "ext4", "xfs"]):
                        drives.append(mount_point)
        # Remove duplicates while preserving order
        seen = set()
        drives = [x for x in drives if not (x in seen or seen.add(x))]
    except Exception:
        pass
    return drives


def _prompt_storage_paths(existing_config: dict[str, Any], external_drives: list[str]) -> dict[str, str]:
    """Prompt user for storage paths, offering external drives if found."""
    # State dir ALWAYS stays on local filesystem for proper permissions
    default_state = str(existing_config.get("state_dir") or "~/.local/share/bugcam")

    # Storage dirs can be on external drive
    default_input = str(existing_config.get("input_dir") or str(Path.home() / "bugcam" / "incoming"))
    default_output = str(existing_config.get("output_dir") or str(Path.home() / "bugcam" / "outputs"))
    default_pending = str(existing_config.get("pending_dir") or str(Path.home() / "bugcam" / "pending"))

    # Ask about external drive ONLY for storage dirs (input/output/pending)
    if external_drives:
        if len(external_drives) == 1:
            drive = external_drives[0]
            console.print(f"\n[cyan]External drive detected:[/cyan] {drive}")
            if typer.confirm("Use external drive for video storage?", default=True):
                base = f"{drive}/bugcam"
                default_input = base + "/incoming"
                default_output = base + "/outputs"
                default_pending = base + "/pending"
        else:
            console.print("\n[cyan]External drives detected:[/cyan]")
            for i, drive in enumerate(external_drives, 1):
                console.print(f"  {i}. {drive}")
            if typer.confirm("Use an external drive for video storage?", default=True):
                if len(external_drives) == 1:
                    drive = external_drives[0]
                else:
                    choice = typer.prompt("Select drive (enter number)", type=int, default=1)
                    drive = external_drives[choice - 1]
                base = f"{drive}/bugcam"
                default_input = base + "/incoming"
                default_output = base + "/outputs"
                default_pending = base + "/pending"

    console.print("\n[dim]Storage paths (press Enter for defaults):[/dim]")
    console.print("[yellow]Note: State directory should stay on local filesystem for proper permissions[/yellow]")
    state_dir = typer.prompt("State directory", default=default_state)
    input_dir = typer.prompt("Input directory", default=default_input)
    output_dir = typer.prompt("Output directory", default=default_output)
    pending_dir = typer.prompt("Pending directory", default=default_pending)

    return {
        "state_dir": state_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "pending_dir": pending_dir,
    }


def _prompt_registration_settings(existing_config: dict[str, Any]) -> dict[str, Any]:
    external_drives = _detect_external_drives()
    storage_paths = _prompt_storage_paths(existing_config, external_drives)

    return {
        "api_url": typer.prompt("API URL", default=str(existing_config.get("api_url") or DEFAULT_API_URL)),
        "flick_id": typer.prompt("Device ID (unique name for this device)", default=_existing_flick_id(existing_config)),
        "dot_count": typer.prompt("Number of DOT sensors (0 if none)", default=_existing_dot_count(existing_config), type=int),
        "timezone": _prompt_timezone(existing_config),
        **storage_paths,
    }


def _prompt_timezone(existing_config: dict[str, Any]) -> str:
    """Prompt for the deployment site's IANA timezone, validated locally.

    Blank keeps the device recording around the clock; a timezone enables the
    default daytime recording window and localizes day boundaries.
    """
    from zoneinfo import ZoneInfo

    while True:
        value = typer.prompt(
            "Timezone (IANA name, e.g. Europe/London or America/Toronto; blank = record 24/7)",
            default=str(existing_config.get("timezone") or ""),
        ).strip()
        if not value:
            return ""
        try:
            ZoneInfo(value)
            return value
        except Exception:
            console.print(
                f"[red]Unknown timezone: {value}[/red] "
                "Use an IANA name like Europe/London, Europe/Amsterdam, or America/Toronto."
            )


def _register_device(api_url: str, setup_code: str, flick_id: str, dot_count: int) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/devices/register",
            json={"setup_code": setup_code, "flick_id": flick_id, "dot_count": dot_count},
            timeout=30,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise ValueError("Cannot reach the BugCam server. Check your WiFi connection.") from exc
    except requests.Timeout as exc:
        raise ValueError("Cannot reach the BugCam server. The request timed out.") from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        if status_code in {401, 403}:
            raise ValueError("Invalid setup code. Check that you copied it correctly.") from exc
        error_message = _extract_registration_error(exc.response)
        raise ValueError(f"Registration failed: {status_code} — {error_message}") from exc

    payload = response.json()
    required_fields = ("device_id", "api_key", "flick_id", "dot_ids")
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        joined = ", ".join(missing_fields)
        raise ValueError(f"Registration response missing fields: {joined}")
    return payload


def _extract_registration_error(response: requests.Response | None) -> str:
    if response is None:
        return "Unknown server error"
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "Unknown server error"
    message = payload.get("error") if isinstance(payload, dict) else None
    return str(message or response.text.strip() or "Unknown server error")


def _should_reregister(existing_config: dict[str, Any], flick_id: str) -> bool:
    existing_api_key = str(existing_config.get("api_key") or "")
    existing_flick_id = _existing_flick_id(existing_config)
    if not existing_api_key or not existing_flick_id:
        return True
    if flick_id != existing_flick_id:
        return True
    return typer.confirm(f"Already registered as {existing_flick_id}. Re-register?", default=False)


def _build_saved_config(
    existing_config: dict[str, Any],
    api_url: str,
    api_key: str,
    flick_id: str,
    dot_ids: list[str],
    input_dir: str,
    output_dir: str,
    state_dir: str,
    pending_dir: str,
    timezone_name: str = "",
) -> dict[str, Any]:
    preserved = {
        key: value
        for key, value in existing_config.items()
        if key not in {"api_url", "api_key", "device_id", "device_name", "flick_id", "dot_ids", "s3_bucket",
                      "input_dir", "output_dir", "state_dir", "pending_dir", "timezone"}
    }
    return {
        **preserved,
        "api_url": api_url.rstrip("/"),
        "api_key": api_key,
        "flick_id": flick_id,
        "dot_ids": dot_ids,
        "s3_bucket": str(existing_config.get("s3_bucket") or DEFAULT_S3_BUCKET),
        "input_dir": input_dir,
        "output_dir": output_dir,
        "state_dir": state_dir,
        "pending_dir": pending_dir,
        # Blank clears the key: no timezone means record around the clock.
        **({"timezone": timezone_name} if timezone_name else {}),
    }


def _run_status_check() -> None:
    cmd = [sys.executable, "-m", "bugcam.cli", "status"]
    console.print("[cyan]Running bugcam status...[/cyan]")
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]\n")
    result = subprocess.run(cmd, timeout=60)
    if result.returncode != 0:
        console.print("[yellow]Status check reported issues. Review the output above.[/yellow]")


def _install_sen55_binary() -> None:
    """Compile and install the bundled SEN55 helper binary without blocking setup on failure."""
    binary_dir = get_state_dir() / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    try:
        console.print("[cyan]Compiling SEN55 reader...[/cyan]")
        _run_command(["make"], cwd=str(SEN55_SOURCE_DIR), timeout=120)
        shutil.copy2(SEN55_SOURCE_DIR / "sen55_reader", binary_dir / "sen55_reader")
        console.print(f"[green]Installed SEN55 reader to {binary_dir / 'sen55_reader'}[/green]\n")
    except Exception as exc:
        console.print(f"[yellow]SEN55 reader compilation skipped:[/yellow] {exc}\n")


def _create_storage_dirs(config: dict[str, Any]) -> None:
    """Create storage directories if they don't exist."""
    dirs_to_create = [
        config.get("state_dir"),
        config.get("input_dir"),
        config.get("output_dir"),
        config.get("pending_dir"),
    ]
    for dir_path in dirs_to_create:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    console.print("[green]Storage directories created.[/green]\n")


def _print_registration_summary(flick_id: str, dot_ids: list[str]) -> None:
    console.print(f"Registered device: [cyan]{flick_id}[/cyan]")
    if not dot_ids:
        console.print()
        return
    console.print("DOT sensors:")
    for dot_id in dot_ids:
        console.print(f"  [cyan]{dot_id}[/cyan]")
    console.print("Share these IDs with your DOT sensor operators.\n")


@app.callback(invoke_without_command=True)
def setup() -> None:
    """Install Hailo dependencies, register the device, and save local config."""
    if platform.system() != "Linux":
        console.print("[yellow]Note: bugcam detection only works on Raspberry Pi (Linux)[/yellow]")
        console.print(f"Current platform: {platform.system()}")
        raise typer.Exit(1)

    try:
        _install_hailo_environment()
        existing_config = load_config()
        did_register = False
        settings = _prompt_registration_settings(existing_config)
        if settings["dot_count"] < 0:
            raise ValueError("Number of dots must be >= 0")

        if _should_reregister(existing_config, settings["flick_id"]):
            setup_code = typer.prompt("Setup code (from your project lead)", hide_input=True)
            registration = _register_device(
                settings["api_url"],
                setup_code,
                settings["flick_id"],
                settings["dot_count"],
            )
            api_key = str(registration["api_key"])
            flick_id = str(registration["flick_id"])
            dot_ids = [str(dot_id) for dot_id in registration["dot_ids"]]
            did_register = True
        else:
            api_key = str(existing_config.get("api_key") or "")
            flick_id = settings["flick_id"]
            dot_ids = build_dot_ids(flick_id, settings["dot_count"])

        saved_config = _build_saved_config(
            existing_config=existing_config,
            api_url=settings["api_url"],
            api_key=api_key,
            flick_id=flick_id,
            dot_ids=dot_ids,
            input_dir=settings["input_dir"],
            output_dir=settings["output_dir"],
            state_dir=settings["state_dir"],
            pending_dir=settings["pending_dir"],
            timezone_name=settings["timezone"],
        )
        save_config(saved_config)
        console.print(f"[green]Saved config for {saved_config['flick_id']}[/green]\n")

        _create_storage_dirs(saved_config)

        if did_register:
            _print_registration_summary(flick_id, dot_ids)
        _install_sen55_binary()
        _run_status_check()
        console.print("[green]Setup complete![/green]\n")
        console.print("[bold]Next steps[/bold]")
        console.print("[cyan]bugcam models list[/cyan]")
        console.print("[cyan]bugcam models download <name>[/cyan]")
        console.print("[cyan]bugcam run --model <name>[/cyan]")
        console.print(f"[cyan]bugcam autostart enable --model <name> --bucket {saved_config['s3_bucket']}[/cyan]")
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
