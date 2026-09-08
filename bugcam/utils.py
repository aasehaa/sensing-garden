import subprocess
import platform
from rich.console import Console
from rich.table import Table
from .settings import get_python_for_detection

console = Console()

def print_success(message: str):
    console.print(f"[green]✓[/green] {message}")

def print_error(message: str):
    console.print(f"[red]✗[/red] {message}")

def print_info(message: str):
    console.print(f"[blue]ℹ[/blue] {message}")

def create_table(title: str, columns: list[str]) -> Table:
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    return table

def require_linux() -> bool:
    """Check if running on Linux. Returns True on Linux, prints error and returns False otherwise."""
    if platform.system() != "Linux":
        console.print("[red]This command only works on Linux systems[/red]")
        return False
    return True

def preflight_check(python_exe: str | None = None) -> bool:
    """Check if detection dependencies are available in the Python interpreter."""
    if platform.system() != "Linux":
        return True  # Can't check on non-Linux
    try:
        python_exe = python_exe or get_python_for_detection()
        result = subprocess.run(
            [python_exe, "-c", "import gi, hailo, hailo_apps, numpy, cv2"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False

def handle_numpy_error(console: Console) -> None:
    """Print helpful message for NumPy binary incompatibility errors."""
    console.print("\n[red]NumPy binary incompatibility detected.[/red]")
    console.print("This usually happens when system packages are out of sync.\n")
    console.print("Fix with: [cyan]sudo apt install --reinstall python3-numpy python3-picamera2 python3-libcamera python3-simplejpeg[/cyan]\n")
    console.print("If that doesn't work, also remove any pip numpy:")
    console.print("[cyan]rm -rf ~/.local/lib/python*/site-packages/numpy*[/cyan]\n")
    console.print("Then run: [cyan]bugcam check camera[/cyan] to verify the fix.")

def handle_hailo_lib_error(console: Console) -> None:
    """Print helpful message for missing Hailo post-process library."""
    console.print("\n[red]Missing Hailo post-processing libraries.[/red]")
    console.print("Run [cyan]bugcam setup[/cyan] to compile the required libraries.\n")
