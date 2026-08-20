"""Tests for bugcam setup command."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
from typer.testing import CliRunner
from bugcam.cli import app


def test_create_storage_dirs_expands_tilde(tmp_path: Path, monkeypatch) -> None:
    """A literal '~' from the setup prompt must resolve to $HOME."""
    from bugcam.commands.setup import _create_storage_dirs

    monkeypatch.setenv("HOME", str(tmp_path))
    _create_storage_dirs({"state_dir": "~/data/bugcam"})
    assert (tmp_path / "data" / "bugcam").is_dir()
    assert not (tmp_path / "~").exists()


def test_setup_help(cli_runner: CliRunner) -> None:
    """Test setup command help."""
    result = cli_runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0


def test_setup_fails_on_non_linux(cli_runner: CliRunner) -> None:
    """Test setup exits with error on non-Linux platforms."""
    with patch('bugcam.commands.setup.platform.system', return_value='Darwin'):
        result = cli_runner.invoke(app, ["setup"])
        assert result.exit_code == 1
        assert "linux" in result.output.lower() or "raspberry pi" in result.output.lower()


def test_setup_detects_hailo_venv_python(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test get_python_for_detection prefers hailo venv if available."""
    from bugcam.config import get_python_for_detection

    # Create fake hailo venv
    hailo_python = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv" / "bin" / "python"
    hailo_python.parent.mkdir(parents=True)
    hailo_python.touch()

    with patch.object(Path, 'home', return_value=tmp_path):
        python = get_python_for_detection()
        assert python == str(hailo_python)


def test_setup_falls_back_to_system_python(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test get_python_for_detection falls back to /usr/bin/python3 on Linux."""
    from bugcam.config import get_python_for_detection

    with patch('bugcam.config.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path):
        # Hailo venv doesn't exist, falls back to system python
        python = get_python_for_detection()
        assert python == "/usr/bin/python3"


def test_setup_uses_sys_executable_fallback(cli_runner: CliRunner) -> None:
    """Test get_python_for_detection uses sys.executable as final fallback."""
    from bugcam.config import get_python_for_detection
    import sys

    with patch('bugcam.config.platform.system', return_value='Darwin'), \
         patch('pathlib.Path.is_file', return_value=False):
        python = get_python_for_detection()
        assert python == sys.executable


def test_check_import_success(cli_runner: CliRunner) -> None:
    """Test check_import returns True when import succeeds."""
    from bugcam.commands.setup import check_import

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch('subprocess.run', return_value=mock_result) as mock_run:
        result = check_import("/usr/bin/python3", "hailo_apps")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["/usr/bin/python3", "-c", "import hailo_apps"]


def test_check_import_failure(cli_runner: CliRunner) -> None:
    """Test check_import returns False when import fails."""
    from bugcam.commands.setup import check_import

    mock_result = MagicMock()
    mock_result.returncode = 1

    with patch('subprocess.run', return_value=mock_result):
        result = check_import("/usr/bin/python3", "nonexistent_module")
        assert result is False


def test_check_import_handles_exception(cli_runner: CliRunner) -> None:
    """Test check_import returns False on exception."""
    from bugcam.commands.setup import check_import

    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("test", 10)):
        result = check_import("/usr/bin/python3", "hailo_apps")
        assert result is False


@pytest.mark.xfail(reason="SG-027: setup tests drifted from setup.py 'already complete' flow", strict=False)
def test_setup_clones_repo_if_not_exists(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test setup clones hailo-rpi5-examples to temp directory if venv doesn't exist."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    # Create fake venv structure in temp location
    temp_clone_dir = Path("/tmp/hailo-rpi5-examples-setup")
    temp_venv_dir = temp_clone_dir / "venv_hailo_rpi_examples"
    hailo_venv_dir = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv"

    # Mock Path.exists to control which paths exist
    original_exists = Path.exists
    def mock_path_exists(self: Path) -> bool:
        if self == hailo_venv_dir:
            return False  # Venv doesn't exist yet
        if self == temp_clone_dir:
            return False  # Temp dir doesn't exist before clone
        if self == temp_venv_dir:
            return True  # Venv created by install.sh
        if str(self).endswith("install.sh"):
            return True  # install.sh exists
        return original_exists(self)

    with patch('bugcam.commands.setup.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path), \
         patch('subprocess.run', return_value=mock_result) as mock_run, \
         patch('bugcam.commands.setup.shutil.move') as mock_move, \
         patch('bugcam.commands.setup.shutil.rmtree') as mock_rmtree, \
         patch.object(Path, 'exists', mock_path_exists), \
         patch('bugcam.commands.setup._is_hailo_platform_available', return_value=False), \
         patch.object(Path, 'mkdir'), \
         patch('bugcam.commands.setup.check_import', side_effect=lambda exe, mod: mod != "hailo_platform"):
        result = cli_runner.invoke(app, ["setup"])

        # Should have called git clone to /tmp/hailo-rpi5-examples-setup
        calls = [call[0][0] for call in mock_run.call_args_list]
        git_clone_call = next((c for c in calls if len(c) > 2 and c[0] == "git" and c[1] == "clone"), None)
        assert git_clone_call is not None
        assert "hailo-apps.git" in " ".join(git_clone_call)
        assert "/tmp/hailo-rpi5-examples-setup" in " ".join(git_clone_call)


@pytest.mark.xfail(reason="SG-027: setup tests drifted from setup.py 'already complete' flow", strict=False)
def test_setup_skips_clone_if_exists(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test setup skips setup if hailo venv already exists."""
    # Create fake hailo venv at permanent location
    hailo_venv_dir = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv"
    hailo_venv_dir.mkdir(parents=True)
    hailo_python = hailo_venv_dir / "bin" / "python"
    hailo_python.parent.mkdir(parents=True, exist_ok=True)
    hailo_python.touch()

    with patch('bugcam.commands.setup.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path), \
         patch('subprocess.run') as mock_run, \
         patch('bugcam.commands.setup.check_import', return_value=True):
        result = cli_runner.invoke(app, ["setup"])

        # Should not have called git clone
        calls = [call[0][0] for call in mock_run.call_args_list]
        git_clone_calls = [c for c in calls if isinstance(c, list) and len(c) > 1 and c[0] == "git" and c[1] == "clone"]
        assert len(git_clone_calls) == 0
        assert "Hailo setup already complete" in result.output


@pytest.mark.xfail(reason="SG-027: setup tests drifted from setup.py 'already complete' flow", strict=False)
def test_setup_runs_install_script(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test setup runs install.sh script from temp directory."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    temp_clone_dir = Path("/tmp/hailo-rpi5-examples-setup")
    temp_venv_dir = temp_clone_dir / "venv_hailo_rpi_examples"
    hailo_venv_dir = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv"

    # Mock Path.exists to control which paths exist
    original_exists = Path.exists
    def mock_path_exists(self: Path) -> bool:
        if self == hailo_venv_dir:
            return False  # Venv doesn't exist yet
        if self == temp_clone_dir:
            return False  # Temp dir doesn't exist before clone
        if self == temp_venv_dir:
            return True  # Venv created by install.sh
        if str(self).endswith("install.sh"):
            return True  # install.sh exists
        return original_exists(self)

    with patch('bugcam.commands.setup.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path), \
         patch('subprocess.run', return_value=mock_result) as mock_run, \
         patch('bugcam.commands.setup.shutil.move') as mock_move, \
         patch('bugcam.commands.setup.shutil.rmtree') as mock_rmtree, \
         patch.object(Path, 'exists', mock_path_exists), \
         patch('bugcam.commands.setup._is_hailo_platform_available', return_value=False), \
         patch.object(Path, 'mkdir'), \
         patch('bugcam.commands.setup.check_import', side_effect=lambda exe, mod: mod != "hailo_platform"):
        result = cli_runner.invoke(app, ["setup"])

        # Should have called ./install.sh with cwd set to temp directory
        calls = [call for call in mock_run.call_args_list]
        install_call = next((c for c in calls if c[0][0] == ["sudo", "./install.sh", "--no-tappas-required"]), None)
        assert install_call is not None
        assert install_call[1]['cwd'] == str(temp_clone_dir)


@pytest.mark.xfail(reason="SG-027: setup tests drifted from setup.py 'already complete' flow", strict=False)
def test_setup_install_script_failure(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test setup handles install.sh failure from temp directory."""
    mock_success = MagicMock()
    mock_success.returncode = 0
    mock_failure = MagicMock()
    mock_failure.returncode = 1

    temp_clone_dir = Path("/tmp/hailo-rpi5-examples-setup")
    hailo_venv_dir = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv"

    def run_side_effect(cmd, **kwargs):
        if "install.sh" in cmd:
            return mock_failure
        return mock_success

    # Mock Path.exists to control which paths exist
    original_exists = Path.exists
    def mock_path_exists(self: Path) -> bool:
        if self == hailo_venv_dir:
            return False  # Venv doesn't exist yet
        if self == temp_clone_dir:
            return False  # Temp dir doesn't exist before clone
        if str(self).endswith("install.sh"):
            return True  # install.sh exists
        return original_exists(self)

    with patch('bugcam.commands.setup.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path), \
         patch('subprocess.run', side_effect=run_side_effect), \
         patch('bugcam.commands.setup.shutil.rmtree') as mock_rmtree, \
         patch.object(Path, 'exists', mock_path_exists), \
         patch('bugcam.commands.setup._is_hailo_platform_available', return_value=False), \
         patch('bugcam.commands.setup.check_import', side_effect=lambda exe, mod: mod != "hailo_platform"), \
         patch.object(Path, 'mkdir'):
        result = cli_runner.invoke(app, ["setup"])
        assert result.exit_code == 1
        assert "failed" in result.output.lower()


@pytest.mark.xfail(reason="SG-027: setup tests drifted from setup.py 'already complete' flow", strict=False)
def test_setup_verifies_hailo_apps(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test setup verifies hailo_apps installation after moving venv."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    temp_clone_dir = Path("/tmp/hailo-rpi5-examples-setup")
    temp_venv_dir = temp_clone_dir / "venv_hailo_rpi_examples"
    hailo_venv_dir = tmp_path / ".local" / "share" / "bugcam" / "hailo-venv"

    # Mock Path.exists to control which paths exist
    original_exists = Path.exists
    def mock_path_exists(self: Path) -> bool:
        if self == hailo_venv_dir:
            return False  # Venv doesn't exist yet
        if self == temp_clone_dir:
            return True  # For cleanup check (exists after clone)
        if self == temp_venv_dir:
            return True  # Venv created by install.sh
        if str(self).endswith("install.sh"):
            return True  # install.sh exists
        return original_exists(self)

    existing_config = {"api_key": "test-key", "flick_id": "test-flick", "s3_bucket": "test-bucket", "api_url": "https://api.test.com/v1"}

    with patch('bugcam.commands.setup.platform.system', return_value='Linux'), \
         patch.object(Path, 'home', return_value=tmp_path), \
         patch('subprocess.run', return_value=mock_result), \
         patch('bugcam.commands.setup.shutil.move') as mock_move, \
         patch('bugcam.commands.setup.shutil.rmtree') as mock_rmtree, \
         patch.object(Path, 'exists', mock_path_exists), \
         patch.object(Path, 'mkdir'), \
         patch('bugcam.commands.setup._is_hailo_platform_available', return_value=False), \
         patch('bugcam.commands.setup.check_import', side_effect=lambda exe, mod: mod != "hailo_platform") as mock_check, \
         patch('bugcam.commands.setup.load_config', return_value=existing_config), \
         patch('bugcam.commands.setup.save_config'), \
         patch('bugcam.commands.setup._install_sen55_binary'):
        cli_runner.invoke(app, ["setup"], input="\n\n\nn\n")
        # The install path verifies hailo_apps importability after moving the venv.
        assert any(c.args[1] == "hailo_apps" for c in mock_check.call_args_list)
