"""Tests for the local-time recording window and UTC video-stem helpers."""
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bugcam.record_window import (
    DEFAULT_RECORD_WINDOW,
    RecordingWindow,
    local_video_date,
    video_stem_utc_iso,
)


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


# --- from_config ---

def test_from_config_defaults_to_daytime_window_with_timezone() -> None:
    window = RecordingWindow.from_config(None, "Europe/London")
    assert not window.always_open
    assert window.describe().startswith(DEFAULT_RECORD_WINDOW)


def test_from_config_always_open_without_timezone() -> None:
    window = RecordingWindow.from_config(None, None)
    assert window.always_open
    assert window.is_open(_utc("2026-07-14T03:00:00"))


def test_from_config_explicit_always_with_timezone() -> None:
    window = RecordingWindow.from_config("always", "Europe/London")
    assert window.always_open


def test_from_config_rejects_bad_format() -> None:
    with pytest.raises(ValueError):
        RecordingWindow.from_config("5am-10pm", "Europe/London")


def test_from_config_rejects_empty_window() -> None:
    with pytest.raises(ValueError):
        RecordingWindow.from_config("08:00-08:00", "Europe/London")


def test_from_config_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError):
        RecordingWindow.from_config("05:00-22:00", "Europe/Narnia")


def test_from_config_window_without_timezone_uses_system_local() -> None:
    # Must not raise; evaluated in system-local time.
    window = RecordingWindow.from_config("00:00-23:59", None)
    assert window.is_open(datetime.now(timezone.utc)) in (True, False)


# --- resolve_zone: the unconfigured (system-local) fallback ---

def test_resolve_zone_unconfigured_delegates_to_tzlocal(monkeypatch) -> None:
    """Pin the actual fix: resolve_zone(None) must return tzlocal's
    DST-aware local zone, not a fixed offset snapshotted at call time."""
    from bugcam import record_window

    sentinel = ZoneInfo("Europe/Amsterdam")
    monkeypatch.setattr(record_window, "get_localzone", lambda: sentinel)

    assert record_window.resolve_zone(None) is sentinel


def test_resolve_zone_unconfigured_is_dst_aware(monkeypatch) -> None:
    """Before this fix, resolve_zone(None) snapshotted a fixed UTC offset at
    call time (datetime.now().astimezone().tzinfo) -- correct only until the
    next DST transition, then wrong by an hour for the rest of the daemon's
    (weeks-long) run. The returned zone must instead recompute its offset
    per-instant, like a real IANA zone."""
    from bugcam import record_window

    monkeypatch.setattr(record_window, "get_localzone", lambda: ZoneInfo("Europe/Amsterdam"))

    zone = record_window.resolve_zone(None)
    winter = datetime(2026, 1, 15, tzinfo=timezone.utc)
    summer = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert zone.utcoffset(winter) != zone.utcoffset(summer)


# --- is_open: wall-clock semantics in the configured zone ---

def test_window_boundaries_in_london_summer() -> None:
    # BST is UTC+1: 05:00 local == 04:00 UTC.
    window = RecordingWindow.from_config("05:00-22:00", "Europe/London")
    assert not window.is_open(_utc("2026-07-14T03:59:00"))  # 04:59 local
    assert window.is_open(_utc("2026-07-14T04:00:00"))      # 05:00 local (inclusive)
    assert window.is_open(_utc("2026-07-14T20:59:00"))      # 21:59 local
    assert not window.is_open(_utc("2026-07-14T21:00:00"))  # 22:00 local (exclusive)


def test_window_follows_dst_wall_clock() -> None:
    # GMT in winter: 05:00 local == 05:00 UTC.
    window = RecordingWindow.from_config("05:00-22:00", "Europe/London")
    assert not window.is_open(_utc("2026-01-14T04:30:00"))  # 04:30 local
    assert window.is_open(_utc("2026-01-14T05:30:00"))      # 05:30 local


def test_window_in_toronto() -> None:
    # EDT is UTC-4: 02:00 UTC == 22:00 local the previous evening.
    window = RecordingWindow.from_config("05:00-22:00", "America/Toronto")
    assert not window.is_open(_utc("2026-07-14T02:00:00"))
    assert window.is_open(_utc("2026-07-14T13:00:00"))      # 09:00 local


def test_overnight_window() -> None:
    window = RecordingWindow.from_config("22:00-05:00", "Europe/London")
    assert window.is_open(_utc("2026-07-14T22:00:00"))      # 23:00 local
    assert window.is_open(_utc("2026-07-14T02:00:00"))      # 03:00 local
    assert not window.is_open(_utc("2026-07-14T11:00:00"))  # 12:00 local


# --- video stem helpers (recorder writes UTC stems) ---

def test_video_stem_utc_iso_has_explicit_offset() -> None:
    assert video_stem_utc_iso("20260714_130000_123456") == "2026-07-14T13:00:00+00:00"


def test_video_stem_utc_iso_invalid_stem_returns_none() -> None:
    assert video_stem_utc_iso("notastem") is None


def test_local_video_date_converts_to_configured_zone() -> None:
    # 02:00 UTC on the 14th is 22:00 EDT on the 13th.
    tz = ZoneInfo("America/Toronto")
    assert local_video_date("20260714_020000_123456", tz) == "20260713"


def test_local_video_date_without_zone_uses_stem_date() -> None:
    assert local_video_date("20260714_020000_123456", None) == "20260714"


def test_local_video_date_bad_stem_falls_back_to_prefix() -> None:
    assert local_video_date("20260714_garbage", ZoneInfo("Europe/London")) == "20260714"


# --- recorder integration ---

def test_chunk_filenames_are_utc(tmp_path: Path) -> None:
    from bugcam.edge26.recorder import VideoRecorder

    recorder = VideoRecorder(
        output_dir=str(tmp_path),
        fps=30,
        chunk_duration=60,
        resolution=(1080, 1080),
        device_id="flick-test",
    )
    stem = recorder._generate_chunk_path().stem
    stamp = datetime.strptime(
        stem[len("flick-test") + 1:], "%Y%m%d_%H%M%S_%f"
    ).replace(tzinfo=timezone.utc)
    skew = abs((stamp - datetime.now(timezone.utc)).total_seconds())
    assert skew < 60, f"filename stamp is {skew:.0f}s from UTC now; expected UTC stamps"


class _StubWindow:
    def __init__(self, open_: bool) -> None:
        self._open = open_

    def is_open(self, now=None) -> bool:
        return self._open

    def describe(self) -> str:
        return "stub"

    @property
    def always_open(self) -> bool:
        return False


def _make_recorder(tmp_path: Path, window):
    from bugcam.edge26.recorder import VideoRecorder

    return VideoRecorder(
        output_dir=str(tmp_path),
        fps=30,
        chunk_duration=60,
        resolution=(1080, 1080),
        device_id="flick-test",
        record_window=window,
    )


def test_wait_for_window_passes_through_when_open(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, _StubWindow(True))
    assert recorder._wait_for_window() is True


def test_wait_for_window_returns_false_on_stop(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, _StubWindow(False))
    # Release the wait shortly after it starts polling.
    threading.Timer(0.2, recorder.stop_event.set).start()
    assert recorder._wait_for_window() is False


def test_wait_for_window_without_window_is_open(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, None)
    assert recorder._wait_for_window() is True


# --- config plumbing ---

def test_edge26_config_carries_timezone_and_window(tmp_path: Path) -> None:
    from bugcam.edge26.config import build_edge26_config

    config = build_edge26_config(
        flick_id="flick-test",
        dot_ids=[],
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        model_path="model.hef",
        labels_path="labels.txt",
        timezone_name="Europe/Amsterdam",
        record_window="06:00-21:00",
    )
    assert config["device"]["timezone"] == "Europe/Amsterdam"
    assert config["pipeline"]["record_window"] == "06:00-21:00"


def test_heartbeat_payload_includes_timezone(tmp_path: Path, monkeypatch) -> None:
    from bugcam.commands import heartbeat

    monkeypatch.setattr(heartbeat, "_read_cpu_temperature_celsius", lambda: 42.0)
    monkeypatch.setattr(heartbeat, "_read_uptime_seconds", lambda: 100.0)
    monkeypatch.setattr(heartbeat, "_read_network_usage", lambda: [])
    payload = heartbeat.build_heartbeat_payload(
        "flick-test", tmp_path, [], timezone_name="America/Toronto"
    )
    assert payload["timezone"] == "America/Toronto"


def test_resolve_recording_window_prefers_cli_over_config(monkeypatch) -> None:
    from bugcam.commands import run as run_module

    monkeypatch.setattr(
        run_module,
        "load_config",
        lambda: {"timezone": "Europe/London", "record_window": "06:00-20:00"},
    )
    tz_name, window = run_module._resolve_recording_window("America/Toronto", None)
    assert tz_name == "America/Toronto"
    assert window == "06:00-20:00"

    tz_name, window = run_module._resolve_recording_window(None, "always")
    assert tz_name == "Europe/London"
    assert window == "always"


def test_setup_saves_timezone() -> None:
    from bugcam.commands.setup import _build_saved_config

    saved = _build_saved_config(
        existing_config={"timezone": "Europe/London"},
        api_url="https://api.example.com",
        api_key="key",
        flick_id="flick-test",
        dot_ids=[],
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        state_dir="/tmp/state",
        pending_dir="/tmp/pending",
        timezone_name="America/Toronto",
    )
    assert saved["timezone"] == "America/Toronto"


def test_setup_blank_timezone_clears_key() -> None:
    from bugcam.commands.setup import _build_saved_config

    saved = _build_saved_config(
        existing_config={"timezone": "Europe/London"},
        api_url="https://api.example.com",
        api_key="key",
        flick_id="flick-test",
        dot_ids=[],
        input_dir="/tmp/in",
        output_dir="/tmp/out",
        state_dir="/tmp/state",
        pending_dir="/tmp/pending",
        timezone_name="",
    )
    assert "timezone" not in saved
