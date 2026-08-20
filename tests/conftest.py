"""Shared fixtures.

The executor's whole job is to cause real effects on this machine, so tests that
touch it must replace the system boundary. Without this, `pytest tests/` opens
Terminal, opens a Finder window, lowers the system volume and writes a note into
Apple Notes -- verified before this fixture existed.
"""

from __future__ import annotations

import pytest

from mini_siri_local.executor import macos
from mini_siri_local.executor.handlers import Executor
from mini_siri_local.executor.notes import LocalNoteStore, NoteStore


@pytest.fixture
def system_calls(monkeypatch) -> list[list[str]]:
    """Record what the executor would run instead of running it."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(macos, "dispatch", recorded.append)
    monkeypatch.setattr(
        macos, "open_application", lambda name: recorded.append(["open", "-a", name])
    )
    monkeypatch.setattr(macos, "open_file_path", lambda path: recorded.append(["open", path]))
    monkeypatch.setattr(macos, "quit_application", lambda name: recorded.append(["quit", name]))
    monkeypatch.setattr(macos, "run_osascript", lambda script, timeout_s=3.0: (False, "stubbed"))
    monkeypatch.setattr(macos, "is_running", lambda name: False)
    monkeypatch.setattr(macos, "spotlight_search", lambda query, limit, search_root: [])
    monkeypatch.setattr(macos, "spotlight_search_folders", lambda query, limit, search_root: [])
    monkeypatch.setattr(
        macos, "reveal_file_path", lambda path: recorded.append(["open", "-R", path])
    )

    # Status readings shell out too. They change nothing, but they are slow
    # (system_profiler is ~230 ms) and machine-dependent, so a test asserting on
    # this laptop's battery level would fail on any other.
    monkeypatch.setattr(macos, "battery_percent", lambda: (77, False))
    monkeypatch.setattr(macos, "wifi_is_on", lambda: True)
    monkeypatch.setattr(macos, "bluetooth_is_on", lambda: False)
    monkeypatch.setattr(macos, "chip_name", lambda: "Apple M-test")
    monkeypatch.setattr(macos, "device_name", lambda: "Test Mac")
    return recorded


@pytest.fixture
def executor(system_calls, tmp_path) -> Executor:
    notes = NoteStore(prefer_apple_notes=False)
    notes.local = LocalNoteStore(tmp_path / "notes.jsonl")
    ex = Executor(notify=lambda message: None, notes=notes)
    yield ex
    ex.shutdown()
