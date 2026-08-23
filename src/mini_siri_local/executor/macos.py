"""Thin wrappers over the macOS commands the executor needs.

Isolated here so the intent handlers read as intent logic rather than
subprocess plumbing, and so the cost of each system call is documented in one
place.

Measured on an M1 Pro:
    osascript round trip   ~200 ms
    pgrep -x               ~20 ms
    open -a                ~500 ms (hence fire-and-forget)

Those numbers are why `media_control` uses pgrep instead of an osascript process
query, and why launches are dispatched without waiting.
"""

from __future__ import annotations

import re
import subprocess

OSASCRIPT_TIMEOUT_S = 3.0
PGREP_TIMEOUT_S = 1.0
# Spotlight is usually instant off its index, but can stall on a cold or
# rebuilding index. Bound it so a search never hangs the turn.
SPOTLIGHT_TIMEOUT_S = 4.0


def is_running(app_name: str) -> bool:
    """True if a process with this exact name exists.

    pgrep is ~10x faster than `tell application "System Events" to (name of
    processes) contains ...`, which matters because media_control would
    otherwise make two such queries before doing anything.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-x", app_name], capture_output=True, timeout=PGREP_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def run_osascript(script: str, timeout_s: float = OSASCRIPT_TIMEOUT_S) -> tuple[bool, str]:
    """Run AppleScript and wait. Returns (succeeded, stdout-or-stderr).

    Use only when the RESULT is needed. To merely cause an effect, prefer
    dispatch() -- waiting ~200 ms for a volume change the user already heard is
    pure added latency.
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return False, "osascript timed out"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def dispatch(argv: list[str]) -> None:
    """Launch a command without waiting for it to finish.

    The spoken confirmation does not depend on completion: saying "Opening
    Terminal" while it launches is both faster and more natural than waiting
    ~500 ms for `open -a` to return. The trade is that a failure surfaces in the
    log rather than in the reply -- acceptable for these actions, and there are
    no destructive ones for which it would not be.
    """
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  [executor] dispatch failed: {argv[0]}: {exc}")


def escape_applescript(text: str) -> str:
    r"""Escape a Python string for an AppleScript literal.

    Backslashes first, then quotes -- reversing the order double-escapes the backslashes
    the quote pass introduces.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, text: str) -> None:
    """Post a macOS notification banner. Fire-and-forget, like every other dispatch.

    Used only by the menu-bar mode: in terminal mode the turn is already printed, and an
    always-listening assistant that banner-ed every overheard sentence would be unusable.
    """
    dispatch(
        [
            "osascript",
            "-e",
            f'display notification "{escape_applescript(text)}" '
            f'with title "{escape_applescript(title)}"',
        ]
    )


def open_application(app_name: str) -> None:
    """`open -a`. Caller must have resolved app_name through the allowlist."""
    dispatch(["open", "-a", app_name])


def open_file_path(path: str) -> None:
    """`open <path>`. Caller must have resolved path through the allowlist."""
    dispatch(["open", path])


def reveal_file_path(path: str) -> None:
    """Select the file in Finder without opening it.

    Used for anything `open` would EXECUTE rather than display -- see
    OPENABLE_SUFFIXES in handlers.py.
    """
    dispatch(["open", "-R", path])


def quit_application(app_name: str) -> None:
    """Ask an app to quit. Caller must have resolved app_name through the allowlist.

    `quit` is a polite request, not a kill: the app runs its normal shutdown and
    can prompt about unsaved work. That is deliberate -- a voice command should
    never be able to discard someone's unsaved document.
    """
    dispatch(["osascript", "-e", f'tell application "{app_name}" to quit'])


def spotlight_search_folders(query: str, limit: int, search_root: str) -> list[str]:
    """Find DIRECTORIES by name, scoped to one root.

    The counterpart to spotlight_search for `open_folder`: same safety model
    (`-onlyin` confines the search, the model supplies terms not paths), but
    constrained to folders so "open the audio_lab folder" cannot surface a file.
    """
    expression = f'kMDItemContentType == "public.folder" && kMDItemFSName == "*{query}*"cd'
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", search_root, expression],
            capture_output=True,
            text=True,
            timeout=SPOTLIGHT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()][:limit]


def spotlight_search(query: str, limit: int, search_root: str) -> list[str]:
    """Find files by name using Spotlight, scoped to one directory.

    `-onlyin` confines results to the user's home directory, so a query can
    never surface system files. `-name` matches the FILENAME rather than file
    contents, which is what "find my resume" means to a person.

    Returns absolute paths in whatever order Spotlight gives -- mdfind does not
    document an ordering, so "the best match" is really "the first hit".
    """
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", search_root, "-name", query],
            capture_output=True,
            text=True,
            timeout=SPOTLIGHT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    paths = [line for line in result.stdout.splitlines() if line.strip()]
    return paths[:limit]


# --- system status: cheap local reads, nothing here changes anything ---------

STATUS_TIMEOUT_S = 3.0


def _read(argv: list[str], timeout_s: float = STATUS_TIMEOUT_S) -> str:
    """Run a read-only command and return stdout, or "" if anything goes wrong.

    Status readings are conveniences: a missing binary or a slow response should
    produce "I couldn't check that", never kill the turn.
    """
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def battery_percent() -> tuple[int | None, bool]:
    """(percent, on_ac_power). `pmset -g batt` is the fastest source at ~17 ms."""
    output = _read(["pmset", "-g", "batt"])
    if not output:
        return None, False
    on_ac = "AC Power" in output
    match = re.search(r"(\d+)%", output)
    return (int(match.group(1)) if match else None), on_ac


def wifi_is_on() -> bool | None:
    """None means the reading failed -- distinct from a confident False."""
    output = _read(["networksetup", "-getairportpower", "en0"])
    if not output:
        return None
    return output.strip().endswith("On")


def bluetooth_is_on() -> bool | None:
    """`system_profiler` costs ~230 ms and is the only reliable source: the
    com.apple.Bluetooth preference key it used to read no longer exists."""
    output = _read(["system_profiler", "SPBluetoothDataType"], timeout_s=5.0)
    if not output:
        return None
    match = re.search(r"State:\s*(\w+)", output)
    return match.group(1).lower() == "on" if match else None


def chip_name() -> str:
    return _read(["sysctl", "-n", "machdep.cpu.brand_string"])


def device_name() -> str:
    return _read(["scutil", "--get", "ComputerName"])
