"""Intent handlers -- the only code that performs actions.

Accepts ONLY a validated IntentCall (schema/intents.py). Destructive
capabilities are ABSENT from this package by design: there is no delete, no
send, no spend, no arbitrary shell. A hallucinated or prompt-injected intent has
nothing to call, which is a stronger guarantee than a policy check.

System plumbing lives in macos.py and note storage in notes.py, so the handlers
below read as intent logic.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..schema.intents import MAX_LISTED_ITEMS, Intent, IntentCall
from . import macos, reminders
from .notes import NoteStore

DEFAULT_VOLUME_STEP = 20
FILE_RESULT_LIMIT = 5

# `open` RUNS some file types instead of displaying them (.command and .app
# outright, .sh and .scpt if the user has pointed them at a shell, .pkg/.dmg at
# an installer). find_file is the one intent where the model chooses the target,
# so anything outside this set is revealed in Finder rather than opened.
OPENABLE_SUFFIXES = frozenset(
    {
        ".pdf", ".txt", ".md", ".rtf", ".csv", ".json", ".xml", ".log",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pages", ".numbers", ".key",
        ".png", ".jpg", ".jpeg", ".gif", ".heic", ".tiff", ".bmp", ".webp", ".svg",
        ".mp3", ".m4a", ".wav", ".aiff", ".flac", ".mp4", ".mov", ".m4v",
    }
)
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class ActionResult:
    """Outcome of one action. `speech` is what the confirmation should say --
    empty means stay silent (the correct response to `unknown`)."""

    ok: bool
    speech: str
    detail: str = ""


def describe_duration(seconds: int) -> str:
    """Render a duration the way a person would say it, for the confirmation."""
    if seconds >= SECONDS_PER_HOUR and seconds % SECONDS_PER_HOUR == 0:
        count, unit = seconds // SECONDS_PER_HOUR, "hour"
    elif seconds >= SECONDS_PER_MINUTE and seconds % SECONDS_PER_MINUTE == 0:
        count, unit = seconds // SECONDS_PER_MINUTE, "minute"
    else:
        count, unit = seconds, "second"
    return f"{count} {unit}" + ("s" if count != 1 else "")


class Executor:
    """Dispatches validated intents to handlers.

    `notify` is called when a timer fires, from the timer's own thread.
    """

    def __init__(
        self,
        notify: Callable[[str], None] | None = None,
        notes: NoteStore | None = None,
    ):
        self.notify = notify or (lambda message: print(f"  [timer] {message}"))
        self.notes = notes or NoteStore()
        self._timers: list[threading.Timer] = []

    # -- handlers ---------------------------------------------------------

    def _set_timer(self, args: dict) -> ActionResult:
        seconds = args["duration_seconds"]
        spoken = describe_duration(seconds)

        timer = threading.Timer(seconds, lambda: self.notify(f"Your {spoken} timer is up."))
        timer.daemon = True  # session-scoped: dies with the process, by design
        timer.start()
        # Fired timers are never removed otherwise, so the list grows for the
        # life of the session and `cancel` walks dead threads.
        self._timers = [t for t in self._timers if t.is_alive()]
        self._timers.append(timer)
        return ActionResult(True, f"Timer set for {spoken}.", f"{seconds}s")

    def _open_app(self, args: dict) -> ActionResult:
        app_name = args["app_name"]  # already resolved against the allowlist
        macos.open_application(app_name)
        return ActionResult(True, f"Opening {app_name}.")

    def _close_app(self, args: dict) -> ActionResult:
        """Quit an app. Only ever apps we would also open (same allowlist)."""
        app_name = args["app_name"]
        if not macos.is_running(app_name):
            return ActionResult(True, f"{app_name} isn't running.")
        macos.quit_application(app_name)
        return ActionResult(True, f"Closing {app_name}.")

    def _find_file(self, args: dict) -> ActionResult:
        """Spotlight search, scoped to the user's home directory.

        Reports what it found and opens the top match. Opening only the FIRST
        result is deliberate -- "open my resume" with six matches should not
        launch six windows.

        This is the only intent whose target the MODEL picks rather than an
        allowlist, so file types that `open` would execute are revealed in
        Finder instead. "find my install script" must not be able to run one.
        """
        query = args["query"]
        matches = macos.spotlight_search(
            query, limit=FILE_RESULT_LIMIT, search_root=str(Path.home())
        )
        if not matches:
            return ActionResult(True, f"I couldn't find anything matching {query}.")

        best = Path(matches[0])
        found = "" if len(matches) == 1 else f"I found {len(matches)} matches. "
        if best.suffix.lower() in OPENABLE_SUFFIXES:
            macos.open_file_path(str(best))
            return ActionResult(True, f"{found}Opening {best.name}.", str(best))
        macos.reveal_file_path(str(best))
        return ActionResult(
            True, f"{found}I found {best.name}. Showing it in Finder.", str(best)
        )

    def _open_path(self, args: dict) -> ActionResult:
        path = Path(args["path"])  # already resolved against the allowlist
        if not path.exists():
            return ActionResult(False, "That folder doesn't exist.", str(path))
        macos.open_file_path(str(path))
        return ActionResult(True, f"Opening {path.name}.")

    def _set_volume(self, args: dict) -> ActionResult:
        direction = args["direction"]

        if direction in ("mute", "unmute"):
            muted = "true" if direction == "mute" else "false"
            macos.dispatch(["osascript", "-e", f"set volume output muted {muted}"])
            return ActionResult(True, "Muted." if direction == "mute" else "Unmuted.")

        if direction == "set":
            level = args["level"]
            script = f"set volume output volume {level}\nset volume output muted false"
            speech = f"Volume set to {level}."
        else:
            # Read, clamp and set in ONE script: each osascript round trip is
            # ~200 ms, so doing this as two calls would double the cost.
            step = args.get("level", DEFAULT_VOLUME_STEP)
            operator = "+" if direction == "up" else "-"
            unmute = "\nset volume output muted false" if direction == "up" else ""
            script = (
                "set currentVolume to output volume of (get volume settings)\n"
                f"set targetVolume to currentVolume {operator} {step}\n"
                "if targetVolume > 100 then set targetVolume to 100\n"
                "if targetVolume < 0 then set targetVolume to 0\n"
                "set volume output volume targetVolume" + unmute
            )
            speech = f"Volume {direction}."

        macos.dispatch(["osascript", "-e", script])
        return ActionResult(True, speech)

    def _media_control(self, args: dict) -> ActionResult:
        action = args["action"]
        applescript_verb = {
            "play": "play",
            "pause": "pause",
            "playpause": "playpause",
            "next": "next track",
            "previous": "previous track",
        }[action]

        # Control whichever player is already running; never launch one.
        for app_name in ("Spotify", "Music"):
            if macos.is_running(app_name):
                macos.dispatch(
                    ["osascript", "-e", f'tell application "{app_name}" to {applescript_verb}']
                )
                speech = {"play": "Played.", "pause": "Paused."}.get(action, "Done.")
                return ActionResult(True, speech, app_name)
        return ActionResult(False, "Nothing is playing.")

    def _capture_note(self, args: dict) -> ActionResult:
        note = self.notes.append(args["text"])
        return ActionResult(True, "Noted.", note.text[:60])

    def _search_notes(self, args: dict) -> ActionResult:
        matches = self.notes.search(args["query"])
        if not matches:
            return ActionResult(True, "I couldn't find anything about that.")

        plural = "s" if len(matches) != 1 else ""
        speech = f"I found {len(matches)} note{plural}. {matches[0].text}"
        return ActionResult(True, speech, f"{len(matches)} hits")

    def _get_time(self, args: dict) -> ActionResult:
        # Local wall-clock time is exactly what "what time is it" means.
        now = datetime.now().astimezone()
        # Spell out AM/PM so the TTS says "A M" rather than a mangled acronym.
        spoken = now.strftime("%-I:%M %p").replace("AM", "A M").replace("PM", "P M")
        return ActionResult(True, f"It's {spoken}.")

    def _cancel(self, args: dict) -> ActionResult:
        cancelled = 0
        for timer in self._timers:
            if timer.is_alive():
                timer.cancel()
                cancelled += 1
        self._timers.clear()
        return ActionResult(
            True, "Cancelled." if cancelled else "Nothing to cancel.", f"{cancelled} timers"
        )

    # -- capabilities added after live testing ----------------------------

    def _get_status(self, args: dict) -> ActionResult:
        """One handler, one reading. Every branch is a cheap local read."""
        item = args["item"]
        if item in ("date", "day"):
            now = datetime.now().astimezone()
            spoken = now.strftime("%A, %B %-d") if item == "day" else now.strftime("%B %-d, %Y")
            return ActionResult(True, f"It's {spoken}.")

        if item == "battery":
            percent, on_ac = macos.battery_percent()
            if percent is None:
                return ActionResult(False, "I couldn't read the battery.")
            charging = " and charging" if on_ac else ""
            return ActionResult(True, f"Battery is at {percent} percent{charging}.")

        if item in ("wifi", "bluetooth"):
            reader = macos.wifi_is_on if item == "wifi" else macos.bluetooth_is_on
            state = reader()
            if state is None:
                return ActionResult(False, f"I couldn't check {item}.")
            name = "Wi-Fi" if item == "wifi" else "Bluetooth"
            return ActionResult(True, f"{name} is {'on' if state else 'off'}.")

        if item == "storage":
            free_gb = shutil.disk_usage(Path.home()).free / 1e9
            return ActionResult(True, f"You have {free_gb:.0f} gigabytes free.")

        chip, device = macos.chip_name(), macos.device_name()
        if item == "chip":
            return ActionResult(bool(chip), f"This is an {chip}." if chip else "I couldn't tell.")
        return ActionResult(bool(device), f"This is a {device}." if device else "I couldn't tell.")

    def _list_items(self, args: dict) -> ActionResult:
        """Enumerate, and say how many there are before naming a few.

        The count is the useful part when spoken aloud -- reading forty filenames
        is worse than useless, so only MAX_LISTED_ITEMS are named.
        """
        if args["what"] == "notes":
            names = [note.text for note in self.notes.all()]
            noun = "note"
        else:
            folder = Path(args["where"])
            if not folder.exists():
                return ActionResult(False, "That folder doesn't exist.", str(folder))
            names = sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))
            noun = "item"

        if not names:
            return ActionResult(True, f"There are no {noun}s there.")
        plural = "" if len(names) == 1 else "s"
        head = ", ".join(names[:MAX_LISTED_ITEMS])
        more = "" if len(names) <= MAX_LISTED_ITEMS else ", and more"
        return ActionResult(True, f"You have {len(names)} {noun}{plural}. {head}{more}.", head[:60])

    def _open_folder(self, args: dict) -> ActionResult:
        """Spotlight, folders only, scoped to home. Mirrors _find_file."""
        name = args["name"]
        matches = macos.spotlight_search_folders(
            name, limit=FILE_RESULT_LIMIT, search_root=str(Path.home())
        )
        if not matches:
            return ActionResult(True, f"I couldn't find a folder called {name}.")
        best = Path(matches[0])
        macos.open_file_path(str(best))
        found = "" if len(matches) == 1 else f"I found {len(matches)} folders. "
        return ActionResult(True, f"{found}Opening {best.name}.", str(best))

    def _create_reminder(self, args: dict) -> ActionResult:
        """Alarms and reminders both land here -- see executor/reminders.py."""
        when = reminders.resolve_datetime(
            args.get("hour"), args.get("minute", 0), args.get("day")
        )
        try:
            spoken_time = reminders.create(args["text"], when)
        except reminders.RemindersUnavailableError as exc:
            return ActionResult(
                False,
                "I couldn't reach Reminders. Check Automation permission in System Settings.",
                str(exc)[:80],
            )
        if not spoken_time:
            return ActionResult(True, "Reminder added.")
        day_word = "tomorrow" if when.date() > datetime.now().astimezone().date() else "today"
        return ActionResult(True, f"Reminder set for {spoken_time} {day_word}.", spoken_time)

    def _unknown(self, args: dict) -> ActionResult:
        # Success path, not an error. Declining is the correct behaviour, and
        # staying silent is essential: confirming every stray sentence in the
        # room would make an always-listening assistant unusable.
        return ActionResult(True, "", "declined")

    # -- dispatch ---------------------------------------------------------

    def execute(self, call: IntentCall) -> ActionResult:
        handler = {
            Intent.SET_TIMER: self._set_timer,
            Intent.OPEN_APP: self._open_app,
            Intent.CLOSE_APP: self._close_app,
            Intent.OPEN_PATH: self._open_path,
            Intent.SET_VOLUME: self._set_volume,
            Intent.MEDIA_CONTROL: self._media_control,
            Intent.CAPTURE_NOTE: self._capture_note,
            Intent.SEARCH_NOTES: self._search_notes,
            Intent.FIND_FILE: self._find_file,
            Intent.GET_TIME: self._get_time,
            Intent.GET_STATUS: self._get_status,
            Intent.LIST_ITEMS: self._list_items,
            Intent.OPEN_FOLDER: self._open_folder,
            Intent.CREATE_REMINDER: self._create_reminder,
            Intent.CANCEL: self._cancel,
            Intent.UNKNOWN: self._unknown,
        }[call.intent]
        try:
            return handler(call.args)
        except Exception as exc:
            # Report honestly; never swallow. A failed action must not look like
            # a successful one.
            return ActionResult(False, "Something went wrong.", f"{type(exc).__name__}: {exc}")

    def shutdown(self) -> None:
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
