"""Intent schema and the validation boundary.

THE TRUST BOUNDARY OF THE SYSTEM. An IntentCall is constructible only by validate().
There is no code path from raw model text to an action -- not by discipline, by types.

Fail closed: anything unparseable, out-of-enum, out-of-range, or out-of-allowlist
produces a Rejection and NO action. Never guess, never fall back to a near match.

App names are the one field validated against the FILESYSTEM rather than a fixed
list: _installed_apps() scans the standard app directories once per process, so
the model can name any app actually on the machine. This does not weaken the
guarantee -- the set is still closed and still checked, it is just computed from
disk instead of hardcoded, so it stops going stale as apps are installed.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Intent(StrEnum):
    """Closed set. Adding a member is a scope decision, recorded in ENGINEERING.md.

    The four added after live testing are deliberately GROUPED rather than one
    intent per capability: `get_status{item}` covers battery/wifi/bluetooth/
    date/day/chip/storage in one class instead of seven. Fewer classes for a
    1.7B model to separate, an order of magnitude less seed data to author, and
    the same output-token count regardless of how many items exist.
    """

    SET_TIMER = "set_timer"
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    SET_VOLUME = "set_volume"
    CAPTURE_NOTE = "capture_note"
    SEARCH_NOTES = "search_notes"
    FIND_FILE = "find_file"
    OPEN_PATH = "open_path"
    MEDIA_CONTROL = "media_control"
    GET_TIME = "get_time"
    GET_STATUS = "get_status"
    LIST_ITEMS = "list_items"
    OPEN_FOLDER = "open_folder"
    CREATE_REMINDER = "create_reminder"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


# --- allowlists: model output is never passed through as a free string -------

# Directories macOS installs applications into. Scanned once per process by
# _installed_apps() -- the only place validate() touches the filesystem.
APP_SEARCH_DIRS: tuple[Path, ...] = (
    Path("/Applications"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)

ALLOWED_APPS: dict[str, str] = {
    # Spoken ALIASES that don't match their bundle name -- the installed-app
    # scan below cannot know "chrome" means "Google Chrome.app". Checked first;
    # anything not listed here falls through to the scan.
    "terminal": "Terminal",
    "iterm": "iTerm",
    "safari": "Safari",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "finder": "Finder",
    "notes": "Notes",
    "calendar": "Calendar",
    "mail": "Mail",
    "music": "Music",
    "apple music": "Music",
    "spotify": "Spotify",
    "slack": "Slack",
    "whatsapp": "WhatsApp",
    "messages": "Messages",
    "preview": "Preview",
    "system settings": "System Settings",
    "activity monitor": "Activity Monitor",
    "visual studio code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
}


@functools.lru_cache(maxsize=1)
def _installed_apps() -> dict[str, str]:
    """Every .app bundle on this machine: lowercased name -> real name.

    Computed once per process, not per turn -- newly installed apps need a
    restart to be found, an acceptable trade for not touching disk every turn.
    A missing or unreadable directory is skipped rather than raised: this must
    never be the reason a turn fails.
    """
    apps: dict[str, str] = {}
    for directory in APP_SEARCH_DIRS:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.suffix == ".app":
                apps[entry.stem.lower()] = entry.stem
    return apps


def resolve_app_name(name: str) -> str | None:
    """Spoken name -> real macOS app name, or None if nothing matches.

    ALLOWED_APPS is checked first for the aliases the filesystem scan cannot
    know about; anything else must be the literal name of an installed app.
    """
    key = name.strip().lower()
    return ALLOWED_APPS.get(key) or _installed_apps().get(key)


ALLOWED_PATHS: dict[str, Path] = {
    "downloads": Path.home() / "Downloads",
    "documents": Path.home() / "Documents",
    "desktop": Path.home() / "Desktop",
    "home": Path.home(),
    "applications": Path("/Applications"),
    "pictures": Path.home() / "Pictures",
    # "photos" is what people actually say for the Pictures folder; found when
    # "go to my photos folder" was rejected on a real held-out row.
    "photos": Path.home() / "Pictures",
    "music": Path.home() / "Music",
    "movies": Path.home() / "Movies",
}

VOLUME_DIRECTIONS = {"up", "down", "mute", "unmute", "set"}
MEDIA_ACTIONS = {"play", "pause", "playpause", "next", "previous"}

MAX_TIMER_SECONDS = 86_400  # 24 h

# The model names a unit; Python does the multiplication. See _check_timer.
TIMER_UNIT_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600}
MAX_NOTE_CHARS = 500
MAX_QUERY_CHARS = 200
MAX_FILE_QUERY_CHARS = 120

# get_status: one intent, many readings. Every one is a cheap read of local
# system state -- nothing here changes anything, which is why it needs no
# allowlist beyond this set of names.
STATUS_ITEMS = {"battery", "wifi", "bluetooth", "storage", "chip", "device", "date", "day"}

# list_items: what can be enumerated. `where` only applies to files and must be
# an ALLOWED_PATHS key, so listing can never walk outside the standard folders.
LIST_TARGETS = {"notes", "files"}
MAX_LISTED_ITEMS = 10  # spoken aloud, so a long list is useless anyway

MAX_REMINDER_CHARS = 200
# create_reminder `day`. Weekday names are here because declining them created a
# worse problem than supporting them: "remind me ... next Tuesday" had to be a
# decline, and that decline shares its surface form with thirty positive reminder
# examples. Python resolves a weekday to its NEXT occurrence.
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
REMINDER_DAYS = ("today", "tomorrow", *WEEKDAYS)
MAX_FOLDER_QUERY_CHARS = 80


@dataclass(frozen=True)
class IntentCall:
    """A validated, executable intent.

    Only validate() constructs this. The executor accepts nothing else, so an
    unvalidated string cannot reach an action even by mistake.
    """

    intent: Intent
    args: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


DEFAULT_DECLINE_SPEECH = "Sorry, I didn't catch that."


@dataclass(frozen=True)
class Rejection:
    """`reason` is for logs and is never spoken -- it can name intents, allowlist
    keys, and other things a user did not say. `speech` is what the pipeline
    actually says out loud, and defaults to a generic decline so a checker only
    needs to set it when it has something more specific and still SAFE to say
    (see _check_open_app for why "not found" is fine to speak but "not in
    allowlist" would be confusing)."""

    reason: str
    raw: str = ""
    speech: str = DEFAULT_DECLINE_SPEECH


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output.

    Models wrap JSON in prose or fences even when told not to. This is lenient
    about the wrapper and strict about the contents.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
        return int(v)
    return None


def validate(raw: str) -> IntentCall | Rejection:
    """Raw model output -> IntentCall or Rejection. The only constructor of IntentCall."""
    obj = _extract_json(raw)
    if obj is None:
        return Rejection("output is not parseable JSON", raw)

    name = obj.get("intent")
    if not isinstance(name, str):
        return Rejection("missing 'intent' field", raw)
    try:
        intent = Intent(name.strip().lower())
    except ValueError:
        # Never map an unknown name onto a near match -- that is guessing at an action.
        return Rejection(f"unknown intent {name!r}", raw)

    args = obj.get("args") or {}
    if not isinstance(args, dict):
        return Rejection("'args' is not an object", raw)

    checker = _CHECKERS.get(intent)
    if checker is None:
        return IntentCall(intent, {}, raw)
    return checker(args, raw)


# --- per-intent argument validation -----------------------------------------


def _check_timer(args: dict, raw: str) -> IntentCall | Rejection:
    """Accepts {"value": n, "unit": "minutes"} -- and legacy {"duration_seconds": n}.

    The value+unit form exists because ASKING THE MODEL TO MULTIPLY IS A BUG
    SOURCE. Measured on the fine-tuned checkpoint emitting duration_seconds
    directly:

        "three hours"  -> 1080   (want 10800)   digit dropped
        "45 minutes"   -> 270    (want 2700)    digit dropped
        "1 hour"       -> 600    (want 3600)    wrong multiplier

    A 1.7B model generating a large integer digit-by-digit is unreliable, and a
    10x error in a duration is a silently wrong action. Having it copy the
    number it heard ("3") and name the unit ("hours") moves the arithmetic to
    Python, where it cannot be wrong.

    duration_seconds is still accepted so older prompts/checkpoints keep working.
    """
    value = _as_int(args.get("value"))
    unit = args.get("unit")

    if value is not None and isinstance(unit, str):
        multiplier = TIMER_UNIT_SECONDS.get(unit.strip().lower().rstrip("s") + "s")
        if multiplier is None:
            return Rejection(
                f"set_timer unit must be one of {sorted(TIMER_UNIT_SECONDS)}, got {unit!r}", raw
            )
        seconds = value * multiplier
    else:
        seconds = _as_int(args.get("duration_seconds"))
        if seconds is None:
            return Rejection("set_timer needs {value, unit} or duration_seconds", raw)

    if not 1 <= seconds <= MAX_TIMER_SECONDS:
        return Rejection(f"timer duration {seconds}s out of range 1..{MAX_TIMER_SECONDS}", raw)
    return IntentCall(Intent.SET_TIMER, {"duration_seconds": seconds}, raw)


def _check_open_app(args: dict, raw: str) -> IntentCall | Rejection:
    name = args.get("app_name")
    if not isinstance(name, str) or not name.strip():
        return Rejection("open_app needs app_name", raw)
    resolved = resolve_app_name(name)
    if resolved is None:
        # Safe to speak: it names only what the USER just said, nothing about
        # the resolution mechanism -- unlike "not in allowlist", which would be.
        speech = f"I don't have an app called {name.strip()}."
        return Rejection(f"app {name!r} not found on this machine", raw, speech)
    return IntentCall(Intent.OPEN_APP, {"app_name": resolved}, raw)


def _check_close_app(args: dict, raw: str) -> IntentCall | Rejection:
    """Same resolution as open_app -- we only ever quit apps we would open."""
    name = args.get("app_name")
    if not isinstance(name, str) or not name.strip():
        return Rejection("close_app needs app_name", raw)
    resolved = resolve_app_name(name)
    if resolved is None:
        speech = f"I don't have an app called {name.strip()}."
        return Rejection(f"app {name!r} not found on this machine", raw, speech)
    return IntentCall(Intent.CLOSE_APP, {"app_name": resolved}, raw)


def _check_find_file(args: dict, raw: str) -> IntentCall | Rejection:
    """Spotlight search terms.

    The query is a SEARCH STRING, never a path -- the executor runs it through
    Spotlight and only ever surfaces results under the user's home directory.
    Rejecting path-like input here keeps the model from smuggling one through.
    """
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return Rejection("find_file needs a non-empty query", raw)
    cleaned = query.strip()
    if cleaned.startswith(("/", "~")) or ".." in cleaned:
        return Rejection(f"find_file query {cleaned!r} looks like a path, not search terms", raw)
    return IntentCall(Intent.FIND_FILE, {"query": cleaned[:MAX_FILE_QUERY_CHARS]}, raw)


def _check_volume(args: dict, raw: str) -> IntentCall | Rejection:
    direction = args.get("direction")
    if not isinstance(direction, str) or direction.strip().lower() not in VOLUME_DIRECTIONS:
        return Rejection(f"set_volume direction must be one of {sorted(VOLUME_DIRECTIONS)}", raw)
    direction = direction.strip().lower()
    out: dict[str, Any] = {"direction": direction}
    if "level" in args and args["level"] is not None:
        level = _as_int(args["level"])
        if level is None or not 0 <= level <= 100:
            return Rejection("set_volume level must be an integer 0..100", raw)
        out["level"] = level
    elif direction == "set":
        return Rejection("set_volume direction 'set' requires a level", raw)
    return IntentCall(Intent.SET_VOLUME, out, raw)


def _check_note(args: dict, raw: str) -> IntentCall | Rejection:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return Rejection("capture_note needs non-empty text", raw)
    return IntentCall(Intent.CAPTURE_NOTE, {"text": text.strip()[:MAX_NOTE_CHARS]}, raw)


def _check_search(args: dict, raw: str) -> IntentCall | Rejection:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return Rejection("search_notes needs non-empty query", raw)
    return IntentCall(Intent.SEARCH_NOTES, {"query": query.strip()[:MAX_QUERY_CHARS]}, raw)


def _check_open_path(args: dict, raw: str) -> IntentCall | Rejection:
    key = args.get("path")
    if not isinstance(key, str) or not key.strip():
        return Rejection("open_path needs path", raw)
    resolved = ALLOWED_PATHS.get(key.strip().lower())
    if resolved is None:
        # Deliberately no arbitrary filesystem paths: an allowlist key or nothing.
        return Rejection(f"path {key!r} not in allowlist", raw)
    return IntentCall(Intent.OPEN_PATH, {"path": str(resolved)}, raw)


def _check_media(args: dict, raw: str) -> IntentCall | Rejection:
    action = args.get("action")
    if not isinstance(action, str) or action.strip().lower() not in MEDIA_ACTIONS:
        return Rejection(f"media_control action must be one of {sorted(MEDIA_ACTIONS)}", raw)
    return IntentCall(Intent.MEDIA_CONTROL, {"action": action.strip().lower()}, raw)


def _check_status(args: dict, raw: str) -> IntentCall | Rejection:
    """One intent, one enum-checked reading. See STATUS_ITEMS."""
    item = args.get("item")
    if not isinstance(item, str) or item.strip().lower() not in STATUS_ITEMS:
        return Rejection(f"get_status item must be one of {sorted(STATUS_ITEMS)}", raw)
    return IntentCall(Intent.GET_STATUS, {"item": item.strip().lower()}, raw)


def _check_list_items(args: dict, raw: str) -> IntentCall | Rejection:
    """`where` is an ALLOWED_PATHS key, never a path.

    Listing is a read, but it still reveals filenames, so it is confined to the
    same eight standard folders `open_path` allows rather than anywhere on disk.
    """
    what = args.get("what")
    if not isinstance(what, str) or what.strip().lower() not in LIST_TARGETS:
        return Rejection(f"list_items what must be one of {sorted(LIST_TARGETS)}", raw)
    what = what.strip().lower()

    out: dict[str, Any] = {"what": what}
    where = args.get("where")
    if what == "files":
        if not isinstance(where, str) or not where.strip():
            return Rejection("list_items what='files' needs a folder", raw)
        resolved = ALLOWED_PATHS.get(where.strip().lower())
        if resolved is None:
            speech = f"I can only list the standard folders, not {where.strip()}."
            return Rejection(f"list folder {where!r} not in allowlist", raw, speech)
        out["where"] = str(resolved)
    return IntentCall(Intent.LIST_ITEMS, out, raw)


def _check_open_folder(args: dict, raw: str) -> IntentCall | Rejection:
    """A folder NAME to search for, never a path.

    `open_path` handles the eight standard folders. This is the other case --
    "open the audio_lab folder" -- and it resolves through Spotlight scoped to
    the home directory, exactly like find_file. Same rule applies: the model
    supplies search terms, and anything path-shaped is refused here so it can
    never smuggle one through.
    """
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return Rejection("open_folder needs a name", raw)
    cleaned = name.strip()
    if cleaned.startswith(("/", "~")) or ".." in cleaned:
        return Rejection(f"open_folder name {cleaned!r} looks like a path", raw)
    return IntentCall(Intent.OPEN_FOLDER, {"name": cleaned[:MAX_FOLDER_QUERY_CHARS]}, raw)


def _check_reminder(args: dict, raw: str) -> IntentCall | Rejection:
    """Reminders and alarms are the same thing: text plus a time.

    The model reports the clock face it HEARD -- hour, minute, meridiem, day --
    and Python builds the datetime, for the same reason timers report
    {value, unit} instead of seconds: a 1.7B model doing date arithmetic is a
    silent-error generator. A reminder with no time is still valid; Reminders.app
    accepts an undated one.
    """
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return Rejection("create_reminder needs non-empty text", raw)

    out: dict[str, Any] = {"text": text.strip()[:MAX_REMINDER_CHARS]}

    hour = _as_int(args.get("hour"))
    if hour is not None:
        if not 0 <= hour <= 23:
            return Rejection(f"reminder hour {hour} out of range 0..23", raw)
        minute = _as_int(args.get("minute")) or 0
        if not 0 <= minute <= 59:
            return Rejection(f"reminder minute {minute} out of range 0..59", raw)

        meridiem = args.get("meridiem")
        if isinstance(meridiem, str) and meridiem.strip().lower() in ("am", "pm"):
            meridiem = meridiem.strip().lower()
            if not 1 <= hour <= 12:
                return Rejection(f"hour {hour} with a meridiem must be 1..12", raw)
            hour = hour % 12 + (12 if meridiem == "pm" else 0)

        out["hour"] = hour
        out["minute"] = minute

    day = args.get("day")
    if isinstance(day, str) and day.strip().lower() in REMINDER_DAYS:
        out["day"] = day.strip().lower()
    return IntentCall(Intent.CREATE_REMINDER, out, raw)


_CHECKERS = {
    Intent.SET_TIMER: _check_timer,
    Intent.OPEN_APP: _check_open_app,
    Intent.CLOSE_APP: _check_close_app,
    Intent.SET_VOLUME: _check_volume,
    Intent.CAPTURE_NOTE: _check_note,
    Intent.SEARCH_NOTES: _check_search,
    Intent.FIND_FILE: _check_find_file,
    Intent.OPEN_PATH: _check_open_path,
    Intent.MEDIA_CONTROL: _check_media,
    Intent.GET_STATUS: _check_status,
    Intent.LIST_ITEMS: _check_list_items,
    Intent.OPEN_FOLDER: _check_open_folder,
    Intent.CREATE_REMINDER: _check_reminder,
}
