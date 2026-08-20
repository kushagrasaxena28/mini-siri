"""Reminders.app, via AppleScript.

Alarms and reminders are the same capability here. An in-process alarm would die
with the process exactly as timers do, and "wake me at 2am" is precisely the case
where that is useless -- so both route to Reminders.app, which survives a restart
and syncs to the user's other devices.

Like Notes, this needs macOS Automation permission and is only requested on first
use. Unlike Notes there is no local fallback: a reminder that only exists inside
a process that has exited is not a reminder, so an unavailable Reminders.app is
reported honestly rather than papered over.

THERE IS NO DELETE PATH. Same guarantee as notes.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from ..schema.intents import WEEKDAYS
from . import macos


class RemindersUnavailableError(RuntimeError):
    """Reminders.app could not be reached -- usually a missing Automation grant."""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def resolve_datetime(
    hour: int | None, minute: int, day: str | None, now: datetime | None = None
) -> datetime | None:
    """Clock face -> absolute datetime, choosing the NEXT occurrence.

    "set an alarm for 2 am" said at 11 pm means 2 am tomorrow, not 2 am today,
    which has already passed. A weekday name resolves the same way: forward.

    Python does this rather than the model for the same reason it does timer
    arithmetic: date maths is where a 1.7B model silently produces something
    plausible and wrong.
    """
    if hour is None:
        return None
    now = now or datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day == "tomorrow":
        return target + timedelta(days=1)
    if day == "today":
        return target
    if day in WEEKDAYS:
        # Next occurrence of that weekday. Saying "Tuesday" ON Tuesday means the
        # one coming, not the moment that just passed.
        ahead = (WEEKDAYS.index(day) - target.weekday()) % 7
        return target + timedelta(days=ahead or 7)
    return target + timedelta(days=1) if target <= now else target


def create(text: str, when: datetime | None) -> str:
    """Create a reminder. Returns a human-readable description of when it fires."""
    properties = [f'name:"{_escape(text)}"']
    date_setup = ""
    if when is not None:
        # Build the date from components rather than parsing a string: AppleScript's
        # `date "..."` is locale-dependent and silently misreads day/month order.
        date_setup = (
            "  set theDate to current date\n"
            f"  set year of theDate to {when.year}\n"
            f"  set month of theDate to {when.month}\n"
            f"  set day of theDate to {when.day}\n"
            f"  set hours of theDate to {when.hour}\n"
            f"  set minutes of theDate to {when.minute}\n"
            "  set seconds of theDate to 0\n"
        )
        properties.append("remind me date:theDate")

    script = (
        'tell application "Reminders"\n'
        f"{date_setup}"
        f"  make new reminder with properties {{{', '.join(properties)}}}\n"
        "end tell"
    )
    succeeded, output = macos.run_osascript(script, timeout_s=10.0)
    if not succeeded:
        raise RemindersUnavailableError(output)
    return when.strftime("%-I:%M %p").lstrip() if when else ""


def is_available() -> bool:
    succeeded, _ = macos.run_osascript(
        'tell application "Reminders" to count lists', timeout_s=25.0
    )
    return succeeded
