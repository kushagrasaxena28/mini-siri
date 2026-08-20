"""Note storage, backed by Apple Notes with a local fallback.

Two backends behind one interface:

  AppleNotesStore  -- writes real notes into Notes.app, so they sync to your
                      phone and are readable outside this app. The default.
  LocalNoteStore   -- append-only JSONL under ~/.mini-siri-local. Used when Notes.app is
                      unavailable or the user has not granted automation access.

WHY A FALLBACK EXISTS
---------------------
Talking to Notes.app needs macOS Automation permission, which is a separate
grant from the microphone and is only requested on first use. If the user
declines -- or the prompt never appears, which happens for unsigned processes --
notes must still be captured rather than silently lost.

NEITHER BACKEND CAN DELETE. There is no delete path in this module at all, so a
hallucinated or prompt-injected intent has nothing destructive to reach
(see ENGINEERING.md section 2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import macos

DEFAULT_NOTES_PATH = Path.home() / ".mini-siri-local" / "notes.jsonl"
MIN_KEYWORD_LENGTH = 3  # shorter tokens ("the", "a") match nearly everything
NOTE_TITLE_WORDS = 6  # words taken from the body to title an Apple note
# Non-printable ASCII, effectively untypable into Notes.app -- used to pull
# {title, body} pairs across the AppleScript bridge without ambiguity.
_RECORD_SEP_CODE = 30  # separates one note from the next
_FIELD_SEP_CODE = 31  # separates a note's title from its body


@dataclass(frozen=True)
class Note:
    text: str
    timestamp: str

    @classmethod
    def from_json(cls, line: str) -> Note | None:
        try:
            data = json.loads(line)
            return cls(text=data["text"], timestamp=data.get("timestamp", ""))
        except (json.JSONDecodeError, KeyError):
            return None  # skip corrupt lines rather than failing the whole search


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _keywords(query: str) -> list[str]:
    return [w for w in query.lower().split() if len(w) >= MIN_KEYWORD_LENGTH]


def _has_word(text: str, keyword: str) -> bool:
    """Whole-word match: a query for "one" must not hit "phone" or "someone".

    Found in live testing: "chapter one" returned 17 notes and read out an
    unrelated one, because the previous substring check matched "one" inside
    almost any note.
    """
    return re.search(rf"\b{re.escape(keyword)}\b", text.lower()) is not None


def _rank_matches(records: list[tuple[str, str]], keywords: list[str]) -> list[str]:
    """Titles of notes containing EVERY keyword, falling back to any of them.

    Whole-word matching alone was not enough. Measured against a real library,
    "chapter one" still returned 13 notes under any-keyword matching, because
    "one" is a legitimate word in ten unrelated ones. Requiring every keyword
    cuts that to 3.

    The fallback exists because strictness alone loses real hits: "meeting
    budget" matches every-keyword in zero notes but any-keyword in two, and
    returning nothing is a worse answer than returning something adjacent.

    `records` is (title, text_to_search).
    """
    all_matched = [title for title, text in records if all(_has_word(text, k) for k in keywords)]
    if all_matched:
        return all_matched
    return [title for title, text in records if any(_has_word(text, k) for k in keywords)]


def _escape_applescript(text: str) -> str:
    r"""Escape a Python string for embedding in an AppleScript literal.

    Backslashes first, then quotes -- reversing the order would double-escape
    the backslashes introduced by the quote pass.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


class LocalNoteStore:
    """Append-only JSONL. Always available, never syncs anywhere."""

    backend_name = "local file"

    def __init__(self, path: Path = DEFAULT_NOTES_PATH):
        self.path = path

    def append(self, text: str) -> Note:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        note = Note(text=text, timestamp=_now_iso())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"text": note.text, "timestamp": note.timestamp}) + "\n")
        return note

    def all(self) -> list[Note]:
        if not self.path.exists():
            return []
        parsed = (Note.from_json(line) for line in self.path.read_text("utf-8").splitlines())
        return [note for note in parsed if note is not None]

    def search(self, query: str) -> list[Note]:
        """Whole-word keyword match, newest first. See _rank_matches for why it
        prefers notes containing every keyword before falling back."""
        keywords = _keywords(query)
        if not keywords:
            return []
        notes = self.all()
        wanted = set(_rank_matches([(n.text, n.text) for n in notes], keywords))
        return list(reversed([n for n in notes if n.text in wanted]))

    @property
    def is_available(self) -> bool:
        return True


class AppleNotesStore:
    """Real notes in Notes.app, via AppleScript."""

    backend_name = "Apple Notes"

    def append(self, text: str) -> Note:
        # Notes.app renders HTML; the first line becomes the visible title, so
        # lead with a short title derived from the note itself.
        title = " ".join(text.split()[:NOTE_TITLE_WORDS])
        body = _escape_applescript(f"{title}<br><br>{text}")
        script = f'tell application "Notes" to make new note with properties {{body:"{body}"}}'

        succeeded, output = macos.run_osascript(script)
        if not succeeded:
            raise NotesUnavailableError(output)
        return Note(text=text, timestamp=_now_iso())

    def search(self, query: str) -> list[Note]:
        """Match note BODIES on whole words, return note TITLES.

        AppleScript's `whose ... contains` is a substring test with no
        word-boundary option -- and that was a real bug: "chapter one" matched
        seventeen notes because "one" is a substring of "phone" and "someone".

        The `whose` clause is kept as a cheap, over-inclusive PREFILTER, which is
        still worth having: it avoids marshalling every note across the bridge on
        a large library. The real whole-word ranking (`_rank_matches`, the same one
        LocalNoteStore uses) then runs in Python against the fetched body, so both
        backends agree on what counts as a match.
        """
        keywords = _keywords(query)
        if not keywords:
            return []

        conditions = " or ".join(f'body contains "{_escape_applescript(k)}"' for k in keywords)
        record_sep, field_sep = _RECORD_SEP_CODE, _FIELD_SEP_CODE
        # The repeat loop MUST sit inside the tell block. Outside it, `body of n`
        # cannot be resolved against a note reference and AppleScript fails with
        # "Can't make body of item 1 of {...} into type Unicode text (-1700)" --
        # which killed every search against a real Notes library.
        script = (
            f"set RS to (ASCII character {record_sep})\n"
            f"set US to (ASCII character {field_sep})\n"
            'set output to ""\n'
            'tell application "Notes"\n'
            f"  set candidates to every note whose {conditions}\n"
            "  repeat with n in candidates\n"
            "    set output to output & ((name of n) as text) & US "
            "& ((body of n) as text) & RS\n"
            "  end repeat\n"
            "end tell\n"
            "return output"
        )
        succeeded, output = macos.run_osascript(script)
        if not succeeded:
            raise NotesUnavailableError(output)

        records = []
        for record in output.split(chr(record_sep)):
            if chr(field_sep) not in record:
                continue
            title, body = record.split(chr(field_sep), 1)
            if title.strip():
                records.append((title.strip(), body))
        return [Note(text=title, timestamp="") for title in _rank_matches(records, keywords)]

    def all(self) -> list[Note]:
        """Every note title. Titles only -- listing is a count-and-name
        operation, and pulling every body across the bridge to read ten names
        aloud would cost seconds."""
        succeeded, output = macos.run_osascript(
            "set AppleScript's text item delimiters to linefeed\n"
            'tell application "Notes" to set found to name of every note\n'
            "return found as text"
        )
        if not succeeded:
            raise NotesUnavailableError(output)
        titles = [line.strip() for line in output.splitlines() if line.strip()]
        return [Note(text=title, timestamp="") for title in titles]

    @property
    def is_available(self) -> bool:
        succeeded, _ = macos.run_osascript('tell application "Notes" to count notes')
        return succeeded


class NotesUnavailableError(RuntimeError):
    """Notes.app could not be reached -- usually a missing Automation grant."""


class NoteStore:
    """Apple Notes when reachable, local file otherwise.

    Availability is probed ONCE at construction rather than per call: the
    AppleScript round trip is ~200 ms and the answer does not change within a
    session. A failure at write time still falls back, so a permission revoked
    mid-session cannot lose a note.
    """

    def __init__(self, prefer_apple_notes: bool = True):
        self.local = LocalNoteStore()
        self.apple: AppleNotesStore | None = None

        if prefer_apple_notes:
            candidate = AppleNotesStore()
            if candidate.is_available:
                self.apple = candidate

    @property
    def backend_name(self) -> str:
        return self.apple.backend_name if self.apple else self.local.backend_name

    @property
    def path(self) -> Path:
        """Local file path. Present even when Apple Notes is active, because the
        local store is still the fallback target."""
        return self.local.path

    def append(self, text: str) -> Note:
        if self.apple is not None:
            try:
                return self.apple.append(text)
            except NotesUnavailableError as exc:
                print(f"  [notes] Apple Notes unavailable ({exc}); saving locally")
                self.apple = None
        return self.local.append(text)

    def search(self, query: str) -> list[Note]:
        if self.apple is not None:
            try:
                return self.apple.search(query)
            except NotesUnavailableError as exc:
                print(f"  [notes] Apple Notes unavailable ({exc}); searching local notes")
                self.apple = None
        return self.local.search(query)

    def all(self) -> list[Note]:
        if self.apple is not None:
            try:
                return self.apple.all()
            except NotesUnavailableError as exc:
                print(f"  [notes] Apple Notes unavailable ({exc}); listing local notes")
                self.apple = None
        return self.local.all()
