"""Note search must match WHOLE words, not substrings.

Live testing found the bug directly: "chapter one" returned 17 notes and read
out an unrelated one, because "one" is a substring of "phone", "someone", and
"done". Both backends -- LocalNoteStore and AppleNotesStore -- had this bug and
both are tested here, since they used to disagree and are not allowed to again.
"""

from __future__ import annotations

import pytest

from mini_siri_local.executor.notes import AppleNotesStore, LocalNoteStore

# The exact case from the live session, plus the neighbours the substring
# matcher used to catch by accident.
CONTAMINATING_NOTES = [
    "remember to call mom on the phone tomorrow",
    "meeting with someone from finance",
    "the code is done, ship it",
]
TARGET_NOTE = "chapter one starts on page ten"


@pytest.fixture
def local_store(tmp_path):
    store = LocalNoteStore(tmp_path / "notes.jsonl")
    for text in [*CONTAMINATING_NOTES, TARGET_NOTE]:
        store.append(text)
    return store


def test_local_search_excludes_substring_matches(local_store):
    results = [n.text for n in local_store.search("chapter one")]
    assert results == [TARGET_NOTE], f"substring matcher regressed: {results}"


def test_local_search_still_matches_a_real_whole_word(local_store):
    """The fix must not become so strict it stops matching real hits."""
    results = [n.text for n in local_store.search("phone")]
    assert any("phone" in r for r in results)


def test_every_keyword_is_preferred_over_any(local_store):
    """"chapter one" must not drag in every note containing the word "one".
    Measured on a real library: 13 results under any-keyword, 3 under all."""
    local_store.append("one more thing to remember")
    results = [n.text for n in local_store.search("chapter one")]
    assert results == [TARGET_NOTE], f"all-keyword ranking regressed: {results}"


def test_falls_back_to_any_keyword_when_nothing_matches_all(local_store):
    """Strictness alone loses real hits, so a query matching no note on every
    keyword still returns the notes matching some of them."""
    results = [n.text for n in local_store.search("chapter finance")]
    assert TARGET_NOTE in results
    assert any("finance" in r for r in results)


@pytest.mark.parametrize(
    ("note", "query", "should_match"),
    [
        ("chapter one starts on page ten", "one", True),
        ("remember to call mom on the phone", "one", False),
        ("meeting with someone from finance", "one", False),
        ("the code is done ship it", "one", False),
        ("note-1 is about the encoder", "one", False),  # a real limitation, documented
    ],
)
def test_whole_word_matching_cases(note, query, should_match):
    from mini_siri_local.executor.notes import _has_word

    matched = _has_word(note, query)
    assert matched is should_match, f"{query!r} in {note!r}: expected {should_match}, got {matched}"


# --- AppleNotesStore: same bug, fixed via prefilter + Python re-check --------


@pytest.fixture
def apple_store(monkeypatch):
    """Fake `run_osascript` to return the {title US body RS} format the real
    search() now builds and parses, so the AppleScript itself is not under
    test -- the record/field parsing and the whole-word re-check are."""
    from mini_siri_local.executor import notes as notes_module

    notes = [
        ("Phone note", "remember to call mom on the phone tomorrow"),
        ("Chapter one", TARGET_NOTE),
        ("Someone note", "meeting with someone from finance"),
    ]
    record_sep, field_sep = chr(notes_module._RECORD_SEP_CODE), chr(notes_module._FIELD_SEP_CODE)
    encoded = "".join(f"{title}{field_sep}{body}{record_sep}" for title, body in notes)

    def fake_osascript(script, timeout_s=3.0):
        return True, encoded

    monkeypatch.setattr(notes_module.macos, "run_osascript", fake_osascript)
    return AppleNotesStore()


def test_apple_search_excludes_substring_matches(apple_store):
    results = [n.text for n in apple_store.search("chapter one")]
    assert results == ["Chapter one"], f"substring matcher regressed: {results}"


def test_apple_search_unavailable_raises(monkeypatch):
    from mini_siri_local.executor import macos
    from mini_siri_local.executor import notes as notes_module
    from mini_siri_local.executor.notes import NotesUnavailableError

    monkeypatch.setattr(macos, "run_osascript", lambda script, timeout_s=3.0: (False, "denied"))
    with pytest.raises(NotesUnavailableError):
        notes_module.AppleNotesStore().search("anything")
