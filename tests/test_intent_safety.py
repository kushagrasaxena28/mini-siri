"""The validation boundary and executor safety.

Focus is the fail-closed path. An always-listening system that mis-fires on
overheard speech is worse than useless, so every way a bad model output could
reach an action is tested here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_siri_local.schema.intents import (
    ALLOWED_APPS,
    Intent,
    IntentCall,
    Rejection,
    validate,
)

# --- parsing tolerance -------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"get_time","args":{}}',
        '```json\n{"intent":"get_time","args":{}}\n```',
        'Sure! {"intent":"get_time","args":{}} hope that helps',
        '  \n {"intent":"get_time","args":{}}  ',
    ],
)
def test_accepts_json_in_common_wrappers(raw):
    """Models wrap JSON in fences and prose even when told not to."""
    assert isinstance(validate(raw), IntentCall)


# --- fail closed -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        "{",
        '{"args":{}}',  # no intent
        '{"intent":"delete_everything","args":{}}',  # not in the enum
        '{"intent":"set_timer","args":{}}',  # missing required arg
        '{"intent":"set_timer","args":{"duration_seconds":0}}',
        '{"intent":"set_timer","args":{"duration_seconds":-5}}',
        '{"intent":"set_timer","args":{"duration_seconds":999999}}',
        '{"intent":"set_timer","args":{"duration_seconds":"soon"}}',
        '{"intent":"open_app","args":{"app_name":"Keychain Access"}}',
        '{"intent":"open_path","args":{"path":"/etc/passwd"}}',
        '{"intent":"set_volume","args":{"direction":"sideways"}}',
        '{"intent":"set_volume","args":{"direction":"set"}}',  # level required
        '{"intent":"set_volume","args":{"direction":"set","level":500}}',
        '{"intent":"media_control","args":{"action":"self_destruct"}}',
        '{"intent":"capture_note","args":{"text":""}}',
        '{"intent":"get_time","args":"not an object"}',
    ],
)
def test_invalid_output_is_rejected(raw):
    """Anything malformed produces a Rejection and therefore NO action."""
    assert isinstance(validate(raw), Rejection), f"should have rejected: {raw!r}"


def test_unknown_intent_name_is_never_coerced():
    """No nearest-match guessing -- guessing an action is the failure we prevent."""
    r = validate('{"intent":"set_timerr","args":{"duration_seconds":60}}')
    assert isinstance(r, Rejection)


def test_unknown_is_a_valid_intent_not_a_rejection():
    """Declining is a success path, not an error."""
    r = validate('{"intent":"unknown","args":{}}')
    assert isinstance(r, IntentCall) and r.intent is Intent.UNKNOWN


# --- allowlists --------------------------------------------------------------


def test_app_names_resolve_through_allowlist():
    r = validate('{"intent":"open_app","args":{"app_name":"terminal"}}')
    assert isinstance(r, IntentCall)
    assert r.args["app_name"] == "Terminal"  # canonical macOS name, not model text


def test_arbitrary_paths_are_refused():
    for path in ("/etc/passwd", "~/.ssh", "../../secrets", "/System"):
        r = validate(f'{{"intent":"open_path","args":{{"path":"{path}"}}}}')
        assert isinstance(r, Rejection)


def test_allowlist_keys_are_lowercase_for_matching():
    """Lookup lowercases the model's output, so keys must be lowercase or the
    entry is unreachable and every request for that app silently fails."""
    assert all(k == k.lower() for k in ALLOWED_APPS)


def test_allowlist_values_are_nonempty_and_clean():
    # Values go straight to `open -a`, so leading/trailing whitespace breaks them.
    # (Not asserting capitalisation: "iTerm" is genuinely lowercase-initial.)
    assert all(v and v == v.strip() for v in ALLOWED_APPS.values())


# --- executor safety ---------------------------------------------------------


def test_executor_has_no_destructive_capabilities():
    """Destructive verbs must be ABSENT from the module, not merely unregistered."""
    import inspect

    from mini_siri_local.executor import handlers

    src = inspect.getsource(handlers).lower()
    for forbidden in ("shutil.rmtree", "os.remove", "os.unlink", "shell=true", "eval(", "exec("):
        assert forbidden not in src, f"destructive capability present: {forbidden}"


def test_every_intent_has_a_handler(executor):
    """A missing handler would raise KeyError at dispatch, mid-turn."""
    for intent in Intent:
        result = executor.execute(IntentCall(intent, _ARGS_FOR.get(intent, {})))
        assert result is not None


_ARGS_FOR = {
    Intent.SET_TIMER: {"duration_seconds": 1},
    Intent.OPEN_APP: {"app_name": "Terminal"},
    Intent.OPEN_PATH: {"path": str(Path.home())},
    Intent.SET_VOLUME: {"direction": "down"},
    Intent.MEDIA_CONTROL: {"action": "pause"},
    Intent.CAPTURE_NOTE: {"text": "test note"},
    Intent.SEARCH_NOTES: {"query": "test"},
}


def test_unknown_intent_produces_no_speech(executor):
    """Confirming every stray sentence in the room would make it unusable."""
    result = executor.execute(IntentCall(Intent.UNKNOWN, {}))
    assert result.ok and result.speech == ""


def test_executor_reports_failure_honestly(executor):
    """Handler exceptions surface as ok=False, never silently swallowed."""
    result = executor.execute(IntentCall(Intent.OPEN_PATH, {"path": "/nonexistent/xyz"}))
    assert result.ok is False


# --- timer schema: arithmetic belongs to Python, not the model ----------------


@pytest.mark.parametrize(
    ("raw", "expected_seconds"),
    [
        ('{"intent":"set_timer","args":{"value":3,"unit":"hours"}}', 10800),
        ('{"intent":"set_timer","args":{"value":45,"unit":"minutes"}}', 2700),
        ('{"intent":"set_timer","args":{"value":1,"unit":"hours"}}', 3600),
        ('{"intent":"set_timer","args":{"value":30,"unit":"seconds"}}', 30),
        # singular units: models emit "hour" as readily as "hours"
        ('{"intent":"set_timer","args":{"value":1,"unit":"hour"}}', 3600),
        ('{"intent":"set_timer","args":{"value":2,"unit":"minute"}}', 120),
        # legacy form stays supported so older checkpoints keep working
        ('{"intent":"set_timer","args":{"duration_seconds":600}}', 600),
    ],
)
def test_timer_units_convert_exactly(raw, expected_seconds):
    """The 10x digit errors that motivated this schema (3 hours -> 1080) are
    impossible once Python does the multiplication."""
    result = validate(raw)
    assert isinstance(result, IntentCall)
    assert result.args["duration_seconds"] == expected_seconds


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"set_timer","args":{"value":99,"unit":"hours"}}',  # over 24h cap
        '{"intent":"set_timer","args":{"value":3,"unit":"fortnights"}}',
        '{"intent":"set_timer","args":{"value":0,"unit":"minutes"}}',
        '{"intent":"set_timer","args":{"unit":"minutes"}}',  # no value
        '{"intent":"set_timer","args":{"value":"three","unit":"hours"}}',
    ],
)
def test_invalid_timer_specs_are_rejected(raw):
    assert isinstance(validate(raw), Rejection)


# --- close_app: doing the OPPOSITE of the request is worse than declining ------


def test_close_app_uses_the_same_allowlist_as_open():
    """We only ever quit apps we would also open."""
    result = validate('{"intent":"close_app","args":{"app_name":"whatsapp"}}')
    assert isinstance(result, IntentCall)
    assert result.args["app_name"] == "WhatsApp"


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"close_app","args":{}}',
        '{"intent":"close_app","args":{"app_name":""}}',
        '{"intent":"close_app","args":{"app_name":"Keychain Access"}}',
    ],
)
def test_invalid_close_app_is_rejected(raw):
    assert isinstance(validate(raw), Rejection)


# --- find_file: the query is SEARCH TERMS, never a path -----------------------


def test_find_file_accepts_search_terms():
    result = validate('{"intent":"find_file","args":{"query":"resume pdf"}}')
    assert isinstance(result, IntentCall)
    assert result.args["query"] == "resume pdf"


@pytest.mark.parametrize(
    "raw",
    [
        '{"intent":"find_file","args":{"query":"/etc/passwd"}}',
        '{"intent":"find_file","args":{"query":"~/.ssh/id_rsa"}}',
        '{"intent":"find_file","args":{"query":"../../secrets"}}',
        '{"intent":"find_file","args":{"query":""}}',
        '{"intent":"find_file","args":{}}',
    ],
)
def test_find_file_rejects_path_like_queries(raw):
    """Spotlight is scoped to the home directory, but rejecting path-shaped input
    here stops the model smuggling one through in the first place."""
    assert isinstance(validate(raw), Rejection)


def test_notes_store_has_no_delete_path():
    """Neither notes backend can destroy a note -- there is no delete method at
    all, so an injected intent has nothing to reach."""
    import inspect

    from mini_siri_local.executor import notes

    source = inspect.getsource(notes).lower()
    for forbidden in ("delete note", "remove note", "def delete", "unlink"):
        assert forbidden not in source, f"delete capability present: {forbidden}"


# --- find_file is the one intent whose target the MODEL picks -----------------


@pytest.mark.parametrize(
    ("filename", "expect_opened"),
    [
        ("resume.pdf", True),
        ("holiday.jpg", True),
        ("notes.md", True),
        ("install.command", False),
        ("Some App.app", False),
        ("setup.sh", False),
        ("build.scpt", False),
        ("installer.pkg", False),
        ("disk.dmg", False),
        ("payload", False),  # no extension at all
    ],
)
def test_find_file_never_opens_an_executable_type(
    executor, system_calls, monkeypatch, filename, expect_opened
):
    """`open` RUNS .command/.app/.pkg rather than displaying them. Anything not
    known-viewable is revealed in Finder instead."""
    from mini_siri_local.executor import macos

    path = f"/Users/someone/{filename}"
    monkeypatch.setattr(macos, "spotlight_search", lambda q, limit, search_root: [path])
    monkeypatch.setattr(macos, "reveal_file_path", lambda p: system_calls.append(["open", "-R", p]))

    result = executor.execute(IntentCall(Intent.FIND_FILE, {"query": "x"}))
    assert result.ok
    assert system_calls == [["open", path] if expect_opened else ["open", "-R", path]]
