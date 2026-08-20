"""The capabilities added after live testing: get_status, list_items,
open_folder, create_reminder.

Grouped by argument rather than one intent per capability, so most of the risk
sits in the ARGUMENT validation -- an out-of-enum item or a folder outside the
allowlist has to fail closed exactly like everything else.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mini_siri_local.executor.reminders import resolve_datetime
from mini_siri_local.schema.intents import (
    STATUS_ITEMS,
    Intent,
    IntentCall,
    Rejection,
    validate,
)

# --- get_status --------------------------------------------------------------


@pytest.mark.parametrize("item", sorted(STATUS_ITEMS))
def test_every_status_item_validates_and_has_a_handler(item, executor):
    result = validate(f'{{"intent":"get_status","args":{{"item":"{item}"}}}}')
    assert isinstance(result, IntentCall)
    assert executor.execute(result) is not None


@pytest.mark.parametrize("raw", [
    '{"intent":"get_status","args":{"item":"password"}}',
    '{"intent":"get_status","args":{"item":""}}',
    '{"intent":"get_status","args":{}}',
])
def test_unknown_status_item_is_rejected(raw):
    assert isinstance(validate(raw), Rejection)


def test_date_and_day_do_not_route_to_get_time():
    """The silently-wrong failure this exists to prevent: "what day is it"
    answered with the clock time."""
    for item in ("date", "day"):
        result = validate(f'{{"intent":"get_status","args":{{"item":"{item}"}}}}')
        assert isinstance(result, IntentCall)
        assert result.intent is Intent.GET_STATUS


# --- list_items --------------------------------------------------------------


def test_listing_notes_needs_no_folder():
    result = validate('{"intent":"list_items","args":{"what":"notes"}}')
    assert isinstance(result, IntentCall)
    assert "where" not in result.args


def test_listing_files_resolves_an_allowlist_key():
    result = validate('{"intent":"list_items","args":{"what":"files","where":"downloads"}}')
    assert isinstance(result, IntentCall)
    assert result.args["where"].endswith("/Downloads")


@pytest.mark.parametrize("raw", [
    '{"intent":"list_items","args":{"what":"files","where":"/etc"}}',
    '{"intent":"list_items","args":{"what":"files","where":"../../"}}',
    '{"intent":"list_items","args":{"what":"files"}}',        # no folder
    '{"intent":"list_items","args":{"what":"passwords"}}',
])
def test_listing_outside_the_allowlist_is_rejected(raw):
    assert isinstance(validate(raw), Rejection)


# --- open_folder -------------------------------------------------------------


def test_open_folder_takes_a_name():
    result = validate('{"intent":"open_folder","args":{"name":"audio_lab"}}')
    assert isinstance(result, IntentCall)
    assert result.args["name"] == "audio_lab"


@pytest.mark.parametrize("name", ["/etc", "~/.ssh", "../../secrets", ""])
def test_open_folder_refuses_path_shaped_names(name):
    """Same rule as find_file: the model supplies a name, never a path."""
    assert isinstance(validate(f'{{"intent":"open_folder","args":{{"name":"{name}"}}}}'), Rejection)


def test_open_folder_only_opens_a_directory_under_home(executor, system_calls, monkeypatch):
    from mini_siri_local.executor import macos

    monkeypatch.setattr(
        macos, "spotlight_search_folders", lambda q, limit, search_root: ["/Users/x/proj"]
    )
    result = executor.execute(IntentCall(Intent.OPEN_FOLDER, {"name": "proj"}))
    assert result.ok
    assert system_calls == [["open", "/Users/x/proj"]]


# --- create_reminder ---------------------------------------------------------


def test_meridiem_is_converted_to_24_hour():
    result = validate(
        '{"intent":"create_reminder","args":{"text":"call mom","hour":7,"meridiem":"pm"}}'
    )
    assert isinstance(result, IntentCall)
    assert result.args["hour"] == 19


def test_midnight_and_noon_convert_correctly():
    for hour, meridiem, expected in ((12, "am", 0), (12, "pm", 12)):
        result = validate(
            f'{{"intent":"create_reminder","args":'
            f'{{"text":"x","hour":{hour},"meridiem":"{meridiem}"}}}}'
        )
        assert isinstance(result, IntentCall)
        assert result.args["hour"] == expected, f"{hour}{meridiem} should be {expected}"


@pytest.mark.parametrize("raw", [
    '{"intent":"create_reminder","args":{"text":"","hour":2}}',
    '{"intent":"create_reminder","args":{"hour":2}}',
    '{"intent":"create_reminder","args":{"text":"x","hour":25}}',
    '{"intent":"create_reminder","args":{"text":"x","hour":2,"minute":99}}',
    '{"intent":"create_reminder","args":{"text":"x","hour":19,"meridiem":"pm"}}',  # 19 pm
])
def test_invalid_reminder_specs_are_rejected(raw):
    assert isinstance(validate(raw), Rejection)


def test_reminder_without_a_time_is_still_valid():
    result = validate('{"intent":"create_reminder","args":{"text":"buy milk"}}')
    assert isinstance(result, IntentCall)
    assert "hour" not in result.args


def test_a_time_already_past_rolls_to_tomorrow():
    """"Set an alarm for 2am" said at 11pm means tomorrow. Python decides this,
    not the model -- date arithmetic is where a 1.7B model is silently wrong."""
    late = datetime(2026, 8, 19, 23, 0).astimezone()
    assert resolve_datetime(2, 0, None, late).day == 20
    assert resolve_datetime(23, 30, None, late).day == 19  # still ahead today


def test_explicit_day_overrides_the_next_occurrence_rule():
    late = datetime(2026, 8, 19, 23, 0).astimezone()
    assert resolve_datetime(2, 0, "today", late).day == 19
    assert resolve_datetime(2, 0, "tomorrow", late).day == 20


def test_no_hour_means_no_datetime():
    assert resolve_datetime(None, 0, None) is None
