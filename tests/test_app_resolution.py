"""App names resolve against installed apps, not just a fixed dictionary.

Before this, "open chess" was rejected outright even though Chess.app was sat
in /Applications the whole time -- ALLOWED_APPS was the only source of truth.
Now ALLOWED_APPS supplies ALIASES (spoken short forms that don't match a bundle
name), and anything else is checked against apps that genuinely exist on this
machine. The filesystem is faked in every test here so behaviour does not
depend on what happens to be installed on whichever machine runs the suite.
"""

from __future__ import annotations

import pytest

from mini_siri_local.schema.intents import (
    ALLOWED_APPS,
    IntentCall,
    Rejection,
    _installed_apps,
    resolve_app_name,
    validate,
)


@pytest.fixture(autouse=True)
def fake_installed_apps(monkeypatch, tmp_path):
    """Real .app bundles in a scratch directory, standing in for /Applications.
    autouse so no test in this file can accidentally hit the real filesystem."""
    _installed_apps.cache_clear()
    for name in ("Chess", "Xcode", "Freeform"):
        (tmp_path / f"{name}.app").mkdir()
    (tmp_path / "not an app.txt").write_text("")  # non-.app entries must be ignored
    from mini_siri_local import schema as schema_module

    monkeypatch.setattr(
        schema_module.intents, "APP_SEARCH_DIRS", (tmp_path,), raising=False
    )
    yield
    _installed_apps.cache_clear()


def test_alias_still_resolves_before_the_filesystem_scan():
    """ALLOWED_APPS entries must win even when the scan directory doesn't
    contain a bundle with that exact spoken name."""
    assert resolve_app_name("chrome") == "Google Chrome"


def test_installed_app_not_in_the_alias_table_resolves():
    assert resolve_app_name("chess") == "Chess"
    assert resolve_app_name("Chess") == "Chess"  # case-insensitive
    assert resolve_app_name("  chess  ") == "Chess"  # whitespace-tolerant


def test_uninstalled_app_does_not_resolve():
    assert resolve_app_name("photoshop") is None


def test_non_app_files_are_never_matched():
    assert resolve_app_name("not an app") is None


def test_open_app_accepts_a_dynamically_resolved_name():
    result = validate('{"intent":"open_app","args":{"app_name":"chess"}}')
    assert isinstance(result, IntentCall)
    assert result.args["app_name"] == "Chess"


def test_close_app_uses_the_same_resolution_as_open_app():
    result = validate('{"intent":"close_app","args":{"app_name":"xcode"}}')
    assert isinstance(result, IntentCall)
    assert result.args["app_name"] == "Xcode"


def test_unresolved_app_is_rejected_with_safe_speech():
    """The spoken decline names only what the user said -- never the mechanism
    (allowlist vs filesystem) that decided it."""
    result = validate('{"intent":"open_app","args":{"app_name":"photoshop"}}')
    assert isinstance(result, Rejection)
    assert "photoshop" in result.speech.lower()
    assert "allowlist" not in result.speech.lower()
    assert "filesystem" not in result.speech.lower()


def test_missing_app_directory_does_not_crash(monkeypatch, tmp_path):
    """A directory that doesn't exist (no /System/Applications/Utilities on
    some setups, a typo, permissions) must be skipped, not raised."""
    _installed_apps.cache_clear()
    from mini_siri_local import schema as schema_module

    monkeypatch.setattr(
        schema_module.intents,
        "APP_SEARCH_DIRS",
        (tmp_path / "does-not-exist",),
        raising=False,
    )
    assert resolve_app_name("anything") is None
    _installed_apps.cache_clear()


def test_installed_apps_cache_is_computed_once(monkeypatch, tmp_path):
    """The scan is meant to happen once per process, not once per turn."""
    calls = []
    real_iterdir = type(tmp_path).iterdir

    def counting_iterdir(self):
        calls.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(type(tmp_path), "iterdir", counting_iterdir)
    _installed_apps.cache_clear()
    resolve_app_name("chess")
    resolve_app_name("chess")
    resolve_app_name("xcode")
    assert len(calls) == 1, f"scanned the filesystem {len(calls)} times, expected 1"
    _installed_apps.cache_clear()


def test_generic_decline_is_unaffected_for_non_app_rejections():
    """Rejection.speech defaults sensibly for checkers that don't set it."""
    result = validate("not json at all")
    assert isinstance(result, Rejection)
    assert result.speech == "Sorry, I didn't catch that."


def test_allowlist_keys_are_lowercase_for_matching():
    assert all(k == k.lower() for k in ALLOWED_APPS)
