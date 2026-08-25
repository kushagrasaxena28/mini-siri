"""Which paths the uninstaller is willing to delete.

This script removes gigabytes, and the Hugging Face cache is SHARED -- someone else's
models sit next to ours in the same directory. The two properties worth pinning are
therefore about restraint, not about deletion working:

  * it targets only the three repos this project downloads
  * dictated notes are user data and need an explicit opt-in

No deletion happens here. The targets() function is pure path selection, so it can be
checked without touching a filesystem the test does not own.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "scripts")


@pytest.fixture
def uninstall(monkeypatch, tmp_path):
    import uninstall as mod

    hub = tmp_path / "hub"
    for name in (*mod.MODEL_REPOS, "models--openai--whisper-large", "models--meta--llama-3"):
        (hub / name).mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    monkeypatch.setattr(mod, "hf_hub_cache", lambda: hub)
    monkeypatch.setattr(mod, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path / "repo")
    return mod


def test_only_this_projects_models_are_targeted(uninstall):
    chosen = {p.name for p, _ in uninstall.targets(include_notes=False)}
    assert chosen == set(uninstall.MODEL_REPOS)


def test_other_peoples_models_are_never_targeted(uninstall):
    chosen = {p.name for p, _ in uninstall.targets(include_notes=True)}
    for stranger in ("models--openai--whisper-large", "models--meta--llama-3"):
        assert stranger not in chosen, f"would have deleted {stranger}, which is not ours"


def test_notes_require_an_explicit_flag(uninstall):
    without = {p for p, _ in uninstall.targets(include_notes=False)}
    with_notes = {p for p, _ in uninstall.targets(include_notes=True)}
    assert uninstall.NOTES_DIR not in without
    assert uninstall.NOTES_DIR in with_notes


def test_notes_are_labelled_as_user_data(uninstall):
    """The label is what a user reads before confirming; it must not look like a cache."""
    label = next(d for p, d in uninstall.targets(include_notes=True) if p == uninstall.NOTES_DIR)
    assert "USER DATA" in label


def test_model_list_matches_what_download_models_fetches():
    """If a model is added to the downloader and not here, uninstall silently leaks it."""
    import download_models
    import uninstall as mod

    downloaded = {
        "models--" + repo.replace("/", "--") for repo, *_ in download_models.HF_REPOS
    }
    assert downloaded == set(mod.MODEL_REPOS)


def test_hf_cache_location_honours_env_overrides(monkeypatch, tmp_path):
    import uninstall as mod

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert mod.hf_hub_cache() == tmp_path / "explicit"

    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert mod.hf_hub_cache() == tmp_path / "home" / "hub"
