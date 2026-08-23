"""The Kokoro download patterns must satisfy mlx_audio's own loader.

mlx_audio calls snapshot_download itself at load time, and huggingface_hub then verifies
that snapshot is COMPLETE for the patterns it asked for. Fetching less than mlx_audio
demands does not degrade -- it makes the model unloadable offline with
IncompleteSnapshotError, and only on a machine whose cache was not already populated.

That is exactly how it shipped broken once: the trimmed download looked fine locally
because the full repo was already cached from before the change.
"""

from __future__ import annotations

import fnmatch
import sys

import pytest

sys.path.insert(0, "scripts")


def _kokoro_patterns() -> list[str]:
    from download_models import HF_REPOS

    for repo, _purpose, patterns, _is_tts in HF_REPOS:
        if "Kokoro" in repo:
            return patterns
    pytest.fail("no Kokoro entry in HF_REPOS")


def test_patterns_cover_everything_mlx_audio_asks_for():
    """Our allow_patterns must be a superset of mlx_audio's DEFAULT_ALLOW_PATTERNS."""
    from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS

    ours = _kokoro_patterns()
    # A file mlx_audio would request must match at least one of our patterns. Checked with
    # representative names rather than the live repo listing, so this needs no network.
    samples = (
        "config.json",
        "kokoro-v1_0.safetensors",
        "voices/af_heart.safetensors",
        "voices/zm_yunyang.safetensors",
    )
    for name in samples:
        demanded = any(fnmatch.fnmatch(name, p) for p in DEFAULT_ALLOW_PATTERNS)
        fetched = any(fnmatch.fnmatch(name, p) for p in ours)
        if demanded:
            assert fetched, (
                f"{name} is requested by mlx_audio but our patterns {ours} would skip it; "
                "the snapshot would be incomplete and fail to load offline"
            )


def test_patterns_still_skip_what_is_never_loaded():
    """The point of the patterns: demo audio and duplicate .pt voices stay unfetched."""
    ours = _kokoro_patterns()
    for name in ("samples/HEARME.wav", "voices/af_heart.pt", "README.md"):
        assert not any(fnmatch.fnmatch(name, p) for p in ours), f"{name} should be skipped"
