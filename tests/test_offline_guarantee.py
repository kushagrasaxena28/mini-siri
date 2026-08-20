"""The privacy claim, as a test.

README: "No network calls at runtime. Turn off Wi-Fi and it still works."

That was false before `config.enforce_offline()` existed -- huggingface_hub
revalidated every repo on load, costing ~850 ms of cold start and quietly making
the headline claim untrue. An unenforceable claim is marketing, so it gets a
test that fails if anything opens a connection.

Slow: loads every model.
    uv run pytest tests/test_offline_guarantee.py -v -m slow
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.slow

CLIP = Path("benchmarks/datasets/synth/timer.wav")


class NetworkAccessError(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Creating a socket is harmless; CONNECTING is the network call."""

    def refuse(self, address, *args, **kwargs):
        raise NetworkAccessError(f"opened a connection to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


def test_enforce_offline_sets_the_hub_flag():
    import os

    from mini_siri_local.config import enforce_offline

    enforce_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_enforce_offline_respects_an_explicit_override(monkeypatch):
    """A developer who deliberately wants the Hub must be able to say so --
    setdefault, not overwrite."""
    from mini_siri_local.config import enforce_offline

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    enforce_offline()
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_a_full_turn_opens_no_connection(no_network, system_calls, tmp_path):
    """Load every model and run audio -> action with the network refused."""
    from mini_siri_local.asr.offline import OfflineAsr
    from mini_siri_local.config import AsrConfig, SlmConfig, enforce_offline
    from mini_siri_local.executor.handlers import Executor
    from mini_siri_local.executor.notes import LocalNoteStore, NoteStore
    from mini_siri_local.schema.intents import IntentCall, validate
    from mini_siri_local.slm.parser import SlmParser
    from mini_siri_local.vad.silero import SileroVad

    enforce_offline()

    asr = OfflineAsr(AsrConfig())
    vad = SileroVad()
    slm = SlmParser(SlmConfig())
    notes = NoteStore(prefer_apple_notes=False)
    notes.local = LocalNoteStore(tmp_path / "notes.jsonl")
    executor = Executor(notify=lambda message: None, notes=notes)

    audio, _ = sf.read(str(CLIP), dtype="float32")
    from mini_siri_local.config import VAD_FRAME_SAMPLES

    probabilities = [
        vad(audio[i : i + VAD_FRAME_SAMPLES])
        for i in range(0, len(audio) - VAD_FRAME_SAMPLES + 1, VAD_FRAME_SAMPLES)
    ]
    assert max(probabilities) > 0.9, "VAD saw no speech; the turn under test never happened"

    text = asr.transcribe(np.asarray(audio, dtype=np.float32))
    assert text.strip(), "ASR produced nothing; the turn under test never happened"

    result = validate(slm.parse(text).raw)
    assert isinstance(result, IntentCall), f"model output rejected: {result.reason}"
    assert executor.execute(result).ok
    executor.shutdown()
