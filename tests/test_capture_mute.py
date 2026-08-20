"""Mute must release the microphone at the device.

The first version of this control only skipped frames downstream while the
CoreAudio stream stayed open, so macOS kept showing the recording indicator
while the UI said "not listening". These tests pin the mechanism that makes the
claim true.
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_siri_local.audio.capture import Capture
from mini_siri_local.config import VAD_FRAME_SAMPLES, AudioConfig


class FakeStream:
    open_count = 0

    def __init__(self, **kwargs):
        self.closed = False
        FakeStream.open_count += 1

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def capture(monkeypatch):
    FakeStream.open_count = 0
    monkeypatch.setattr("sounddevice.InputStream", FakeStream)
    monkeypatch.setattr(
        "mini_siri_local.audio.capture.select_input", lambda cfg, name: (0, "Fake Microphone")
    )
    return Capture(AudioConfig())


def test_mute_closes_the_input_stream(capture):
    capture.start()
    assert capture.is_open
    capture.set_muted(True)
    assert not capture.is_open, "the device is still open; mute is only a software gate"
    capture.set_muted(False)
    assert capture.is_open
    assert FakeStream.open_count == 2


def test_mute_is_idempotent(capture):
    capture.start()
    capture.set_muted(True)
    capture.set_muted(True)
    capture.set_muted(False)
    capture.set_muted(False)
    assert capture.is_open
    assert FakeStream.open_count == 2  # not reopened once per call


def test_mute_discards_queued_audio(capture):
    """Audio heard before the mute must not surface after it."""
    capture.start()
    capture.queue.put_nowait((np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32), 0))
    capture.set_muted(True)
    assert capture.queue.empty()


def test_frames_yields_none_while_idle(capture):
    """Without an idle tick the loop never regains control, so an unmute
    request could never be seen."""
    capture.start()
    frames = capture.frames()
    assert next(frames) is None
    capture.stop()


def test_drop_oldest_never_blocks_the_callback(capture):
    """Backpressure must discard, not block: blocking stalls the audio device."""
    capture.start()
    block = np.zeros((VAD_FRAME_SAMPLES, 1), dtype=np.float32)
    for _ in range(capture.config.queue_max_frames + 5):
        capture._callback(block, VAD_FRAME_SAMPLES, None, None)
    assert capture.queue.qsize() == capture.config.queue_max_frames
    assert capture.dropped_frames == 5
