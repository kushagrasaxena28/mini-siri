"""Audio playback through a persistent output stream.

Separate from synthesis because the constraints are opposite: the output
callback runs on a real-time thread and must never allocate, block, or raise,
while synthesis is slow and GPU-bound.

The stream is opened once and held for the process lifetime -- opening it per
utterance would add device-start latency to every confirmation.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

TTFA_WAIT_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class PlaybackTiming:
    """`time_to_first_audio_ms` is measured to the first sample reaching the
    DEVICE, not to when generation returned -- the gap between those is real
    latency the user hears. It is NaN for a clip that had to queue behind
    another, because the wait is then playback time, not pipeline latency."""

    time_to_first_audio_ms: float
    duration_s: float


class Player:
    def __init__(self, sample_rate_hz: int, blocksize: int = 512):
        self.sample_rate_hz = sample_rate_hz

        # Touched by the audio callback: keep every operation trivial.
        # deque append/popleft is atomic under the GIL, so no lock is needed.
        self._queue: deque[np.ndarray] = deque()
        self._current: np.ndarray | None = None
        self._position = 0
        self._submitted_at_ns = 0
        self._first_audio_at_ns = 0
        self._finished = threading.Event()
        self._finished.set()

        self.stream = sd.OutputStream(
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            callback=self._callback,
        )
        self.stream.start()

    # -- real-time audio thread -------------------------------------------
    def _callback(self, outdata, frames, time_info, status) -> None:
        if self._current is None:
            if not self._queue:
                outdata.fill(0)
                return
            self._current = self._queue.popleft()
            self._position = 0

        if self._first_audio_at_ns == 0:
            self._first_audio_at_ns = time.monotonic_ns()

        chunk = self._current[self._position : self._position + frames]
        if len(chunk) < frames:
            # A clip ending mid-block leaves the rest of the block silent rather
            # than splicing the next one in: one block of gap (~21 ms) between
            # queued clips, for a callback that stays trivial.
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk) :].fill(0)
            self._current = None
            self._position = 0
            if not self._queue:
                self._finished.set()
        else:
            outdata[:, 0] = chunk
            self._position += frames

    # -- control -----------------------------------------------------------
    def play(self, waveform: np.ndarray) -> PlaybackTiming:
        """Hand a waveform to the output stream.

        Clips QUEUE rather than replacing each other: a timer firing mid-sentence
        used to cut off the confirmation that was still playing.
        """
        if waveform.size == 0:
            return PlaybackTiming(0.0, 0.0)

        duration_s = len(waveform) / self.sample_rate_hz
        was_idle = self._current is None and not self._queue
        self._finished.clear()
        self._queue.append(waveform)
        if not was_idle:
            return PlaybackTiming(float("nan"), duration_s)

        # Idle before this: return as soon as the device has taken the first
        # sample, so the caller can measure real time-to-first-audio.
        self._first_audio_at_ns = 0
        self._submitted_at_ns = time.monotonic_ns()
        deadline = time.monotonic() + TTFA_WAIT_TIMEOUT_S
        while self._first_audio_at_ns == 0 and time.monotonic() < deadline:
            time.sleep(0.001)

        if not self._first_audio_at_ns:
            return PlaybackTiming(float("nan"), duration_s)
        return PlaybackTiming((self._first_audio_at_ns - self._submitted_at_ns) / 1e6, duration_s)

    @property
    def is_playing(self) -> bool:
        return not self._finished.is_set()

    def wait(self, timeout_s: float = 5.0) -> None:
        self._finished.wait(timeout=timeout_s)

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
