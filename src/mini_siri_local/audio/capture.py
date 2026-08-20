"""Microphone capture into a bounded frame queue.

The CoreAudio callback does the minimum: copy the block, stamp it, enqueue.
No logging, no locks, no model calls -- anything slow there causes dropouts.
Backpressure is drop-oldest with a counter: blocking would propagate into the
callback, which is the one place that must never stall.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from ..telemetry.timing import now_ns
from .devices import select_input

# How long frames() waits before yielding None. The caller needs to be given
# control periodically even in silence, or mute could never be released.
IDLE_POLL_S = 0.25


class Capture:
    """Microphone -> bounded queue of (frame, capture_timestamp_ns).

    Frames are exactly VAD_FRAME_SAMPLES long so the VAD can consume them
    directly without repacking.
    """

    def __init__(self, config: AudioConfig | None = None, device_name: str | None = None):
        self.config = config or AudioConfig()
        self.device_index, self.device_name = select_input(self.config, device_name)
        self.queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(self.config.queue_max_frames)
        self.dropped_frames = 0
        self.callback_errors = 0
        self._stream: sd.InputStream | None = None
        self._stop = threading.Event()

    # -- CoreAudio real-time thread ---------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        """Copy the block and enqueue. Nothing else belongs here.

        No allocation beyond the required copy, no logging, no locks, no raising:
        this runs on a real-time thread and anything slow causes dropouts.
        """
        if status:
            self.callback_errors += 1
        # indata is a buffer CoreAudio reuses -- it must be copied, not referenced.
        item = (indata[:, 0].copy(), now_ns())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            # Drop the OLDEST frame, never block. Blocking here would stall the
            # audio device; stale audio is less useful than current audio anyway.
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
                self.dropped_frames += 1
            except queue.Empty:
                pass

    # -- control ----------------------------------------------------------
    def _open_stream(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.config.sample_rate_hz,  # CoreAudio resamples from 48 kHz
            blocksize=self.config.frame_samples,  # exactly one VAD frame
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def start(self) -> None:
        self._stop.clear()  # allow restart after stop()
        self._open_stream()

    def stop(self) -> None:
        self._stop.set()
        self._close_stream()

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def set_muted(self, muted: bool) -> None:
        """Close or reopen the input device. Idempotent.

        Mute closes the CoreAudio stream rather than discarding frames further
        down, so the microphone is genuinely not being read -- macOS stops
        showing the recording indicator, which is the whole point of the
        control. Queued frames are dropped so nothing heard before the mute can
        surface after it.
        """
        if muted == (self._stream is None):
            return
        if muted:
            self._close_stream()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        else:
            self._open_stream()

    def frames(self):
        """Yield (frame, capture_timestamp_ns), or None when no audio arrived.

        None is yielded on idle so the caller keeps getting control while the
        device is closed; otherwise mute would be a one-way door.
        """
        while not self._stop.is_set():
            try:
                yield self.queue.get(timeout=IDLE_POLL_S)
            except queue.Empty:
                yield None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
