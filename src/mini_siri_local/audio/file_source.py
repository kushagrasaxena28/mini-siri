"""Replay a WAV through the pipeline as if it were live.

Benchmarks and tests must never depend on a live microphone -- nothing about one
is reproducible. Same frame geometry and same timestamps as Capture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import SAMPLE_RATE_HZ, VAD_FRAME_SAMPLES


class FileSource:
    """Yields (frame, t_ns) with synthetic timestamps advancing at real-time rate.

    Unlike Capture, frames() never yields None: a file has no idle intervals.
    Consumers written against Capture already handle None, so this stays a
    drop-in replacement.
    """

    def __init__(self, path: Path | str, pad_tail_s: float = 1.0):
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE_HZ:
            raise ValueError(f"{path}: expected {SAMPLE_RATE_HZ} Hz, got {sr}")
        # Trailing silence so the endpointer can actually close the turn
        if pad_tail_s > 0:
            audio = np.concatenate([audio, np.zeros(int(pad_tail_s * sr), dtype=np.float32)])
        self.audio = audio
        self.dropped_frames = 0
        self.callback_errors = 0
        self.device_index, self.device_name = -1, f"file:{Path(path).name}"

    def frames(self):
        n = VAD_FRAME_SAMPLES
        frame_ns = int(n / SAMPLE_RATE_HZ * 1e9)
        t = 0
        for i in range(0, len(self.audio) - n + 1, n):
            yield self.audio[i : i + n], t
            t += frame_ns

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # API parity with Capture -- a file has no device to open, close or mute.
    def start(self):
        pass

    def stop(self):
        pass

    def set_muted(self, muted: bool):
        pass
