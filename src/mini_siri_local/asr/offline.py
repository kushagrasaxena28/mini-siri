"""Offline transcription via parakeet-mlx, in memory.

Uses get_logmel + generate rather than transcribe(path): measured 81 ms vs 124 ms
on the same audio, and keeps disk I/O off the critical path.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel

from ..config import AsrConfig

# One mel window. Shorter input reaches the frontend as a negative stride and
# fails inside MLX with an unrelated-looking error.
MIN_SAMPLES = 400


class OfflineAsr:
    def __init__(self, cfg: AsrConfig | None = None):
        self.cfg = cfg or AsrConfig()
        self.model = from_pretrained(self.cfg.model_id)

    def warmup(self, seconds: float = 1.0) -> float:
        """Force lazy kernel compilation off the user's first utterance.

        Cold start is ~3 s vs ~85 ms warm -- without this the first command pays it.
        """
        started = time.monotonic()
        self.transcribe(np.zeros(int(seconds * 16_000), dtype=np.float32))
        return (time.monotonic() - started) * 1000

    def transcribe(self, audio_f32: np.ndarray) -> str:
        """Mono float32 at 16 kHz -> text. Too-short or non-finite audio is
        silence, not an error: a glitching input device must not kill the turn."""
        if audio_f32.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {audio_f32.shape}")
        if audio_f32.size < MIN_SAMPLES or not np.isfinite(audio_f32).all():
            return ""
        samples = mx.array(audio_f32.astype(np.float32, copy=False))
        result = self.model.generate(get_logmel(samples, self.model.preprocessor_config))
        return result[0].text.strip() if result else ""
