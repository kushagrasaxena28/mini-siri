"""Spoken confirmations: synthesis + playback, composed.

This is the interface the pipeline uses. The two halves live apart because they
have opposite constraints -- see synthesizer.py (GPU-bound, cached) and
player.py (real-time, must never block).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import TtsConfig
from .player import Player
from .synthesizer import Synthesizer


@dataclass(frozen=True)
class SpeechTiming:
    cached: bool
    synth_ms: float
    ttfa_ms: float
    audio_s: float


class Speaker:
    def __init__(self, config: TtsConfig | None = None):
        self.config = config or TtsConfig()
        self.synthesizer = Synthesizer(self.config)
        self.player = Player(self.config.sample_rate_hz, self.config.blocksize)

        if self.config.prerender:
            self.synthesizer.prerender_fixed()
            self.synthesizer.prerender_parameterised_async()

    def speak(self, text: str) -> SpeechTiming:
        """Synthesise (or fetch from cache) and play. Non-blocking on playback."""
        if not text.strip():
            return SpeechTiming(cached=True, synth_ms=0.0, ttfa_ms=0.0, audio_s=0.0)

        was_cached = self.synthesizer.is_cached(text)
        started = time.monotonic_ns()
        waveform = self.synthesizer.synthesize(text)
        synth_ms = (time.monotonic_ns() - started) / 1e6

        timing = self.player.play(waveform)
        return SpeechTiming(
            cached=was_cached,
            synth_ms=synth_ms,
            ttfa_ms=timing.time_to_first_audio_ms,
            audio_s=timing.duration_s,
        )

    def synthesize(self, text: str):
        """Warm the cache without playing (used by warm-up)."""
        return self.synthesizer.synthesize(text)

    def wait_for_prerender(self, timeout_s: float = 180.0) -> None:
        self.synthesizer.wait_for_background_prerender(timeout_s)

    @property
    def cached_phrase_count(self) -> int:
        return self.synthesizer.cached_count

    @property
    def is_speaking(self) -> bool:
        return self.player.is_playing

    def wait(self, timeout_s: float = 5.0) -> None:
        self.player.wait(timeout_s)

    def close(self) -> None:
        # Synthesiser first: its background thread must stop before the audio device
        # goes away, and before the interpreter starts unloading MLX.
        self.synthesizer.close()
        self.player.close()
