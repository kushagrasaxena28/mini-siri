"""Kokoro speech synthesis with a phrase cache.

Separate from playback (player.py) because the two have different constraints:
synthesis is GPU-bound and slow, playback is real-time and must never block.

The confirmation vocabulary is CLOSED -- a handful of fixed phrases plus a few
parameterised ones. So the fastest correct implementation is not a faster model
but simply not synthesising the same phrase twice. Measured:

    cached    0.0-0.5 ms
    uncached  400-500 ms
"""

from __future__ import annotations

import threading
import time

import mlx.core as mx
import numpy as np

from ..config import TtsConfig

# Fully-fixed confirmations, rendered at startup before the pipeline reports ready.
FIXED_PHRASES = [
    "Noted.",
    "Cancelled.",
    "Nothing to cancel.",
    "Muted.",
    "Unmuted.",
    "Volume up.",
    "Volume down.",
    "Sorry, I didn't catch that.",
    "Something went wrong.",
    "Nothing is playing.",
    "You don't have any notes yet.",
    "I couldn't find anything about that.",
    "Paused.",
    "Played.",
    "Done.",
]

# Parameterised forms, rendered in the BACKGROUND so they do not delay startup
# but are usually warm by the time a user says them.
PARAMETERISED_PHRASES = [
    *(f"Timer set for {n} minutes." for n in (2, 3, 5, 10, 15, 20, 25, 30, 45)),
    "Timer set for 1 minute.",
    *(f"Timer set for {n} seconds." for n in (10, 15, 30, 45)),
    *(f"Timer set for {n} hours." for n in (2, 3)),
    "Timer set for 1 hour.",
    *(
        f"Opening {app}."
        for app in (
            "Terminal",
            "Safari",
            "Google Chrome",
            "Finder",
            "Notes",
            "Music",
            "Spotify",
            "Slack",
            "Visual Studio Code",
            "Mail",
            "Calendar",
            "Messages",
        )
    ),
    *(
        f"Opening {folder}."
        for folder in ("Downloads", "Documents", "Desktop", "Applications", "Pictures", "Movies")
    ),
]


def release_gpu_buffer_cache() -> None:
    """Return MLX's buffer cache to the OS after bulk synthesis.

    Measured: pre-rendering ~47 phrases grows mx.get_cache_memory() to 10.7 GB
    on a 16 GB machine. The next unrelated inference then pays to evict all of
    it -- observed as a 30 SECOND stall on the first ASR+SLM turn (asr 1541 ms,
    slm 30218 ms), dropping to 103 ms / 329 ms immediately after a clear.

    Called incrementally during background pre-render rather than once at the
    end, so the cache never grows large enough to cause a single big eviction.
    """
    mx.clear_cache()


class Synthesizer:
    """Text -> waveform, with a process-lifetime cache keyed on the exact string."""

    def __init__(self, config: TtsConfig | None = None):
        # Deferred: mlx_audio drags in spaCy and misaki, seconds of import time
        # that a --transcribe-only or --no-tts run should not pay.
        from mlx_audio.tts.models.kokoro import KokoroPipeline  # noqa: PLC0415
        from mlx_audio.tts.utils import load_model  # noqa: PLC0415

        self.config = config or TtsConfig()
        self.model = load_model(self.config.model_id)
        self.pipeline = KokoroPipeline(
            lang_code=self.config.lang_code,
            model=self.model,
            repo_id=self.config.model_id,
        )
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._background_thread: threading.Thread | None = None

    def is_cached(self, text: str) -> bool:
        with self._lock:
            return text in self._cache

    @property
    def cached_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 audio at config.sample_rate_hz. Cached by text.

        Releases the MLX buffer cache after every UNCACHED synthesis. This is
        not optional cleanup: measured on a live session where the assistant
        spoke phrases outside the pre-rendered set (e.g. "Opening System
        Settings." -- not in the app pre-render list), repeated uncached calls
        left the cache sitting around 1.5 GB with no release point, and ASR/SLM
        latency degraded and STAYED degraded from turn ~9 onward (438 -> 505 ->
        1547 -> 2080 -> 1891 ms). The bulk pre-render path already released
        periodically; this path -- the one that actually runs during normal
        conversation -- did not.
        """
        with self._lock:
            cached = self._cache.get(text)
        if cached is not None:
            return cached

        audio = None
        for result in self.pipeline(text, voice=self.config.voice, speed=1.0):
            audio = result.audio
            break
        if audio is None:
            return np.zeros(0, dtype=np.float32)

        mx.eval(audio)
        waveform = np.asarray(audio).reshape(-1).astype(np.float32)
        with self._lock:
            self._cache[text] = waveform
        release_gpu_buffer_cache()
        return waveform

    def prerender_fixed(self) -> float:
        """Render the fixed phrases. Blocking -- call before reporting ready."""
        started = time.monotonic()
        for phrase in FIXED_PHRASES:
            self.synthesize(phrase)
        release_gpu_buffer_cache()
        return (time.monotonic() - started) * 1000

    def prerender_parameterised_async(self) -> threading.Thread:
        """Warm the parameterised phrases without delaying startup."""

        def worker() -> None:
            for index, phrase in enumerate(PARAMETERISED_PHRASES, start=1):
                try:
                    self.synthesize(phrase)
                except Exception:
                    continue  # a cold cache entry is not worth crashing over
                if index % self.config.cache_release_interval == 0:
                    release_gpu_buffer_cache()
            release_gpu_buffer_cache()

        thread = threading.Thread(target=worker, daemon=True, name="tts-prerender")
        thread.start()
        self._background_thread = thread
        return thread

    def wait_for_background_prerender(self, timeout_s: float = 180.0) -> None:
        """Block until background pre-rendering finishes. Used by benchmarks so
        they measure steady state rather than contention."""
        if self._background_thread is not None:
            self._background_thread.join(timeout=timeout_s)
