"""Background pre-rendering must stop before the interpreter does.

`prerender_parameterised_async` runs MLX inference on a daemon thread. A daemon thread is
killed abruptly at interpreter exit -- so if it is still synthesising when the process ends,
Metal is torn down mid-operation and the process aborts with `recursive_mutex lock failed`
AFTER the assistant has finished and printed its output.

Reproduced at 2-in-6 with speech synthesis enabled; 0-in-22 after close() started
cancelling and joining the worker.
"""

from __future__ import annotations

import threading
import time

from mini_siri_local.config import TtsConfig
from mini_siri_local.tts import synthesizer as synth_module


class _SlowFakeSynthesizer(synth_module.Synthesizer):
    """Real thread machinery, fake (slow) synthesis -- no model needed."""

    def __init__(self):
        self.config = TtsConfig()
        self._cache = {}
        self._lock = threading.Lock()
        self._background_thread = None
        self._stop_prerender = threading.Event()
        self.calls = 0

    def synthesize(self, text: str):
        self.calls += 1
        time.sleep(0.02)
        return text


def test_close_stops_the_background_prerender():
    s = _SlowFakeSynthesizer()
    thread = s.prerender_parameterised_async()
    time.sleep(0.05)
    assert thread.is_alive()

    s.close(timeout_s=5.0)
    assert not thread.is_alive(), "worker still running after close(); it would be killed at exit"


def test_close_stops_early_rather_than_rendering_everything():
    s = _SlowFakeSynthesizer()
    s.prerender_parameterised_async()
    time.sleep(0.05)
    s.close(timeout_s=5.0)
    assert s.calls < len(synth_module.PARAMETERISED_PHRASES), (
        "close() waited for the whole pre-render instead of cancelling it"
    )


def test_close_is_safe_when_no_thread_was_started():
    _SlowFakeSynthesizer().close(timeout_s=1.0)
