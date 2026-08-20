"""ASR runs during the endpoint hangover, not after it.

The optimisation rests on one claim: the audio available at the FIRST silent
frame is identical to the audio the turn ends with. If that is ever false the
assistant acts on a transcript of the wrong audio, and nothing raises. So the
claim is asserted directly, and then the loop is driven end to end with a fake
recogniser to prove it actually takes the shortcut.

Fast: no models are loaded. That is the point of Assistant taking its
components rather than building them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_siri_local.config import VAD_FRAME_MS, VAD_FRAME_SAMPLES, VadConfig
from mini_siri_local.pipeline.assistant import Assistant
from mini_siri_local.vad.endpoint import Endpointer, State, TurnEvent

FRAME_NS = int(VAD_FRAME_MS * 1e6)
CONFIG = VadConfig(start_frames=2, hangover_ms=300, min_utterance_ms=0)


def _frame(value: float) -> np.ndarray:
    return np.full(VAD_FRAME_SAMPLES, value, dtype=np.float32)


def _probabilities(speech: int, silence: int) -> list[float]:
    return [0.9] * speech + [0.0] * silence


def test_audio_at_first_silent_frame_equals_the_final_utterance():
    """The load-bearing claim. take_audio() drops the trailing silence run, so
    the buffer contents stop changing the moment speech stops."""
    endpointer = Endpointer(CONFIG)
    guess = None
    for index, probability in enumerate(_probabilities(10, endpointer.hangover_frames + 1)):
        event = endpointer.update(_frame(index + 1), probability, index * FRAME_NS)
        if event is TurnEvent.NONE and endpointer.silence_frames == 1:
            guess = endpointer.peek_audio()
        if event is TurnEvent.TURN_END:
            final = endpointer.take_audio()
            assert guess is not None, "never speculated"
            np.testing.assert_array_equal(guess, final)
            return
    pytest.fail("turn never ended")


def test_resumed_speech_makes_the_guess_stale_and_detectable():
    """A mid-sentence pause invalidates the guess. The pipeline compares sample
    counts, so the counts must actually differ."""
    endpointer = Endpointer(CONFIG)
    guesses = []
    for index, probability in enumerate(_probabilities(6, 3) + _probabilities(6, 12)):
        event = endpointer.update(_frame(index + 1), probability, index * FRAME_NS)
        if event is TurnEvent.NONE and endpointer.silence_frames == 1:
            guesses.append(endpointer.peek_audio())
        if event is TurnEvent.TURN_END:
            final = endpointer.take_audio()
            assert len(guesses) == 2, "expected one guess per silence gap"
            assert guesses[0].size != final.size, "stale guess is indistinguishable by size"
            np.testing.assert_array_equal(guesses[-1], final)
            return
    pytest.fail("turn never ended")


def test_peek_audio_does_not_consume_the_buffer():
    endpointer = Endpointer(CONFIG)
    for index, probability in enumerate(_probabilities(6, 1)):
        endpointer.update(_frame(index + 1), probability, index * FRAME_NS)
    assert endpointer.state is State.SPEECH
    np.testing.assert_array_equal(endpointer.peek_audio(), endpointer.peek_audio())
    np.testing.assert_array_equal(endpointer.peek_audio(), endpointer.take_audio())


# --- driving the real loop with fakes ----------------------------------------


class ScriptedSource:
    """Yields frames whose VAD probability is fixed by the script."""

    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities

    def frames(self):
        for index, _ in enumerate(self.probabilities):
            yield _frame(index + 1), index * FRAME_NS

    def set_muted(self, muted: bool) -> None:
        pass


class CountingAsr:
    def __init__(self):
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls.append(audio.size)
        return "open terminal"


@pytest.fixture
def scripted_vad(monkeypatch):
    """Replace Silero with the script, so no model is loaded."""
    probabilities: list[float] = []

    class ScriptedVad:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def __call__(self, frame):
            value = probabilities[self.index] if self.index < len(probabilities) else 0.0
            self.index += 1
            return value

        def reset(self):
            pass

    monkeypatch.setattr("mini_siri_local.pipeline.assistant.SileroVad", ScriptedVad)
    return probabilities


def _run(scripted_vad, script: list[float]) -> CountingAsr:
    scripted_vad[:] = script
    asr = CountingAsr()
    assistant = Assistant(asr, vad_config=CONFIG)  # transcribe-only: no SLM, no executor
    assistant.listen(ScriptedSource(script))
    return asr


def test_clean_turn_transcribes_exactly_once_during_the_hangover(scripted_vad):
    """One transcription, and it happens before the turn is declared over."""
    asr = _run(scripted_vad, _probabilities(10, 15))
    assert len(asr.calls) == 1, f"expected one ASR call, got {len(asr.calls)}"


def test_the_guess_covers_exactly_the_final_utterance(scripted_vad):
    """One call is not enough on its own -- it has to be a call on the RIGHT
    audio. A guess taken a frame too early would still be reused, silently
    transcribing a clipped utterance."""
    script = _probabilities(10, 15)
    asr = _run(scripted_vad, script)

    endpointer = Endpointer(CONFIG)
    for index, probability in enumerate(script):
        event = endpointer.update(_frame(index + 1), probability, index * FRAME_NS)
        if event is TurnEvent.TURN_END:
            assert asr.calls == [endpointer.take_audio().size]
            return
    pytest.fail("turn never ended")


def test_mid_utterance_pause_costs_one_extra_transcription(scripted_vad):
    """A discarded guess is the honest price of the optimisation. It must cost
    ONE extra call, not one per silent frame."""
    asr = _run(scripted_vad, _probabilities(6, 3) + _probabilities(6, 15))
    assert len(asr.calls) == 2, f"expected two ASR calls, got {len(asr.calls)}"
    assert asr.calls[0] < asr.calls[1], "second transcription should cover more audio"


def test_noise_burst_never_reaches_the_recogniser(scripted_vad):
    """A blip too short to become a turn opens one anyway (start_frames is 2),
    then gets discarded. Speculating on it would spend ~95 ms of GPU on audio
    that is thrown away -- and in a noisy room that happens constantly."""
    config = VadConfig(start_frames=2, hangover_ms=100, min_utterance_ms=500)
    script = _probabilities(2, 20)
    scripted_vad[:] = script
    asr = CountingAsr()
    Assistant(asr, vad_config=config).listen(ScriptedSource(script))
    assert asr.calls == [], f"transcribed {len(asr.calls)} discarded burst(s)"
