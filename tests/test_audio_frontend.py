"""The audio front end: frame geometry, VAD, and turn boundaries.

Everything here corresponds to a bug that was actually hit, or to a structural
constant that would corrupt the pipeline without raising. Nothing in this file
tests something that would fail loudly on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from mini_siri_local.config import (
    ENCODER_FRAME_MS,
    SAMPLE_RATE_HZ,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    SileroConfig,
    VadConfig,
)
from mini_siri_local.vad.endpoint import Endpointer, State, TurnEvent
from mini_siri_local.vad.silero import CONTEXT_SAMPLES, SileroVad

# --- frame geometry ----------------------------------------------------------


def test_vad_frame_is_512_samples_32ms():
    """Silero's input size is fixed. The spec said 20 ms; it is 32 ms."""
    assert VAD_FRAME_SAMPLES == 512
    assert pytest.approx(32.0) == VAD_FRAME_MS


def test_encoder_frame_is_80ms():
    assert ENCODER_FRAME_MS == 80


def test_frame_sizes_do_not_divide_evenly():
    """80 / 32 = 2.5. Repacking must be handled in one place, not per-consumer."""
    assert pytest.approx(2.5) == ENCODER_FRAME_MS / VAD_FRAME_MS


# --- the VAD context bug -----------------------------------------------------


# These four construct a real SileroVad, so they need the ONNX weights that
# setup.sh downloads. Skip rather than fail on a fresh clone -- the rest of this
# module is pure arithmetic and must keep running without any model present.
requires_silero = pytest.mark.skipif(
    not SileroConfig().model_path.exists(),
    reason="models/silero_vad.onnx missing -- run scripts/download_models.py",
)


def _tone(n: int, hz: float = 200.0) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE_HZ
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@requires_silero
def test_vad_rejects_wrong_frame_size():
    vad = SileroVad()
    with pytest.raises(ValueError):
        vad(np.zeros(256, dtype=np.float32))


@requires_silero
def test_vad_maintains_64_sample_context():
    """The model input is 64 context + 512 chunk.

    Feeding a bare 512 does not error -- it silently returns ~0 on loud speech.
    Measured: max prob 0.524 without context vs 1.000 with it.
    """
    vad = SileroVad()
    assert CONTEXT_SAMPLES == 64
    assert vad.context.shape == (64,)
    frame = _tone(VAD_FRAME_SAMPLES)
    vad(frame)
    # context must now hold the tail of the frame just consumed
    np.testing.assert_array_equal(vad.context, frame[-64:])


@requires_silero
def test_vad_reset_clears_both_state_and_context():
    """A leaked context or LSTM state biases the next turn."""
    vad = SileroVad()
    vad(_tone(VAD_FRAME_SAMPLES))
    assert vad.context.any()
    vad.reset()
    assert not vad.context.any()
    assert not vad.state.any()


@requires_silero
def test_vad_detects_real_speech():
    """End-to-end guard: if the context handling regresses, this drops to ~0."""
    from pathlib import Path

    import soundfile as sf

    clip = Path("benchmarks/datasets/synth/timer.wav")
    if not clip.exists():
        pytest.skip("synthetic test set not generated")
    audio, _ = sf.read(str(clip), dtype="float32")
    vad = SileroVad()
    probs = [
        vad(audio[i : i + VAD_FRAME_SAMPLES])
        for i in range(0, len(audio) - VAD_FRAME_SAMPLES + 1, VAD_FRAME_SAMPLES)
    ]
    assert max(probs) > 0.9, f"VAD failed on real speech (max {max(probs):.3f})"


# --- endpointer --------------------------------------------------------------


def _drive(ep: Endpointer, probs: list[float]) -> list[TurnEvent]:
    frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
    ns = int(VAD_FRAME_MS * 1e6)
    return [ep.update(frame, p, i * ns) for i, p in enumerate(probs)]


def test_turn_opens_after_start_frames():
    ep = Endpointer(VadConfig(start_frames=2))
    events = _drive(ep, [0.9, 0.9])
    assert events[-1] is TurnEvent.TURN_START
    assert ep.state is State.SPEECH


def test_turn_closes_after_hangover():
    cfg = VadConfig(start_frames=2, hangover_ms=100, min_utterance_ms=0)
    ep = Endpointer(cfg)
    events = _drive(ep, [0.9] * 6 + [0.0] * (ep.hangover_frames + 1))
    assert TurnEvent.TURN_END in events


def test_speech_end_is_retrospective():
    """t_speech_end must be the LAST SPEECH frame, not when the FSM concluded.

    Measuring from the conclusion would delete the endpointing cost from every
    latency number in the project.
    """
    cfg = VadConfig(start_frames=2, hangover_ms=100, min_utterance_ms=0)
    ep = Endpointer(cfg)
    n_speech = 6
    _drive(ep, [0.9] * n_speech + [0.0] * (ep.hangover_frames + 1))
    expected_ns = (n_speech - 1) * int(VAD_FRAME_MS * 1e6)
    assert ep.t_speech_end_ns == expected_ns


def test_short_burst_is_discarded():
    cfg = VadConfig(start_frames=2, hangover_ms=64, min_utterance_ms=500)
    ep = Endpointer(cfg)
    events = _drive(ep, [0.9, 0.9] + [0.0] * (ep.hangover_frames + 1))
    assert TurnEvent.DISCARDED in events


def test_preroll_is_included_in_utterance():
    """Without pre-roll the first ~64 ms of every utterance is clipped."""
    cfg = VadConfig(start_frames=2, preroll_ms=320, hangover_ms=100, min_utterance_ms=0)
    ep = Endpointer(cfg)
    _drive(ep, [0.0] * 20 + [0.9] * 6 + [0.0] * (ep.hangover_frames + 1))
    audio = ep.take_audio()
    # pre-roll frames + speech frames, minus trailing hangover
    assert len(audio) > 6 * VAD_FRAME_SAMPLES


def test_hysteresis_keeps_turn_open():
    """Mid-utterance dips below the entry threshold must not close the turn."""
    cfg = VadConfig(
        start_frames=2,
        speech_threshold=0.5,
        exit_threshold=0.35,
        hangover_ms=320,
        min_utterance_ms=0,
    )
    ep = Endpointer(cfg)
    events = _drive(ep, [0.9, 0.9] + [0.4] * 5 + [0.9] * 3)
    assert TurnEvent.TURN_END not in events
    assert ep.state is State.SPEECH
