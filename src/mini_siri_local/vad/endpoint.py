"""Turn boundary detection.

VAD answers "is this frame speech?". It does NOT answer "is the turn over?" --
that is this state machine, and its hangover is paid on every single turn.

Two details that matter:
  * t_speech_end_ns is RETROSPECTIVE (the last speech frame), not when the FSM
    concluded. Measuring from the conclusion would hide the endpointing cost.
  * A pre-roll buffer keeps audio from BEFORE speech was detected, otherwise the
    first ~64 ms of every utterance is clipped and the ASR loses the first word.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from ..config import VAD_FRAME_MS, VadConfig


class State(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


class TurnEvent(Enum):
    NONE = "none"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    DISCARDED = "discarded"  # too short to be speech


class Endpointer:
    def __init__(self, cfg: VadConfig | None = None):
        self.cfg = cfg or VadConfig()
        self.hangover_frames = max(1, round(self.cfg.hangover_ms / VAD_FRAME_MS))
        self.preroll_frames = max(1, round(self.cfg.preroll_ms / VAD_FRAME_MS))
        self.max_frames = round(self.cfg.max_utterance_ms / VAD_FRAME_MS)
        self.min_frames = round(self.cfg.min_utterance_ms / VAD_FRAME_MS)
        self.reset()

    def reset(self) -> None:
        self.state = State.SILENCE
        self._speech_run = 0
        self.silence_frames = 0  # public: the pipeline speculates on the first silent frame
        self._preroll: deque = deque(maxlen=self.preroll_frames)
        self._buffer: list[np.ndarray] = []
        self.t_speech_start_ns = 0
        self.t_speech_end_ns = 0
        self.speech_frames = 0

    def update(self, frame: np.ndarray, prob: float, t_ns: int) -> TurnEvent:
        is_speech = (
            prob >= self.cfg.speech_threshold
            if self.state is State.SILENCE
            else prob >= self.cfg.exit_threshold  # hysteresis
        )

        if self.state is State.SILENCE:
            self._preroll.append((frame, t_ns))
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.cfg.start_frames:
                    self.state = State.SPEECH
                    self._buffer = [f for f, _ in self._preroll]
                    self.t_speech_start_ns = self._preroll[0][1]
                    self.t_speech_end_ns = t_ns
                    self.speech_frames = self._speech_run
                    self.silence_frames = 0
                    self._preroll.clear()
                    return TurnEvent.TURN_START
            else:
                self._speech_run = 0
            return TurnEvent.NONE

        # -- in SPEECH --------------------------------------------------------
        self._buffer.append(frame)
        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
            self.t_speech_end_ns = t_ns  # retrospective end advances
        else:
            self.silence_frames += 1
            if self.silence_frames >= self.hangover_frames:
                return self._close()

        if len(self._buffer) >= self.max_frames:
            return self._close()
        return TurnEvent.NONE

    def _close(self) -> TurnEvent:
        too_short = self.speech_frames < self.min_frames
        self.state = State.SILENCE
        self._speech_run = 0
        return TurnEvent.DISCARDED if too_short else TurnEvent.TURN_END

    @property
    def would_close_as_turn(self) -> bool:
        """True if closing now would yield a TURN_END rather than a DISCARD.

        The pipeline checks this before transcribing during the hangover: a noise
        blip that is about to be thrown away is not worth an ASR call.
        """
        return self.speech_frames >= self.min_frames

    def peek_audio(self) -> np.ndarray:
        """What take_audio() would return right now, without consuming the buffer.

        Because the trailing silence is always dropped, the audio here at the
        FIRST silent frame is identical to what the turn ends with -- unless
        speech resumes, which only ever makes the buffer longer. That is what
        lets the pipeline transcribe during the hangover instead of after it.
        """
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        keep = max(1, len(self._buffer) - self.silence_frames)
        return np.concatenate(self._buffer[:keep])

    def take_audio(self) -> np.ndarray:
        """Utterance audio including pre-roll, minus the trailing hangover silence."""
        audio = self.peek_audio()
        self._buffer = []
        return audio
