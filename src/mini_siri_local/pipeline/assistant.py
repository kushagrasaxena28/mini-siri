"""The voice command pipeline.

    mic -> VAD -> endpointing -> ASR -> SLM -> validation -> executor -> TTS

One loop for every mode, so there is one thing to reason about. Construct with
`Assistant.build()` for the normal path; the constructor takes already-built
components so the loop can be driven in a test without loading 3 GB of models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..asr.offline import OfflineAsr
from ..audio.capture import Capture
from ..config import AsrConfig, AudioConfig, SlmConfig, TtsConfig, VadConfig
from ..executor.handlers import Executor
from ..schema.intents import Intent, IntentCall, validate
from ..slm.parser import SlmParser
from ..ui import Console
from ..vad.endpoint import Endpointer, TurnEvent
from ..vad.silero import SileroVad

WARMUP_ROUNDS = 3


@dataclass(frozen=True)
class TurnTiming:
    """Per-stage cost of one turn. The printed timing line and
    scripts/bench_pipeline.py both read this, so they cannot disagree."""

    transcript: str
    intent: str
    endpoint_ms: float
    asr_ms: float
    slm_ms: float
    validate_ms: float
    exec_ms: float
    ttfa_ms: float

    @property
    def total_ms(self) -> float:
        """Sum of the measured stages.

        ttfa_ms is NaN when time-to-first-audio could not be measured -- the player was
        already busy, so there is no meaningful "time until sound started" for this call.
        NaN is the honest value there, but it must not swallow the whole turn, so
        unmeasured stages are skipped rather than counted as zero.
        """
        stages = (
            self.endpoint_ms,
            self.asr_ms,
            self.slm_ms,
            self.validate_ms,
            self.exec_ms,
            self.ttfa_ms,
        )
        return sum(v for v in stages if math.isfinite(v))


class Assistant:
    """Owns the models and turns one utterance into one action."""

    # Components are injected so the loop can be driven in a test without loading
    # 3 GB of models; that is worth more than a lower argument count.
    def __init__(  # noqa: PLR0913
        self,
        asr,
        slm=None,
        executor=None,
        speaker=None,
        vad_config: VadConfig | None = None,
        *,
        console: Console | None = None,
    ):
        self.asr = asr
        self.slm = slm
        self.executor = executor
        self.speaker = speaker
        self.vad_config = vad_config or VadConfig()
        self.console = console or Console()

        # Optional observers, set by a UI. None means headless CLI behaviour.
        self.on_status: Callable[[str], None] | None = None
        self.on_turn: Callable[[str, str], None] | None = None
        self.last_turn: TurnTiming | None = None
        self._turn_number = 0
        self._loads_ms: dict[str, float] = {}
        self._warmup_ms: float = 0.0

    @classmethod
    def build(  # noqa: PLR0913 -- each argument is a separately swappable component
        cls,
        vad_config: VadConfig | None = None,
        slm_config: SlmConfig | None = None,
        tts_config: TtsConfig | None = None,
        *,
        transcribe_only: bool = False,
        enable_tts: bool = True,
        console: Console | None = None,
    ) -> Assistant:
        """Load and warm every model, then construct.

        Models are loaded once and kept resident -- loading on demand would put
        multi-second latency on the user's first command.
        """
        console = console or Console()
        loads: dict[str, float] = {}

        def timed(label: str, build):
            """Load one model, reporting progress -- 13 s of silence is not a status."""
            console.loading(label)
            started = time.monotonic()
            built = build()
            loads[label] = (time.monotonic() - started) * 1000
            return built

        asr = timed("asr", lambda: OfflineAsr(AsrConfig()))

        slm = executor = speaker = None
        if not transcribe_only:
            slm = timed("slm", lambda: SlmParser(slm_config or SlmConfig()))
            executor = Executor()
            if enable_tts:
                # Deferred: importing mlx_audio pulls in the Kokoro stack and spaCy.
                try:
                    from ..tts.speaker import Speaker  # noqa: PLC0415

                    speaker = timed("tts", lambda: Speaker(tts_config or TtsConfig()))
                except ImportError as exc:
                    # `./setup.sh --lite` installs without the tts extra. Degrade to a
                    # silent assistant rather than refusing to start; every confirmation
                    # is printed anyway.
                    console.tts_unavailable(exc.name or "mlx_audio")

        assistant = cls(asr, slm, executor, speaker, vad_config, console=console)
        if executor is not None:
            executor.notify = assistant._announce_timer
        warmup_ms = assistant.warm_up()
        assistant._loads_ms = loads
        assistant._warmup_ms = warmup_ms
        return assistant

    def warm_up(self) -> float:
        """Warm the INTERLEAVED path, after every model is resident.

        Warming each model alone is not enough: the first turns that mix
        ASR + SLM + TTS on the GPU cost ~900 ms and ~1000 ms before settling to
        ~95 ms and ~250 ms. Paying that here keeps it off the first command.
        """
        started = time.monotonic()
        silence = np.zeros(16_000, dtype=np.float32)
        for _ in range(WARMUP_ROUNDS):
            self.asr.transcribe(silence)
            if self.slm is not None:
                self.slm.parse("what time is it")
            if self.speaker is not None:
                self.speaker.synthesize("Noted.")
        return (time.monotonic() - started) * 1000

    def _announce_timer(self, message: str) -> None:
        self.console.timer(message)
        if self.speaker is not None:
            self.speaker.speak(message)

    def handle_transcript(self, text: str, endpoint_ms: float = 0.0, asr_ms: float = 0.0) -> None:
        """Transcript -> intent -> action -> confirmation, with one timing line."""
        if not text.strip():
            self.console.empty()
            return

        if self.slm is None or self.executor is None:
            self._turn_number += 1
            self.console.transcript_only(self._turn_number, text, endpoint_ms, asr_ms)
            return

        parsed = self.slm.parse(text)
        started = time.monotonic()
        result = validate(parsed.raw)
        validate_ms = (time.monotonic() - started) * 1000

        if not isinstance(result, IntentCall):
            # Fail closed: no action, and say so. `result.reason` is for the log
            # line only -- it can name intents and internal states the user
            # never said. `result.speech` is what gets spoken: a generic decline
            # unless the checker set something more specific AND safe to say.
            if self.on_turn is not None:
                self.on_turn(text, "declined")
            self.slm.discard_last_turn()
            self._turn_number += 1
            self.console.rejected(self._turn_number, text, result.reason, result.raw)
            if self.speaker is not None:
                self.speaker.speak(result.speech)
            return

        started = time.monotonic()
        action = self.executor.execute(result)
        exec_ms = (time.monotonic() - started) * 1000

        ttfa_ms, tts_cached = 0.0, True
        if self.speaker is not None and action.speech:
            timing = self.speaker.speak(action.speech)
            ttfa_ms, tts_cached = timing.ttfa_ms, timing.cached

        if self.on_turn is not None:
            self.on_turn(text, result.intent.value)

        self.last_turn = TurnTiming(
            transcript=text,
            intent=result.intent.value,
            endpoint_ms=endpoint_ms,
            asr_ms=asr_ms,
            slm_ms=parsed.latency_ms,
            validate_ms=validate_ms,
            exec_ms=exec_ms,
            ttfa_ms=ttfa_ms,
        )

        if result.intent is Intent.UNKNOWN:
            # Overheard speech, correctly ignored. Keeping it in the conversation
            # would make the next real command answer to background noise.
            self.slm.discard_last_turn()

        self._turn_number += 1
        if result.intent is Intent.UNKNOWN:
            self.console.declined(self._turn_number, self.last_turn)
        else:
            self.console.action(
                self._turn_number, self.last_turn, result.args, action.speech, tts_cached
            )

    def run(
        self,
        device: str | None = None,
        replay: str | None = None,
        is_muted: Callable[[], bool] | None = None,
    ) -> int:
        """Listen until interrupted.

        `is_muted` is polled once per loop tick. While it returns True the input
        device is CLOSED, not merely ignored -- a flag that kept the microphone
        open while discarding frames would leave macOS showing the recording
        indicator, which is exactly the thing the control exists to disprove.
        """
        if replay:
            from ..audio.file_source import FileSource  # noqa: PLC0415 -- replay-only dependency

            source = FileSource(replay)
            self.console.ready(self._loads_ms, self._warmup_ms, device=None)
            self.console.replaying(replay)
        else:
            source = Capture(AudioConfig(), device)
            self.console.ready(self._loads_ms, self._warmup_ms, source.device_name)
            self.console.listening()

        try:
            with source:
                self.listen(source, is_muted)
        except KeyboardInterrupt:
            pass
        finally:
            self.console.summary()
            self.close()
        return 0

    def listen(self, source, is_muted: Callable[[], bool] | None = None) -> int:
        """Drive one audio source to exhaustion. Returns the number of turns.

        Split out from run() so a test can hand it a scripted source without
        touching a device or a context manager.
        """
        vad = SileroVad()
        endpointer = Endpointer(self.vad_config)
        turn_number = 0
        # Transcript computed early, during the hangover -- see the last branch.
        # Zero samples means "no guess"; a real turn always has some.
        guessed_samples, guessed_text = 0, ""

        for item in source.frames():
            if is_muted is not None and is_muted():
                # Reset turn state too, so a half-heard utterance from before the
                # mute cannot leak into the next one.
                source.set_muted(True)
                endpointer.reset()
                vad.reset()
                guessed_samples = 0
                continue
            source.set_muted(False)

            if item is None:  # idle tick, no audio this interval
                continue
            frame, timestamp_ns = item

            # Barge-in guard: our own confirmation would retrigger the VAD.
            # Crude but effective; proper echo handling is deferred.
            if self.speaker is not None and self.speaker.is_speaking:
                continue

            event = endpointer.update(frame, vad(frame), timestamp_ns)

            if event is TurnEvent.TURN_START:
                turn_number += 1
                self._notify_status("speech")
                # Transient: overwritten by the result, so ctrl-c never leaves a
                # turn header with nothing under it.
                self.console.hearing()
            elif event is TurnEvent.DISCARDED:
                endpointer.take_audio()
                vad.reset()
                guessed_samples = 0
                self._notify_status("listening")
            elif event is TurnEvent.TURN_END:
                self._notify_status("thinking")
                self.console.thinking()
                endpoint_ms = (timestamp_ns - endpointer.t_speech_end_ns) / 1e6
                utterance = endpointer.take_audio()

                started = time.monotonic()
                if guessed_samples == utterance.size:
                    text = guessed_text  # already transcribed during the hangover
                else:
                    text = self.asr.transcribe(utterance)
                asr_ms = (time.monotonic() - started) * 1000
                guessed_samples = 0
                vad.reset()

                self.handle_transcript(text, endpoint_ms, asr_ms)
                self._notify_status("listening")
            elif endpointer.silence_frames == 1 and endpointer.would_close_as_turn:
                # First silent frame: peek_audio() already equals what the turn
                # will end with, so transcribe DURING the 288 ms hangover instead
                # of after it. Costs ~95 ms here, which the capture queue's ~2 s
                # of headroom absorbs, and removes ASR from the critical path.
                # Speech resuming grows the buffer, which the size check catches;
                # a burst too short to become a turn is skipped entirely.
                pending = endpointer.peek_audio()
                guessed_samples, guessed_text = pending.size, self.asr.transcribe(pending)

        return turn_number

    def _notify_status(self, status: str) -> None:
        """Publish pipeline state to an observer (the menu bar), if attached."""
        if self.on_status is not None:
            self.on_status(status)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown()
        if self.speaker is not None:
            self.speaker.close()
