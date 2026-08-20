"""The voice command pipeline.

    mic -> VAD -> endpointing -> ASR -> SLM -> validation -> executor -> TTS

One loop for every mode, so there is one thing to reason about. Construct with
`Assistant.build()` for the normal path; the constructor takes already-built
components so the loop can be driven in a test without loading 3 GB of models.
"""

from __future__ import annotations

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
        return (
            self.endpoint_ms
            + self.asr_ms
            + self.slm_ms
            + self.validate_ms
            + self.exec_ms
            + self.ttfa_ms
        )


class Assistant:
    """Owns the models and turns one utterance into one action."""

    def __init__(
        self,
        asr,
        slm=None,
        executor=None,
        speaker=None,
        vad_config: VadConfig | None = None,
    ):
        self.asr = asr
        self.slm = slm
        self.executor = executor
        self.speaker = speaker
        self.vad_config = vad_config or VadConfig()

        # Optional observers, set by a UI. None means headless CLI behaviour.
        self.on_status: Callable[[str], None] | None = None
        self.on_turn: Callable[[str, str], None] | None = None
        self.last_turn: TurnTiming | None = None

    @classmethod
    def build(
        cls,
        vad_config: VadConfig | None = None,
        slm_config: SlmConfig | None = None,
        tts_config: TtsConfig | None = None,
        *,
        transcribe_only: bool = False,
        enable_tts: bool = True,
    ) -> Assistant:
        """Load and warm every model, then construct.

        Models are loaded once and kept resident -- loading on demand would put
        multi-second latency on the user's first command.
        """
        print("loading models ...")
        started = time.monotonic()
        asr = OfflineAsr(AsrConfig())

        slm = executor = speaker = None
        if not transcribe_only:
            slm = SlmParser(slm_config or SlmConfig())
            executor = Executor()
            if enable_tts:
                # Deferred: importing mlx_audio pulls in the Kokoro stack and spaCy.
                from ..tts.speaker import Speaker  # noqa: PLC0415

                speaker = Speaker(tts_config or TtsConfig())

        assistant = cls(asr, slm, executor, speaker, vad_config)
        if executor is not None:
            executor.notify = assistant._announce_timer
        load_ms = (time.monotonic() - started) * 1000
        warmup_ms = assistant.warm_up()
        print(f"  load {load_ms:.0f} ms | warm-up {warmup_ms:.0f} ms")
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
        print(f"\n  [timer] {message}")
        if self.speaker is not None:
            self.speaker.speak(message)

    def handle_transcript(self, text: str, endpoint_ms: float = 0.0, asr_ms: float = 0.0) -> None:
        """Transcript -> intent -> action -> confirmation, with one timing line."""
        if not text.strip():
            print("  (empty transcript, ignored)")
            return

        if self.slm is None or self.executor is None:
            print(f"  {text!r}")
            print(f"  endpoint {endpoint_ms:5.0f} | asr {asr_ms:5.0f} ms\n")
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
            print(f"  {text!r}\n  -> REJECTED: {result.reason}  (raw={result.raw[:80]!r})")
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

        print(f"  {text!r}")
        print(f"  -> {result.intent.value} {result.args}")
        if result.intent is Intent.UNKNOWN:
            print("     (declined -- no action)")
        elif action.speech:
            print(f'     "{action.speech}"')
        print(
            f"  endpoint {endpoint_ms:5.0f} | asr {asr_ms:5.0f} | slm {parsed.latency_ms:5.0f} "
            f"| exec {exec_ms:5.1f} | ttfa {ttfa_ms:5.1f}"
            f"{' (cached)' if tts_cached else ' (synth)'} | E2E {self.last_turn.total_ms:6.0f} ms\n"
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
            print(f"\nreplaying {replay}\n")
        else:
            source = Capture(AudioConfig(), device)
            print(f"  device: [{source.device_index}] {source.device_name}")
            print("\nlistening -- speak a command (ctrl-c to stop)\n")

        try:
            with source:
                self.listen(source, is_muted)
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
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
                print(f"  [turn {turn_number}]")
            elif event is TurnEvent.DISCARDED:
                endpointer.take_audio()
                vad.reset()
                guessed_samples = 0
                self._notify_status("listening")
            elif event is TurnEvent.TURN_END:
                self._notify_status("thinking")
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
