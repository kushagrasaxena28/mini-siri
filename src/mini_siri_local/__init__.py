"""mini-siri-local -- an always-listening, fully offline voice command layer for macOS.

Pipeline:

    mic -> VAD -> endpointing -> ASR -> SLM -> validation -> executor -> TTS

Module map:

    config          every tunable and model id, in one place
    audio           capture, device selection, file replay
    vad             Silero wrapper and the turn-boundary state machine
    asr             Parakeet transcription
    slm             transcript -> JSON intent
    schema          intent definitions and the validation trust boundary
    executor        the only code that performs actions
    tts             synthesis and playback of confirmations
    pipeline        the loop that ties it together

Start with pipeline/assistant.py -- everything else is reached from there.
"""

from __future__ import annotations

__version__ = "0.1.0"
