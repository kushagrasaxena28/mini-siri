"""Guided recording session for the real-ASR-error dataset slice.

This is the one part of the dataset that cannot be synthesized. Measurement showed
`say`-generated speech corrupts differently from real speech (synthetic-voice
artifacts like "pause the music" -> "Paused music" and outright nonsense on
unclear words) -- training on that would teach the model to handle the wrong
noise distribution.

For each prompt, this script:
  1. shows you a natural-language variant of a command to say (not read woodenly)
  2. records via the mic, runs it through the REAL pipeline (VAD -> endpoint -> ASR)
  3. shows you what Parakeet actually heard
  4. you confirm the intended intent/args (usually just press enter -- it's inferred
     from the prompt), or mark it "skip" if the take was bad, or "unknown" if it
     should train the refusal path
  5. appends {transcript, intent, args, audio_ref} to datasets/intent/asr_captured.jsonl

Takes about 20-30 minutes for full coverage (~80-100 utterances, 10 intents).
You can stop anytime with ctrl-c -- everything recorded so far is saved.

    uv run python scripts/record_dataset.py            # from Terminal.app, not an IDE shell
    uv run python scripts/record_dataset.py --resume    # skip prompts already recorded
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mini_siri_local.asr.offline import OfflineAsr
from mini_siri_local.audio.capture import Capture
from mini_siri_local.config import AsrConfig, AudioConfig, VadConfig
from mini_siri_local.vad.endpoint import Endpointer, TurnEvent
from mini_siri_local.vad.silero import SileroVad

OUT_PATH = Path("datasets/intent/asr_captured.jsonl")
AUDIO_DIR = Path("benchmarks/datasets/voice")


@dataclass
class Prompt:
    say_this: str  # shown to the user -- a natural phrasing to speak
    intent: str
    args: dict


# Deliberately phrased as natural speech to say aloud, not the training text
# verbatim -- reading a script produces unnaturally clean, evenly-paced audio,
# which defeats the purpose of this slice. Vary wording as you actually would.
PROMPTS: list[Prompt] = [
    # set_timer -- include some genuinely mid-sentence pauses/fillers
    Prompt("Ask it to set a timer for ten minutes.", "set_timer", {"duration_seconds": 600}),
    Prompt("Ask for a five minute timer, casually.", "set_timer", {"duration_seconds": 300}),
    Prompt("Ask for a 30 second timer.", "set_timer", {"duration_seconds": 30}),
    Prompt(
        "Say 'set a timer for... uh... two minutes' with a real pause.",
        "set_timer",
        {"duration_seconds": 120},
    ),
    Prompt("Ask for a one hour timer.", "set_timer", {"duration_seconds": 3600}),
    Prompt("Say 'give me a fifteen minute timer'.", "set_timer", {"duration_seconds": 900}),
    # open_app
    Prompt("Ask it to open Terminal.", "open_app", {"app_name": "Terminal"}),
    Prompt("Ask it to open Safari.", "open_app", {"app_name": "Safari"}),
    Prompt("Ask it to launch Spotify.", "open_app", {"app_name": "Spotify"}),
    Prompt("Ask it to open Notes.", "open_app", {"app_name": "Notes"}),
    Prompt("Say 'can you open Slack for me'.", "open_app", {"app_name": "Slack"}),
    # set_volume
    Prompt("Ask it to turn the volume down.", "set_volume", {"direction": "down"}),
    Prompt("Ask it to turn the volume up.", "set_volume", {"direction": "up"}),
    Prompt("Say 'mute'.", "set_volume", {"direction": "mute"}),
    Prompt("Say 'set the volume to 40'.", "set_volume", {"direction": "set", "level": 40}),
    # media_control
    Prompt("Ask it to pause the music.", "media_control", {"action": "pause"}),
    Prompt("Ask it to play the music.", "media_control", {"action": "play"}),
    Prompt("Say 'skip this song'.", "media_control", {"action": "next"}),
    # capture_note -- say something a bit rambling, like a real note
    Prompt(
        "Say 'note, check the cache config tomorrow'.",
        "capture_note",
        {"text": "check the cache config tomorrow"},
    ),
    Prompt("Say 'note to self, buy milk'.", "capture_note", {"text": "buy milk"}),
    Prompt(
        "Say 'make a note that the meeting moved to 3pm'.",
        "capture_note",
        {"text": "the meeting moved to 3pm"},
    ),
    # search_notes
    Prompt("Ask 'what did I note about the encoder'.", "search_notes", {"query": "encoder"}),
    Prompt("Ask 'search my notes for cache config'.", "search_notes", {"query": "cache config"}),
    # open_path
    Prompt("Ask it to open your downloads folder.", "open_path", {"path": "downloads"}),
    Prompt("Ask it to open the desktop.", "open_path", {"path": "desktop"}),
    # get_time
    Prompt("Ask what time it is.", "get_time", {}),
    Prompt("Say 'do you know what time it is' casually.", "get_time", {}),
    # cancel
    Prompt("Say 'never mind'.", "cancel", {}),
    Prompt("Say 'cancel that'.", "cancel", {}),
    # unknown -- background-speech-like, said naturally
    Prompt(
        "Say something like you'd say to a person, e.g. 'did you see the game last night'.",
        "unknown",
        {},
    ),
    Prompt("Trail off mid-sentence: 'so anyway I was thinking we should...'", "unknown", {}),
    Prompt("Say 'what a nice day' as if talking to someone in the room.", "unknown", {}),
    Prompt("Mumble something unclear, like you're distracted.", "unknown", {}),
]


def load_existing() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text().splitlines():
        try:
            done.add(json.loads(line)["say_this"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def record_one(
    capture: Capture, vad: SileroVad, ep: Endpointer, asr: OfflineAsr, timeout_s: float = 12.0
) -> tuple[str, list]:
    """Record one utterance via the real pipeline. Returns (transcript, raw_audio_frames)."""
    vad.reset()
    ep.reset()
    frames = []
    deadline = time.monotonic() + timeout_s
    for item in capture.frames():
        if time.monotonic() > deadline:
            return "", None
        if item is None:  # idle tick, no audio this interval
            continue
        frame, t_ns = item

        frames.append(frame)
        event = ep.update(frame, vad(frame), t_ns)
        if event is TurnEvent.TURN_END:
            audio = ep.take_audio()
            return asr.transcribe(audio), audio
        if event is TurnEvent.DISCARDED:
            ep.take_audio()
            vad.reset()
    return "", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="skip prompts already recorded")
    ap.add_argument("--device", help="input device name substring")
    a = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = load_existing() if a.resume else set()
    todo = [p for p in PROMPTS if p.say_this not in done]
    if not todo:
        print("Nothing left to record.")
        return 0

    print("loading ASR + VAD ...")
    asr = OfflineAsr(AsrConfig())
    asr.warmup()
    vad = SileroVad()
    ep = Endpointer(VadConfig())

    print(f"\n{len(todo)} prompts to go (of {len(PROMPTS)} total).")
    print("Speak naturally -- don't read the instruction verbatim, say it your own way.")
    print("Press ctrl-c anytime to stop; everything so far is saved.\n")

    import soundfile as sf

    with Capture(AudioConfig(), a.device) as capture:
        for i, prompt in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {prompt.say_this}")
            input("  press enter, then speak >")
            text, audio = record_one(capture, vad, ep, asr)

            if not text.strip() or audio is None:
                print("  (nothing captured -- skipping)\n")
                continue

            print(f"  heard: {text!r}")
            resp = input(
                f"  keep as intent={prompt.intent!r} args={prompt.args}? "
                f"[Y]es / [s]kip / [u]nknown / type a correction: "
            ).strip()

            if resp.lower() in ("s", "skip"):
                print("  skipped\n")
                continue

            intent, args = prompt.intent, prompt.args
            if resp.lower() in ("u", "unknown"):
                intent, args = "unknown", {}
            elif resp and resp.lower() not in ("y", "yes", ""):
                print(
                    f"  (free-form correction not auto-parsed -- edit {OUT_PATH} by hand: {resp!r})"
                )

            audio_path = AUDIO_DIR / f"{int(time.time() * 1000)}.wav"
            sf.write(str(audio_path), audio, 16_000)

            with OUT_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "say_this": prompt.say_this,
                            "transcript": text,
                            "intent": intent,
                            "args": args,
                            "audio_file": str(audio_path),
                            "provenance": "asr_captured",
                        }
                    )
                    + "\n"
                )
            print("  saved\n")

    n = sum(1 for _ in OUT_PATH.open()) if OUT_PATH.exists() else 0
    print(f"\ndone -- {n} real utterances captured in {OUT_PATH}")
    print("Next: uv run python scripts/build_dataset.py --include-captured")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped -- progress saved")
        sys.exit(0)
