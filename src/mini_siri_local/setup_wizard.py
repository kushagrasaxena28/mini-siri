"""Interactive first-run check.

`--check` answers "is the environment correct?". This answers the question a new user
actually has: "does it work, and have I granted the permissions it needs?"

The permission prompts are the reason this exists. macOS asks for microphone access and
for Automation access to Notes and Reminders on FIRST USE -- which, without this, is
halfway through someone's first real command. Triggering them deliberately, while the
user is watching and expecting it, turns three confusing interruptions into one guided step.
"""

from __future__ import annotations

import sys
import time

from .config import SAMPLE_RATE_HZ, AudioConfig, SlmConfig, TtsConfig, VadConfig

METER_SECONDS = 5.0
METER_WIDTH = 40
# Below this the mic is almost certainly muted, denied, or pointed at a silent
# virtual device -- measured: a denied mic returns exact digital silence.
QUIET_RMS = 1e-4


def _bar(level: float) -> str:
    filled = min(METER_WIDTH, int(level * METER_WIDTH))
    return "#" * filled + "-" * (METER_WIDTH - filled)


def _mic_meter() -> bool:
    """Live input meter. Returns True if any real signal was seen."""
    import numpy as np  # noqa: PLC0415
    import sounddevice as sd  # noqa: PLC0415

    from .audio.devices import select_input  # noqa: PLC0415

    index, name = select_input(AudioConfig())
    print(f"  listening on [{index}] {name}")
    print(f"  say something for {METER_SECONDS:.0f} seconds\n")

    block_frames = int(0.1 * SAMPLE_RATE_HZ)
    peak = 0.0
    with sd.InputStream(
        device=index, channels=1, samplerate=SAMPLE_RATE_HZ, dtype="float32"
    ) as stream:
        deadline = time.monotonic() + METER_SECONDS
        while time.monotonic() < deadline:
            block, _ = stream.read(block_frames)
            rms = float(np.sqrt(np.mean(block**2)))
            peak = max(peak, rms)
            print(f"\r  [{_bar(min(1.0, rms * 12))}] {rms:.4f}", end="", flush=True)
    print()
    return peak > QUIET_RMS


def run_wizard() -> int:
    from .diagnostics import run_diagnostics  # noqa: PLC0415

    print("mini-siri-local -- first run\n")
    print("Four steps: environment, microphone, a spoken reply, and permissions.\n")

    print("=" * 60)
    print("1/4  Environment")
    print("=" * 60)
    if run_diagnostics(test_microphone=False) != 0:
        print("\nFix the failures above, then re-run:  uv run mini-siri-local --setup")
        return 1

    print("\n" + "=" * 60)
    print("2/4  Microphone")
    print("=" * 60)
    try:
        heard = _mic_meter()
    except Exception as exc:  # any audio failure is a setup failure here
        print(f"\n  could not open the microphone: {exc}")
        heard = False
    if not heard:
        print("\n  No signal detected. Usually one of:")
        print("    - macOS denied microphone access WITHOUT showing a prompt.")
        print("      Run this from Terminal.app or iTerm, or grant access under")
        print("      System Settings > Privacy & Security > Microphone.")
        print("    - the wrong input device was chosen; list them with --check")
        return 1
    print("  microphone works")

    print("\n" + "=" * 60)
    print("3/4  Speaking back")
    print("=" * 60)
    from .pipeline.assistant import Assistant  # noqa: PLC0415

    assistant = Assistant.build(VadConfig(), SlmConfig(), TtsConfig())
    assistant.handle_transcript("what time is it")
    if assistant.speaker is not None:
        assistant.speaker.wait()

    print("\n" + "=" * 60)
    print("4/4  Permissions for Notes and Reminders")
    print("=" * 60)
    print("  macOS asks for Automation access on first use. Triggering it now so it")
    print("  does not interrupt a real command. Approve both dialogs if they appear.\n")

    from .executor import reminders  # noqa: PLC0415
    from .executor.notes import NoteStore  # noqa: PLC0415

    store = NoteStore()
    notes = "granted" if store.backend_name == "apple-notes" else "NOT granted -- local fallback"
    rem = "granted" if reminders.is_available() else "NOT granted -- reminders will fail"
    print(f"  Notes     : {notes}")
    print(f"  Reminders : {rem}")

    assistant.close()
    print("\n" + "=" * 60)
    print("Ready. Start it with:\n")
    print("    uv run mini-siri-local                # terminal")
    print("    uv run mini-siri-local --background   # menu bar")
    return 0


if __name__ == "__main__":
    sys.exit(run_wizard())
