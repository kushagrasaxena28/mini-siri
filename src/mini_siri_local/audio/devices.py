"""Input device selection by NAME.

Indices are unstable. Observed on this machine within one minute: ffmpeg listed the
built-in mic at index 2, then index 2 became invalid when headphones disconnected;
sounddevice numbers the same hardware differently again. A hardcoded index silently
records from a virtual device (digital silence, no error).
"""

from __future__ import annotations

import sounddevice as sd

from ..config import AudioConfig


class NoInputDeviceError(RuntimeError):
    pass


def list_inputs() -> list[tuple[int, dict]]:
    return [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def is_virtual(name: str, cfg: AudioConfig) -> bool:
    return any(h in name.lower() for h in cfg.virtual_hints)


def is_deprioritised(name: str, cfg: AudioConfig) -> bool:
    """Real hardware we should not pick unless asked -- a Continuity iPhone/iPad."""
    return any(h in name.lower() for h in cfg.deprioritised_hints)


def select_input(cfg: AudioConfig, override: str | None = None) -> tuple[int, str]:
    """--device -> built-in mic -> any ordinary device -> non-virtual -> anything.

    The middle two steps exist because a nearby iPhone is offered as an input by
    Continuity and is real enough to pass the virtual filter, so it used to win by
    enumeration order on any Mac whose built-in mic was not literally named
    "MacBook Pro Microphone" -- and a locked phone records silence.
    """
    inputs = list_inputs()
    if not inputs:
        raise NoInputDeviceError("no audio input devices found")

    if override:
        for i, d in inputs:
            if override.lower() in d["name"].lower():
                return i, d["name"]
        raise NoInputDeviceError(f"no input device matching {override!r}")

    for want in cfg.preferred_devices:
        for i, d in inputs:
            if want.lower() in d["name"].lower():
                return i, d["name"]
    for i, d in inputs:
        if not is_virtual(d["name"], cfg) and not is_deprioritised(d["name"], cfg):
            return i, d["name"]
    for i, d in inputs:
        if not is_virtual(d["name"], cfg):
            return i, d["name"]
    return inputs[0][0], inputs[0][1]["name"]
