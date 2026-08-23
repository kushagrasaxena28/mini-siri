"""Input device selection.

A nearby iPhone is offered as an input by macOS Continuity. It is real hardware, so it
passes the virtual-device filter -- and a locked phone in another room records silence.
Reported from a MacBook Air, where the built-in mic matched no preferred name and lost to
the iPhone on enumeration order.

No hardware needed: select_input() takes the device list from list_inputs(), so a fake
one is enough.
"""

from __future__ import annotations

import pytest

from mini_siri_local.audio import devices
from mini_siri_local.audio.devices import NoInputDeviceError, select_input
from mini_siri_local.config import AudioConfig

CFG = AudioConfig()


def fake_inputs(monkeypatch, *names: str) -> None:
    monkeypatch.setattr(
        devices, "list_inputs", lambda: [(i, {"name": n}) for i, n in enumerate(names)]
    )


def test_macbook_air_beats_a_continuity_iphone(monkeypatch):
    """The reported failure: iPhone enumerated first, Air's mic not in the preferred list."""
    fake_inputs(monkeypatch, "iPhone 17 Pro Max Microphone", "MacBook Air Microphone")
    assert select_input(CFG)[1] == "MacBook Air Microphone"


@pytest.mark.parametrize(
    "builtin",
    [
        "MacBook Air Microphone",
        "MacBook Pro Microphone",
        "Built-in Microphone",
        "iMac Microphone",
        "Mac mini Microphone",
        "Mac Studio Microphone",
    ],
)
def test_every_mac_builtin_is_preferred(monkeypatch, builtin):
    fake_inputs(monkeypatch, "iPhone Microphone", "Microsoft Teams Audio", builtin)
    assert select_input(CFG)[1] == builtin


def test_virtual_devices_lose_to_real_ones(monkeypatch):
    fake_inputs(monkeypatch, "ZoomAudioDevice", "BlackHole 2ch", "Some USB Mic")
    assert select_input(CFG)[1] == "Some USB Mic"


def test_iphone_still_beats_a_virtual_device(monkeypatch):
    """Deprioritised is not disqualified -- a real mic beats a silent loopback."""
    fake_inputs(monkeypatch, "ZoomAudioDevice", "iPhone Microphone")
    assert select_input(CFG)[1] == "iPhone Microphone"


def test_iphone_is_used_when_it_is_the_only_input(monkeypatch):
    fake_inputs(monkeypatch, "iPhone Microphone")
    assert select_input(CFG)[1] == "iPhone Microphone"


def test_explicit_override_can_still_pick_the_iphone(monkeypatch):
    """--device is the user asking for it; deprioritisation must not block that."""
    fake_inputs(monkeypatch, "MacBook Air Microphone", "iPhone 17 Pro Max Microphone")
    assert select_input(CFG, override="iphone")[1] == "iPhone 17 Pro Max Microphone"


def test_unmatched_override_raises_rather_than_guessing(monkeypatch):
    fake_inputs(monkeypatch, "MacBook Air Microphone")
    with pytest.raises(NoInputDeviceError):
        select_input(CFG, override="Blue Yeti")


def test_no_inputs_raises(monkeypatch):
    fake_inputs(monkeypatch)
    with pytest.raises(NoInputDeviceError):
        select_input(CFG)
