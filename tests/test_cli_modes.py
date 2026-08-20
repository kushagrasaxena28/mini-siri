"""CLI mode selection.

The default mode is the terminal pipeline and `--background` opts into the
menu bar. That inversion is easy to flip back by accident, and flipping it
silently changes what `uv run mini-siri-local` does, so it is pinned here.
"""

from __future__ import annotations

import pytest

from mini_siri_local.cli import build_parser


def test_no_arguments_selects_terminal_mode():
    args = build_parser().parse_args([])
    assert args.background is False
    assert args.check is False
    assert args.say is None
    assert args.replay is None


def test_background_flag_selects_the_menu_bar():
    assert build_parser().parse_args(["--background"]).background is True


def test_terminal_flag_no_longer_exists():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--terminal"])


@pytest.mark.parametrize(
    "pair",
    [
        ["--background", "--check"],
        ["--background", "--say", "open notes"],
        ["--background", "--replay", "clip.wav"],
        ["--check", "--replay", "clip.wav"],
    ],
)
def test_modes_are_mutually_exclusive(pair):
    with pytest.raises(SystemExit):
        build_parser().parse_args(pair)


@pytest.mark.parametrize(
    "argv",
    [
        ["--background", "--no-tts"],
        ["--background", "--device", "MacBook Pro Microphone"],
        ["--background", "--adapter", "adapters"],
        ["--background", "--hangover-ms", "250"],
    ],
)
def test_background_composes_with_run_options(argv):
    """`--background` used to be the implicit default and ignored these; now
    that it is explicit, silently dropping them would be a real footgun."""
    args = build_parser().parse_args(argv)
    assert args.background is True


def test_hangover_default_comes_from_config():
    from mini_siri_local.config import VadConfig

    assert build_parser().parse_args([]).hangover_ms == VadConfig.hangover_ms


def test_empty_adapter_selects_the_base_model():
    """`or` used to swallow --adapter "", so the base model -- the only
    configuration --fewshot-prompt is correct for -- was unreachable."""
    from mini_siri_local.config import SlmConfig

    def resolve(argv):
        args = build_parser().parse_args(argv)
        return SlmConfig.adapter_path if args.adapter is None else (args.adapter or None)

    assert resolve([]) == SlmConfig.adapter_path
    assert resolve(["--adapter", ""]) is None
    assert resolve(["--adapter", "adapters_v2"]) == "adapters_v2"
