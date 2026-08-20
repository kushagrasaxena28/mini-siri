"""Command-line entry point -- the only one.

    mini-siri-local                    run in the terminal with live timing output
    mini-siri-local --background       menu-bar app, detached from the terminal
    mini-siri-local --check            environment diagnostics, no microphone
    mini-siri-local --say "..."        process one typed command, no audio
    mini-siri-local --replay clip.wav  replay a 16 kHz WAV instead of the microphone
"""

from __future__ import annotations

import argparse
import sys

from .config import VadConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-siri-local",
        description="Offline voice command layer for macOS. Everything runs on-device.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--background",
        action="store_true",
        help="run as a menu-bar app instead of in the terminal",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="run environment diagnostics (models, audio devices, microphone) and exit",
    )
    mode.add_argument(
        "--say",
        metavar="TEXT",
        help="process one command as text, skipping audio entirely",
    )
    mode.add_argument(
        "--replay",
        metavar="WAV",
        help="replay a 16 kHz WAV through the full pipeline instead of the microphone",
    )

    parser.add_argument("--device", help="input device name substring")
    parser.add_argument("--no-tts", action="store_true", help="disable spoken confirmations")
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="stop after ASR (no SLM, executor or TTS) -- useful for tuning the front end",
    )
    parser.add_argument(
        "--hangover-ms",
        type=int,
        default=VadConfig.hangover_ms,
        help=f"silence before a turn is closed (default {VadConfig.hangover_ms})",
    )
    parser.add_argument(
        "--adapter",
        help='LoRA adapter directory for the SLM; pass "" for the base model',
    )
    parser.add_argument(
        "--fewshot-prompt",
        action="store_true",
        help="use the long few-shot prompt (only correct for the un-finetuned base model)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from .config import enforce_offline  # noqa: PLC0415 -- must run before any model import

    enforce_offline()

    if args.check:
        from .diagnostics import run_diagnostics  # noqa: PLC0415 -- only needed for this mode

        return run_diagnostics()

    from .config import SlmConfig, TtsConfig  # noqa: PLC0415

    vad_config = VadConfig(hangover_ms=args.hangover_ms)
    slm_config = SlmConfig(
        # `or` would swallow --adapter "", leaving no way to select the base
        # model -- which is the only configuration --fewshot-prompt is correct for.
        adapter_path=SlmConfig.adapter_path if args.adapter is None else (args.adapter or None),
        fewshot_prompt=args.fewshot_prompt,
    )

    if args.background:
        # Imported last because it pulls in the macOS UI stack.
        from .app.menubar import main as run_menubar  # noqa: PLC0415

        return run_menubar(
            vad_config,
            slm_config,
            TtsConfig(),
            device=args.device,
            enable_tts=not args.no_tts,
        )

    from .pipeline.assistant import Assistant  # noqa: PLC0415 -- loading models is slow

    assistant = Assistant.build(
        vad_config,
        slm_config,
        TtsConfig(),
        transcribe_only=args.transcribe_only,
        enable_tts=not args.no_tts,
    )

    if args.say:
        assistant.handle_transcript(args.say)
        if assistant.speaker is not None:
            assistant.speaker.wait()
        assistant.close()
        return 0

    return assistant.run(device=args.device, replay=args.replay)


if __name__ == "__main__":
    sys.exit(main())
