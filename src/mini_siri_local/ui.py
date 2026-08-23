"""Terminal rendering for the assistant loop.

Split out of pipeline/assistant.py so the loop reads as orchestration rather than
formatting. The pipeline calls these methods; nothing here knows how a turn is produced.

Colour is opt-out and auto-detected: escape codes are emitted only when stdout is a TTY
and NO_COLOR is unset, so piping to a file or a test harness yields plain text.
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import sys
import textwrap
from dataclasses import dataclass

RULE = "─"
MAX_RULE = 66


def _supports_colour(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


@dataclass(frozen=True)
class Palette:
    """ANSI codes, or empty strings when colour is off -- so format strings are
    identical either way and there is only one rendering path to reason about."""

    dim: str = ""
    bold: str = ""
    green: str = ""
    yellow: str = ""
    cyan: str = ""
    red: str = ""
    reset: str = ""

    @classmethod
    def enabled(cls) -> Palette:
        return cls(
            dim="\033[2m",
            bold="\033[1m",
            green="\033[32m",
            yellow="\033[33m",
            cyan="\033[36m",
            red="\033[31m",
            reset="\033[0m",
        )


class Console:
    """Renders the assistant's turns. One instance per run."""

    def __init__(self, stream=None, colour: bool | None = None):
        self.stream = stream or sys.stdout
        use = _supports_colour(self.stream) if colour is None else colour
        self.c = Palette.enabled() if use else Palette()
        self.interactive = use
        self._totals: list[float] = []
        self._counts = {"action": 0, "declined": 0, "rejected": 0}

    # -- plumbing ---------------------------------------------------------

    def _w(self, text: str = "") -> None:
        print(text, file=self.stream)

    def _rule(self) -> str:
        width = min(MAX_RULE, shutil.get_terminal_size((80, 24)).columns - 2)
        return f"{self.c.dim}  {RULE * width}{self.c.reset}"

    def _transient(self, text: str) -> None:
        """Status that the next line overwrites. Skipped when not a terminal."""
        if self.interactive:
            body = f"\r  {self.c.dim}{text}{self.c.reset}\033[K"
            print(body, end="", file=self.stream, flush=True)

    def _clear_transient(self) -> None:
        if self.interactive:
            print("\r\033[K", end="", file=self.stream, flush=True)

    # -- startup ----------------------------------------------------------

    def loading(self, what: str) -> None:
        self._transient(f"loading {what} …")

    def ready(self, loads_ms: dict[str, float], warmup_ms: float, device: str | None) -> None:
        self._clear_transient()
        c = self.c
        self._w()
        self._w(f"  {c.bold}mini-siri-local{c.reset}")
        self._w(self._rule())
        parts = " · ".join(f"{name} {ms:.0f}ms" for name, ms in loads_ms.items())
        self._w(f"  {c.dim}models {c.reset} {parts}")
        self._w(f"  {c.dim}warm-up{c.reset}  {warmup_ms:.0f} ms")
        if device:
            self._w(f"  {c.dim}input  {c.reset}  {device}")
        self._w(self._rule())

    def listening(self) -> None:
        c = self.c
        self._w(f"  {c.green}●{c.reset} listening {c.dim}— ctrl-c to stop{c.reset}")
        self._w()

    def replaying(self, path: str) -> None:
        self._w(f"  {self.c.dim}replaying {path}{self.c.reset}")
        self._w()

    def tts_unavailable(self, module: str) -> None:
        self._clear_transient()
        c = self.c
        self._w(f"  {c.yellow}!{c.reset} speech synthesis unavailable ({module});"
                f" continuing silently")
        self._w(f"    {c.dim}install it with: uv sync --extra tts --extra dev{c.reset}")

    # -- turns ------------------------------------------------------------

    def hearing(self) -> None:
        self._transient("hearing you …")

    def thinking(self) -> None:
        self._transient("working …")

    def _head(self, n: int, transcript: str) -> None:
        self._clear_transient()
        c = self.c
        self._w(f"  {c.dim}{n:>2}{c.reset}  {c.bold}▸{c.reset} {transcript}")

    def _timing(self, timing, *, stages: bool = True) -> None:
        c = self.c
        total = f"{timing.total_ms:.0f} ms"
        if not stages:
            self._w(f"      {c.dim}{total}{c.reset}")
            return
        # An em dash rather than 0 for TTFA: NaN means "not measurable here" (the player
        # was already busy), and printing 0 would claim a measurement that was not made.
        say = f"{timing.ttfa_ms:.0f}" if math.isfinite(timing.ttfa_ms) else "—"
        detail = (
            f"wait {timing.endpoint_ms:.0f} · asr {timing.asr_ms:.0f} · "
            f"model {timing.slm_ms:.0f} · act {timing.exec_ms:.0f} · say {say}"
        )
        self._w(f"      {c.dim}{total}  ·  {detail}{c.reset}")

    def action(self, n: int, timing, args: dict, speech: str, cached: bool) -> None:
        c = self.c
        self._head(n, timing.transcript)
        shown = "  ".join(f"{k}={v}" for k, v in args.items())
        detail = f"  {c.dim}{shown}{c.reset}" if shown else ""
        self._w(f"      {c.green}✓{c.reset} {timing.intent}{detail}")
        if speech:
            tag = "" if cached else f" {c.dim}(synth){c.reset}"
            self._w(f'      {c.cyan}♪{c.reset} {c.dim}"{speech}"{c.reset}{tag}')
        self._timing(timing)
        self._w()
        if math.isfinite(timing.total_ms):
            self._totals.append(timing.total_ms)
        self._counts["action"] += 1

    def declined(self, n: int, timing) -> None:
        c = self.c
        self._head(n, timing.transcript)
        self._w(f"      {c.dim}· declined — not a command{c.reset}")
        self._timing(timing)
        self._w()
        if math.isfinite(timing.total_ms):
            self._totals.append(timing.total_ms)
        self._counts["declined"] += 1

    def rejected(self, n: int, transcript: str, reason: str, raw: str) -> None:
        c = self.c
        self._head(n, transcript)
        width = min(MAX_RULE, shutil.get_terminal_size((80, 24)).columns - 10)
        lines = textwrap.wrap(reason, width=width) or [reason]
        self._w(f"      {c.yellow}✗{c.reset} rejected {c.dim}— {lines[0]}{c.reset}")
        for extra in lines[1:]:
            self._w(f"        {c.dim}{extra}{c.reset}")
        if raw:
            self._w(f"        {c.dim}{raw[:96]}{c.reset}")
        self._w()
        self._counts["rejected"] += 1

    def transcript_only(self, n: int, text: str, endpoint_ms: float, asr_ms: float) -> None:
        c = self.c
        self._head(n, text)
        self._w(f"      {c.dim}wait {endpoint_ms:.0f} · asr {asr_ms:.0f} ms{c.reset}")
        self._w()

    def empty(self) -> None:
        self._clear_transient()
        self._w(f"  {self.c.dim}(nothing heard){self.c.reset}")

    def timer(self, message: str) -> None:
        self._clear_transient()
        self._w(f"  {self.c.cyan}⏰ {message}{self.c.reset}")
        self._w()

    # -- shutdown ---------------------------------------------------------

    def summary(self) -> None:
        self._clear_transient()
        c = self.c
        done = sum(self._counts.values())
        if not done:
            self._w(f"\n  {c.dim}stopped — no turns{c.reset}")
            return
        bits = [f"{done} turn{'s' if done != 1 else ''}"]
        plural = {"action": "actions", "declined": "declined", "rejected": "rejected"}
        bits.extend(
            f"{self._counts[label]} {plural[label] if self._counts[label] != 1 else label}"
            for label in ("action", "declined", "rejected")
            if self._counts[label]
        )
        self._w(self._rule())
        self._w(f"  {' · '.join(bits)}")
        if self._totals:
            ordered = sorted(self._totals)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
            self._w(
                f"  {c.dim}end-to-end  median {statistics.median(ordered):.0f} ms · "
                f"p95 {p95:.0f} ms · best {ordered[0]:.0f} ms{c.reset}"
            )
        self._w(self._rule())
