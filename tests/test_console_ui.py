"""Terminal rendering.

The console must degrade to plain text when stdout is not a terminal -- otherwise escape
codes end up in piped logs and in anything that scrapes the output.
"""

from __future__ import annotations

import io

from mini_siri_local.pipeline.assistant import TurnTiming
from mini_siri_local.ui import Console

TIMING = TurnTiming(
    transcript="Close WhatsApp.",
    intent="close_app",
    endpoint_ms=288.0,
    asr_ms=0.0,
    slm_ms=290.0,
    validate_ms=0.03,
    exec_ms=40.3,
    ttfa_ms=16.3,
)


def render(fn) -> str:
    buf = io.StringIO()
    console = Console(stream=buf, colour=False)
    fn(console)
    return buf.getvalue()


def test_no_escape_codes_when_colour_is_off():
    out = render(lambda c: c.action(2, TIMING, {"app_name": "WhatsApp"}, "Closing WhatsApp.", True))
    assert "\033[" not in out


def test_action_shows_transcript_intent_args_and_speech():
    out = render(lambda c: c.action(2, TIMING, {"app_name": "WhatsApp"}, "Closing WhatsApp.", True))
    assert "Close WhatsApp." in out
    assert "close_app" in out
    assert "app_name=WhatsApp" in out
    assert "Closing WhatsApp." in out
    assert "635 ms" in out or "634 ms" in out


def test_declined_is_visibly_different_from_an_action():
    declined = render(lambda c: c.declined(1, TIMING))
    assert "declined" in declined
    assert "✓" not in declined


def test_rejected_shows_the_reason_and_the_raw_output():
    out = render(
        lambda c: c.rejected(7, "What is my IP address?", "item must be one of ...", '{"a":1}')
    )
    assert "rejected" in out
    assert "item must be one of" in out
    assert '{"a":1}' in out


def test_summary_counts_every_outcome():
    def run(c):
        c.action(1, TIMING, {}, "ok", True)
        c.declined(2, TIMING)
        c.rejected(3, "x", "why", "")
        c.summary()

    out = render(run)
    assert "3 turns" in out
    assert "1 action" in out and "1 declined" in out and "1 rejected" in out


def test_summary_is_quiet_when_nothing_happened():
    out = render(lambda c: c.summary())
    assert "no turns" in out


def test_transient_status_is_suppressed_when_not_a_terminal():
    """hearing()/thinking() must write nothing to a pipe -- they rely on \\r overwriting."""
    out = render(lambda c: (c.hearing(), c.thinking()))
    assert out == ""


def test_colour_is_emitted_when_explicitly_enabled():
    buf = io.StringIO()
    Console(stream=buf, colour=True).declined(1, TIMING)
    assert "\033[" in buf.getvalue()
