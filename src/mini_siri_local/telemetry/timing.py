"""Timing primitives.

Deliberately minimal. The whole instrumentation budget is one timing line per
turn, which the pipeline composes itself, plus `scripts/bench_pipeline.py` for
per-stage percentiles. Span trees and a regression harness would be more
machinery than a single-process pipeline with six stages can justify.
"""

from __future__ import annotations

import time


def now_ns() -> int:
    """Monotonic nanoseconds.

    Never `time.time()`: the wall clock can step (NTP, DST) and would produce
    negative or wildly wrong durations. Every timestamp in the audio path uses
    this so they are all in one comparable domain.
    """
    return time.monotonic_ns()
