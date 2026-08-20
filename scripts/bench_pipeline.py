"""Per-stage latency benchmark for the whole pipeline, in ONE process.

Separate processes hide steady state behind cold start, and the background TTS
pre-render competes for the GPU with ASR and the SLM for the first few seconds.
So: load everything, wait for the pre-render, warm the interleaved path, then
measure.

This measures LATENCY. It is not an accuracy benchmark: the clip set is eight
`say`-generated utterances and Parakeet mishears that synthetic voice. Accuracy
lives in scripts/eval_slm.py, against the held-out intent set.

The system boundary (osascript, open, mdfind) is stubbed. This measures OUR
code; a 200 ms osascript round trip is not something the pipeline can improve,
and a benchmark that opened Terminal forty times would be unusable.

    uv run python scripts/bench_pipeline.py
    uv run python scripts/bench_pipeline.py --repeats 10 --no-tts
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from mini_siri_local.config import enforce_offline

enforce_offline()

import mlx.core as mx  # noqa: E402 -- after enforce_offline, which must precede model loading

from mini_siri_local.asr.offline import OfflineAsr  # noqa: E402
from mini_siri_local.audio.file_source import FileSource  # noqa: E402
from mini_siri_local.config import AsrConfig, SlmConfig, VadConfig  # noqa: E402
from mini_siri_local.executor import macos  # noqa: E402
from mini_siri_local.executor.handlers import Executor  # noqa: E402
from mini_siri_local.pipeline.assistant import Assistant  # noqa: E402
from mini_siri_local.slm.parser import SlmParser  # noqa: E402
from mini_siri_local.vad.silero import SileroVad  # noqa: E402

DATA = Path("benchmarks/datasets/synth")
OUT_PATH = Path("benchmarks/results/pipeline_latency.json")
WARMUP_ROUNDS = 3
STAGES = ("endpoint", "asr", "slm", "validate", "exec", "tts", "e2e")

EXPECTED_INTENT = {
    "timer": "set_timer",
    "open_app": "open_app",
    "volume": "set_volume",
    "media": "media_control",
    "time": "get_time",
    "note": "capture_note",
    "pause_mid": "set_timer",
    "pause_mid2": "search_notes",
}


def stub_system_calls() -> None:
    macos.dispatch = lambda argv: None
    macos.open_application = lambda name: None
    macos.open_file_path = lambda path: None
    macos.reveal_file_path = lambda path: None
    macos.quit_application = lambda name: None
    macos.run_osascript = lambda script, timeout_s=3.0: (False, "benchmark stub")
    macos.is_running = lambda name: False
    macos.spotlight_search = lambda query, limit, search_root: []


def summarise(name: str, samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "stage": name,
        "n": len(ordered),
        "best_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[int(len(ordered) * 0.95)],
        "worst_ms": ordered[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="passes over the clip set")
    ap.add_argument("--no-tts", action="store_true", help="skip synthesis and playback")
    args = ap.parse_args()

    stub_system_calls()

    load_ms: dict[str, float] = {}

    def timed(name: str, build):
        started = time.monotonic()
        built = build()
        load_ms[name] = (time.monotonic() - started) * 1000
        return built

    asr = timed("asr", lambda: OfflineAsr(AsrConfig()))
    timed("vad", SileroVad)  # loaded here only to time it; listen() builds its own
    slm = timed("slm", lambda: SlmParser(SlmConfig()))
    executor = timed("executor", lambda: Executor(notify=lambda message: None))

    speaker = None
    if not args.no_tts:
        from mini_siri_local.tts.speaker import Speaker

        # Includes the blocking pre-render AND the background one: measure steady
        # state, not contention with synthesis still running on the GPU.
        speaker = timed("tts", Speaker)
        speaker.wait_for_prerender(timeout_s=300)
        print(f"  {speaker.cached_phrase_count} phrases pre-rendered")

    print("\nCOLD START")
    for name, ms in load_ms.items():
        print(f"  {name:<10} {ms:8.0f} ms")
    print(f"  {'total':<10} {sum(load_ms.values()):8.0f} ms")

    silence = np.zeros(16_000, dtype=np.float32)
    started = time.monotonic()
    for _ in range(WARMUP_ROUNDS):
        asr.transcribe(silence)
        slm.parse("what time is it")
        if speaker is not None:
            speaker.synthesize("Noted.")
    print(f"  {'warm-up':<10} {(time.monotonic() - started) * 1000:8.0f} ms")

    manifest = json.loads((DATA / "manifest.json").read_text())
    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    turns: list[dict] = []

    # Drive the REAL loop. Reimplementing it here would measure something the
    # product does not do -- notably it would miss the speculative transcription
    # that hides ASR inside the endpoint hangover.
    assistant = Assistant(asr, slm, executor, speaker, VadConfig())

    for _ in range(args.repeats):
        for clip in manifest:
            source = FileSource(DATA / clip["file"], pad_tail_s=1.0)
            assistant.last_turn = None
            assistant.listen(source)
            if speaker is not None:
                # The barge-in guard skips frames while a confirmation plays, and a
                # FileSource outruns real time -- without this the next clip is
                # swallowed whole.
                speaker.wait(timeout_s=10)
            timing = assistant.last_turn
            if timing is None:
                turns.append({"clip": clip["name"], "ok": False, "reason": "no turn produced"})
                continue
            for stage, value in (
                ("endpoint", timing.endpoint_ms),
                ("asr", timing.asr_ms),
                ("slm", timing.slm_ms),
                ("validate", timing.validate_ms),
                ("exec", timing.exec_ms),
                ("tts", timing.ttfa_ms),
                ("e2e", timing.total_ms),
            ):
                samples[stage].append(value)
            turns.append(
                {
                    "clip": clip["name"],
                    "ok": True,
                    "text": timing.transcript,
                    "intent": timing.intent,
                    "correct": timing.intent == EXPECTED_INTENT.get(clip["name"]),
                    "e2e_ms": timing.total_ms,
                }
            )

    print(f"\nWARM PER-STAGE (ms) over {len(samples['e2e'])} turns")
    print(f"  {'stage':<14}{'best':>9}{'median':>9}{'mean':>9}{'p95':>9}{'worst':>9}")
    stats = [summarise(stage, samples[stage]) for stage in STAGES if samples[stage]]
    for row in stats:
        print(
            f"  {row['stage']:<14}{row['best_ms']:9.2f}{row['median_ms']:9.2f}"
            f"{row['mean_ms']:9.2f}{row['p95_ms']:9.2f}{row['worst_ms']:9.2f}"
        )

    good = [t for t in turns if t["ok"]]
    # NOT an accuracy measurement -- these eight clips come from macOS `say`, and
    # Parakeet mishears the synthetic voice in ways real speech does not
    # ("pause the music" -> "Posed music"). Intent accuracy is scripts/eval_slm.py.
    # What this line catches is a stage silently producing nothing.
    print(f"\n  matched expected intent : {sum(t['correct'] for t in good)}/{len(good)}")
    print(f"  rejected by validation  : {len(turns) - len(good)}/{len(turns)}")
    print(
        f"  memory         : mlx active {mx.get_active_memory() / 1e9:.2f} GB, "
        f"cache {mx.get_cache_memory() / 1e9:.2f} GB, peak {mx.get_peak_memory() / 1e9:.2f} GB"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "repeats": args.repeats,
                "tts": not args.no_tts,
                "load_ms": load_ms,
                "stages": stats,
                "turns": turns,
                "mlx_peak_gb": mx.get_peak_memory() / 1e9,
            },
            indent=2,
        )
    )
    print(f"\nwritten: {OUT_PATH}")

    executor.shutdown()
    if speaker is not None:
        speaker.wait(timeout_s=10)  # let the output callback drain before closing
        speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
