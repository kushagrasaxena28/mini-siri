"""Sweep the endpoint hangover against the synthetic test set.

The hangover is paid on EVERY turn, so it dominates end-to-end latency. Cutting it
is the cheapest latency win available -- but too short truncates the user
mid-sentence, which is far worse than being slightly slow. So latency and
truncation are always reported together.

    uv run python scripts/tune_endpoint.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from mini_siri_local.asr.offline import OfflineAsr
from mini_siri_local.audio.file_source import FileSource
from mini_siri_local.config import AsrConfig, VadConfig
from mini_siri_local.vad.endpoint import Endpointer, TurnEvent
from mini_siri_local.vad.silero import SileroVad

DATA = Path("benchmarks/datasets/synth")
HANGOVERS_MS = [500, 400, 300, 250, 200, 150, 100]

# The metric here is PIPELINE FIDELITY: does VAD + endpointing degrade what the ASR
# would have produced on the whole clip? That isolates our code.
#
# It is deliberately NOT "does it match the expected text". These clips come from
# macOS `say`, and Parakeet mis-hears the synthetic voice in ways that have nothing
# to do with this pipeline ("pause the music" -> "Paused music"). Verified by
# transcribing the full clips with no VAD: identical errors. Real ASR accuracy needs
# real human speech -- see ENGINEERING.md §4.


def p(values: list[float], quantile: float) -> float:
    return float(np.percentile(values, quantile)) if values else float("nan")


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def run_clip(path: Path, asr: OfflineAsr, vad: SileroVad, cfg: VadConfig):
    """Return list of (transcript, endpoint_ms, asr_ms) -- one per detected turn."""
    ep = Endpointer(cfg)
    vad.reset()
    turns = []
    src = FileSource(path, pad_tail_s=0.0)
    for frame, t_ns in src.frames():
        prob = vad(frame)
        ev = ep.update(frame, prob, t_ns)
        if ev is TurnEvent.TURN_END:
            endpoint_ms = (t_ns - ep.t_speech_end_ns) / 1e6
            audio = ep.take_audio()
            t0 = time.monotonic()
            text = asr.transcribe(audio)
            turns.append((text, endpoint_ms, (time.monotonic() - t0) * 1000))
            vad.reset()
        elif ev is TurnEvent.DISCARDED:
            ep.take_audio()
            vad.reset()
    return turns


def main() -> int:
    manifest = json.loads((DATA / "manifest.json").read_text())
    print("loading models ...")
    asr = OfflineAsr(AsrConfig())
    vad = SileroVad()
    asr.warmup()

    # Reference: full-clip transcription, no VAD/endpointing.
    import soundfile as sf

    reference = {}
    for m in manifest:
        audio, _ = sf.read(str(DATA / m["file"]), dtype="float32")
        reference[m["name"]] = norm(asr.transcribe(audio))

    print(
        f"\n{'hangover':>9} {'e2e P50':>8} {'e2e P90':>8} {'endpoint':>9} {'asr':>7} "
        f"{'fidelity':>9} {'split':>6}"
    )
    print("-" * 66)

    rows = []
    for hm in HANGOVERS_MS:
        cfg = VadConfig(hangover_ms=hm)
        e2es, eps, asrs, exact, split = [], [], [], 0, 0
        for m in manifest:
            turns = run_clip(DATA / m["file"], asr, vad, cfg)
            if len(turns) != 1:
                split += 1  # utterance was cut into multiple turns
            if not turns:
                continue
            joined = " ".join(t[0] for t in turns)
            if norm(joined) == reference[m["name"]]:
                exact += 1
            # cost of the FINAL turn is what the user waits for
            _, ep_ms, asr_ms = turns[-1]
            eps.append(ep_ms)
            asrs.append(asr_ms)
            e2es.append(ep_ms + asr_ms)
        rows.append(
            {
                "hangover_ms": hm,
                "e2e_p50": p(e2es, 50),
                "e2e_p90": p(e2es, 90),
                "endpoint_med": p(eps, 50),
                "asr_med": p(asrs, 50),
                "fidelity": exact,
                "n": len(manifest),
                "split": split,
            }
        )
        flag = "  <-- truncates" if split else ""
        print(
            f"{hm:>7}ms {p(e2es, 50):7.0f}ms {p(e2es, 90):7.0f}ms {p(eps, 50):8.0f}ms "
            f"{p(asrs, 50):6.0f}ms {exact:>6}/{len(manifest)} {split:>5}{flag}"
        )

    out = Path("benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "endpoint_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwritten: {out / 'endpoint_sweep.json'}")
    print("\nfidelity = pipeline output matches full-clip ASR output (our code is lossless)")
    print("split    = utterance broken into >1 turn (hangover fired mid-sentence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
