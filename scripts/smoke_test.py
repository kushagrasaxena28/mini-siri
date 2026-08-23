"""Load every model, measure residency and load time, and transcribe a real file.

Answers the question the 'keep everything warm' design depends on:
does the whole stack fit in memory at once, and how long is cold start?

    uv run python scripts/smoke_test.py [audio.wav]
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import mlx.core as mx

MODELS = Path(__file__).resolve().parent.parent / "models"


def rss_gb() -> float:
    # macOS reports ru_maxrss in bytes (Linux uses KiB)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def mlx_gb() -> float:
    return mx.get_active_memory() / 1e9


def stamp(label: str, t0: float) -> None:
    print(
        f"  {label:<26} {time.monotonic() - t0:6.2f}s   "
        f"rss {rss_gb():5.2f} GB   mlx {mlx_gb():5.2f} GB"
    )


def main() -> int:
    print("mini-siri-local smoke test -- model residency and cold start\n")
    print(f"  {'baseline':<26} {'':>6}    rss {rss_gb():5.2f} GB   mlx {mlx_gb():5.2f} GB")

    # --- ASR -------------------------------------------------------------
    t0 = time.monotonic()
    from parakeet_mlx import from_pretrained

    asr = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    stamp("parakeet-tdt-0.6b-v3", t0)

    # --- VAD -------------------------------------------------------------
    t0 = time.monotonic()
    import numpy as np
    import onnxruntime as ort

    vad = ort.InferenceSession(str(MODELS / "silero_vad.onnx"), providers=["CPUExecutionProvider"])
    stamp("silero-vad", t0)

    # --- SLM -------------------------------------------------------------
    t0 = time.monotonic()
    from mlx_lm import load as load_lm

    _slm, _tokenizer = load_lm("Qwen/Qwen3-1.7B-MLX-4bit")  # held to measure residency
    stamp("qwen3-1.7b-4bit", t0)

    peak_rss, peak_mlx = rss_gb(), mx.get_peak_memory() / 1e9
    print(
        f"\n  PEAK: rss {peak_rss:.2f} GB, mlx peak {peak_mlx:.2f} GB "
        f"(of 16 GB, ~12.7 GB usable by Metal)"
    )

    # --- VAD sanity: one 512-sample frame --------------------------------
    print("\nVAD frame check")
    state = np.zeros((2, 1, 128), dtype=np.float32)
    frame = np.zeros((1, 512), dtype=np.float32)
    t0 = time.monotonic()
    out = vad.run(None, {"input": frame, "state": state, "sr": np.array(16000, dtype=np.int64)})
    dt_ms = (time.monotonic() - t0) * 1000
    print(f"  512 samples (32 ms) -> speech_prob {float(out[0][0][0]):.3f} in {dt_ms:.2f} ms")

    # --- ASR on a real recording ------------------------------------------
    default_clip = Path(__file__).resolve().parent.parent / "benchmarks/datasets/synth/timer.wav"
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else default_clip
    if audio.exists():
        print(f"\nTranscribing {audio}")
        # Read with soundfile and hand over samples, exactly as asr/offline.py does.
        # Passing a PATH to parakeet's transcribe() routes through its own loader, which
        # shells out to ffmpeg -- a system dependency nothing else in this project needs.
        import soundfile as sf
        from parakeet_mlx.audio import get_logmel

        samples, _ = sf.read(str(audio), dtype="float32")
        mel = get_logmel(mx.array(samples), asr.preprocessor_config)

        t0 = time.monotonic()
        result = asr.generate(mel)
        print(f"  cold : {time.monotonic() - t0:.2f}s")
        t0 = time.monotonic()
        result = asr.generate(mel)
        print(f"  warm : {time.monotonic() - t0:.2f}s")
        print(f"  text : {result[0].text!r}")
    else:
        print(f"\n(no audio at {audio} -- skipping transcription)")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
