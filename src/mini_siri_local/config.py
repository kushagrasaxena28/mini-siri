"""Frozen configuration for every stage of the pipeline.

One place for all tunables and model identifiers, so a setting is never hunted
for across modules. Units are in every name -- the 32 ms VAD frame versus the
80 ms encoder frame is the most likely silent bug in this project, and naming is
the cheapest defence.

Structural constants (SAMPLE_RATE_HZ, VAD_FRAME_SAMPLES, ENCODER_FRAME_MS) are
NOT tunable -- they are imposed by the models. The dataclasses below are.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- structural constants: imposed by the models, not free choices --------------

SAMPLE_RATE_HZ = 16_000
VAD_FRAME_SAMPLES = 512  # Silero requires exactly this; not tunable
VAD_FRAME_MS = VAD_FRAME_SAMPLES / SAMPLE_RATE_HZ * 1000  # 32.0
ENCODER_FRAME_MS = 80  # 8x subsampling x 10 ms mel hop -> 12.5 fps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def enforce_offline() -> None:
    """Load models from the local cache only, never the network.

    huggingface_hub revalidates every repo on `from_pretrained`, which is both a
    runtime network call the project claims not to make and ~850 ms of the cold
    start (measured: ASR load 944 -> 192 ms, SLM 1291 -> 1035 ms). Called from
    the entry points, not at import, so `scripts/download_models.py` can still
    reach the network.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = SAMPLE_RATE_HZ
    frame_samples: int = VAD_FRAME_SAMPLES
    # Device is matched by NAME, never index: indices shift when virtual drivers
    # load or headphones are plugged in, and differ between ffmpeg and sounddevice.
    # Substrings, matched case-insensitively -- "macbook" covers Air and Pro, and
    # "built-in" covers the iMac/mini/Studio wording. A list naming only the Pro sent
    # every other Mac down the fallback path.
    preferred_devices: tuple[str, ...] = (
        "macbook",
        "built-in",
        "imac",
        "mac mini",
        "mac studio",
        "external microphone",
    )
    virtual_hints: tuple[str, ...] = (
        "teams",
        "zoom",
        "blackhole",
        "soundflower",
        "loopback",
        "aggregate",
        "virtual",
    )
    # Real hardware, but the wrong choice by default: macOS Continuity offers a nearby
    # iPhone/iPad as an input, and a phone in another room or face-down records silence.
    # Separate from virtual_hints because these ARE usable -- just only when asked for
    # explicitly with --device.
    deprioritised_hints: tuple[str, ...] = ("iphone", "ipad", "continuity")
    queue_max_frames: int = 64  # ~2 s; drop-oldest beyond this


@dataclass(frozen=True)
class VadConfig:
    # Tuned against this machine's built-in mic: speech peaks ~-25 dBFS,
    # noise floor ~-35 dB. Defaults assume a hotter signal, hence the low threshold.
    speech_threshold: float = 0.5
    exit_threshold: float = 0.35  # hysteresis: lower bar to stay in speech
    start_frames: int = 2  # ~64 ms of speech to open a turn
    # 300 ms from the sweep (scripts/tune_endpoint.py): e2e P90 373 ms, comfortably
    # under the 500 ms target with margin for real speech, which pauses more than
    # the synthetic test clips. Re-tune against real recordings before trusting it.
    hangover_ms: int = 300  # silence before the turn is closed
    preroll_ms: int = 320  # audio kept BEFORE speech onset
    min_utterance_ms: int = 200  # shorter than this is noise, discarded
    max_utterance_ms: int = 15_000  # hard stop


@dataclass(frozen=True)
class AsrConfig:
    """FastConformer-TDT. Offline/full-context -- NOT a cache-aware streaming
    checkpoint (see ENGINEERING.md §6)."""

    model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"


@dataclass(frozen=True)
class SileroConfig:
    model_path: Path = MODELS_DIR / "silero_vad.onnx"
    # The model input is 64 context samples + the 512-sample chunk. Feeding a
    # bare 512 does not error -- it silently returns ~0 on loud speech.
    context_samples: int = 64


@dataclass(frozen=True)
class SlmConfig:
    """Intent parsing.

    Default is the LoRA adapter UNFUSED. Fusing is 81 ms faster (278 vs 359 ms
    median) but measurably less accurate, because merging into a 4-bit base
    re-quantises the combined weights:

        adapter  intent 89.5%  exact 73.7%  schema-valid 98.2%  359 ms
        fused    intent 86.0%  exact 71.9%  schema-valid 94.7%  278 ms

    Schema-validity is weighted most heavily: invalid JSON fails closed into a
    spoken "Sorry, I didn't catch that", so it is directly user-visible.

    To trade the other way you must fuse first -- there is no fused checkpoint in
    the repository and download_models.py does not produce one:

        uv run mlx_lm.fuse --model Qwen/Qwen3-1.7B-MLX-4bit \
            --adapter-path adapters --save-path models/qwen3-intent

    then set model_id="models/qwen3-intent" and adapter_path=None.
    """

    model_id: str = "Qwen/Qwen3-1.7B-MLX-4bit"
    adapter_path: str | None = "adapters"  # None for the base (zero-shot) model
    # Completions are ~15 tokens, and generation stops at EOS -- so this cap costs
    # nothing on a normal turn (measured: 251 vs 252 ms median at 48 vs 160). It is
    # sized to fit MAX_NOTE_CHARS: at 48 a long note was truncated mid-JSON and
    # declined, which made the real note limit ~150 chars rather than the 500 the
    # validator advertises.
    max_tokens: int = 160
    temperature: float = 0.0  # deterministic: this is a classifier, not a writer
    # Prefill the fixed system prompt once and reuse its KV cache. Measured
    # Measured 779 ms -> 286 ms per turn.
    reuse_prompt_cache: bool = True
    # True = long few-shot prompt (zero-shot baseline). False = short prompt,
    # which is what a LoRA-tuned checkpoint is trained against. Must be False
    # whenever adapter_path or a fused checkpoint is in use.
    fewshot_prompt: bool = False
    # Carry previous turns so "close it" and "add that to my notes" resolve.
    # History lives in the KV cache, so it costs memory but almost no latency.
    # Declined turns are dropped -- see SlmParser.discard_last_turn.
    keep_history: bool = True


@dataclass(frozen=True)
class TtsConfig:
    model_id: str = "mlx-community/Kokoro-82M-bf16"
    # scripts/download_models.py fetches THIS voice only (the repo carries 54, ~62 MB).
    # Changing it needs a re-download first: the pipeline would otherwise try to fetch the
    # voice at runtime, which HF_HUB_OFFLINE blocks.
    voice: str = "af_heart"
    lang_code: str = "a"  # American English
    sample_rate_hz: int = 24_000  # Kokoro's native rate, NOT the 16 kHz capture rate
    blocksize: int = 512
    # Pre-render fixed confirmations at startup; parameterised ones in the
    # background. Cached playback measured 0 ms synth / <1 ms TTFA.
    prerender: bool = True
    # Release MLX's buffer cache every N background syntheses. Letting it grow
    # unbounded reached 10.7 GB and stalled the next inference by ~30 s.
    cache_release_interval: int = 4
