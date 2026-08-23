"""Download every model the pipeline needs, so runtime is fully offline.

uv run python scripts/download_models.py
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# `patterns` limits what snapshot_download fetches. Kokoro's repo is 122 files / 389 MB,
# of which 327 MB is the one weights file; the rest is 54 voice packs we never load plus
# demo audio. Fetching only the shipped voice saves 62 MB.
#
# af_heart must be fetched HERE and not left to the runtime: KokoroPipeline.load_single_voice
# lazily snapshot_downloads a missing voice, which under HF_HUB_OFFLINE fails outright --
# and would be a network call on the runtime path regardless. Changing TtsConfig.voice
# therefore needs a re-download; noted in config.py.
HF_REPOS = [
    ("mlx-community/parakeet-tdt-0.6b-v3", "ASR encoder+decoder (FastConformer-TDT)", None, False),
    ("Qwen/Qwen3-1.7B-MLX-4bit", "SLM for intent -> JSON", None, False),
    (
        "mlx-community/Kokoro-82M-bf16",
        "TTS confirmations",
        # Explicit paths, not globs: huggingface_hub matches with fnmatch, whose `*`
        # also matches `/`, so "*.safetensors" would keep all 54 voice packs.
        ["config.json", "kokoro-v1_0.safetensors", "voices/af_heart.safetensors"],
        True,
    ),
]

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-tts",
        action="store_true",
        help="skip the speech-synthesis model (matches ./setup.sh --lite)",
    )
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    failed = []

    for repo, purpose, patterns, is_tts in HF_REPOS:
        if is_tts and args.skip_tts:
            print(f"\n>>> {repo}\n    {purpose}\n    skipped (--skip-tts)")
            continue
        print(f"\n>>> {repo}\n    {purpose}")
        t0 = time.monotonic()
        try:
            path = snapshot_download(repo_id=repo, allow_patterns=patterns)
            print(f"    ok  {human(dir_size(Path(path)))} in {time.monotonic() - t0:.0f}s")
        except Exception as e:
            print(f"    FAILED: {str(e)[:160]}")
            failed.append(repo)

    print("\n>>> silero-vad (ONNX)\n    voice activity detection")
    vad_path = MODELS_DIR / "silero_vad.onnx"
    if vad_path.exists():
        print(f"    ok  cached, {human(vad_path.stat().st_size)}")
    else:
        try:
            urllib.request.urlretrieve(SILERO_URL, vad_path)
            print(f"    ok  {human(vad_path.stat().st_size)}")
        except Exception as e:
            print(f"    FAILED: {str(e)[:160]}")
            failed.append("silero-vad")

    total = dir_size(MODELS_DIR)
    print(f"\ntotal on disk: {human(total)}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all models present -- runtime can now run offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
