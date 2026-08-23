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

# `patterns` limits what snapshot_download fetches. Kokoro's repo is 122 files / 389 MB.
#
# These patterns MUST be a superset of mlx_audio's own DEFAULT_ALLOW_PATTERNS. At load time
# mlx_audio calls snapshot_download itself, and huggingface_hub then verifies that snapshot
# is COMPLETE for those patterns -- so anything it would ask for and we did not fetch makes
# the model unloadable offline with `IncompleteSnapshotError`, not merely degraded.
#
# That rules out fetching a single voice: mlx_audio asks for "*.safetensors", and fnmatch's
# `*` matches `/`, so it demands all 54 voice packs. What IS safe to skip is everything it
# never asks for -- demo audio, the duplicate .pt voices, and markdown: 34 MB of 389 MB.
HF_REPOS = [
    ("mlx-community/parakeet-tdt-0.6b-v3", "ASR encoder+decoder (FastConformer-TDT)", None, False),
    ("Qwen/Qwen3-1.7B-MLX-4bit", "SLM for intent -> JSON", None, False),
    (
        "mlx-community/Kokoro-82M-bf16",
        "TTS confirmations",
        ["*.json", "*.safetensors"],
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
