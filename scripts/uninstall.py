"""Remove everything this project downloaded OUTSIDE the repository.

Deleting the clone is not enough: the model weights live in the shared Hugging Face
cache under $HOME, and dictated notes may have been written to a fallback file there
too. This removes those, so afterwards deleting the repo directory finishes the job.

    uv run python scripts/uninstall.py           # dry run -- shows what WOULD be removed
    uv run python scripts/uninstall.py --yes     # actually remove models and caches
    uv run python scripts/uninstall.py --yes --notes   # also remove dictated notes

Deliberately conservative:

  * Only the three model repos THIS project downloads are touched. The Hugging Face
    cache is shared, so anything else in it belongs to another project and is left alone.
  * Notes are user data, not a cache. They need a separate --notes flag, and the file
    is never removed by --yes on its own.
  * espeak-ng, uv and Homebrew are general-purpose tools that other things may depend
    on. They are reported, never removed.
  * Nothing in Apple Notes or Reminders is touched. This script cannot tell your notes
    from the assistant's, and guessing would be destroying data it does not own.

Pure standard library on purpose -- it must still run if the virtualenv is broken:

    python3 scripts/uninstall.py --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must stay in step with HF_REPOS in scripts/download_models.py.
MODEL_REPOS = (
    "models--mlx-community--parakeet-tdt-0.6b-v3",
    "models--Qwen--Qwen3-1.7B-MLX-4bit",
    "models--mlx-community--Kokoro-82M-bf16",
)

NOTES_DIR = Path.home() / ".mini-siri-local"


def hf_hub_cache() -> Path:
    """Where huggingface_hub actually keeps models, honouring its env overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def size_of(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() or p.is_symlink():
                total += p.lstat().st_size
        except OSError:
            continue
    return total


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def targets(include_notes: bool) -> list[tuple[Path, str]]:
    """(path, description) for everything this run would remove."""
    hub = hf_hub_cache()
    found: list[tuple[Path, str]] = [
        (hub / repo, "model weights") for repo in MODEL_REPOS if (hub / repo).exists()
    ]
    models_dir = REPO_ROOT / "models"
    if models_dir.exists():
        found.append((models_dir, "in-repo models (VAD, any fused checkpoint)"))
    if include_notes and NOTES_DIR.exists():
        found.append((NOTES_DIR, "dictated notes -- USER DATA"))
    return found


def report_leftovers() -> None:
    """Things a user may also want gone, that this script will not touch."""
    print("\nNot removed -- shared with other software, or not ours to delete:\n")

    hub = hf_hub_cache()
    if hub.exists():
        others = [
            d for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")
            and d.name not in MODEL_REPOS
        ]
        if others:
            print(f"  {len(others)} other model(s) in the shared cache at {hub}:")
            for d in others[:5]:
                print(f"      {d.name}")
            if len(others) > 5:
                print(f"      ... and {len(others) - 5} more")
        else:
            print(f"  No OTHER models in {hub} --")
            print("  once this project's are gone the directory can be removed entirely.")

    if NOTES_DIR.exists():
        print(f"  {NOTES_DIR}  (dictated notes -- pass --notes to remove)")

    print("""
  Homebrew packages, if you installed them only for this:
      brew uninstall espeak-ng          # speech synthesis pronunciation data
      brew uninstall uv                 # Python package manager -- check nothing else uses it

  Anything the assistant wrote into Apple Notes or Reminders. Those are real entries in
  your own apps; delete them yourself so nothing of yours is caught by a pattern match.

  macOS permissions, under System Settings > Privacy & Security:
      Microphone      -- revoke access for your terminal app
      Automation      -- revoke Notes / Reminders control for your terminal app
""")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument(
        "--notes",
        action="store_true",
        help=f"also delete dictated notes in {NOTES_DIR} (user data, not a cache)",
    )
    args = ap.parse_args()

    found = targets(include_notes=args.notes)
    if not found:
        print("Nothing to remove -- no downloaded models found.")
        report_leftovers()
        return 0

    verb = "Removing" if args.yes else "Would remove"
    print(f"\n{verb}:\n")
    total = 0
    for path, what in found:
        n = size_of(path)
        total += n
        print(f"  {human(n):>9}  {path}")
        print(f"             {what}")
    print(f"\n  {human(total):>9}  total")

    if not args.yes:
        print("\nDry run -- nothing deleted. Re-run with --yes to remove.")
        if not args.notes and NOTES_DIR.exists():
            print(f"Notes at {NOTES_DIR} are NOT included; add --notes to remove them too.")
        report_leftovers()
        return 0

    print()
    failed = 0
    for path, _ in found:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  removed  {path}")
        except OSError as exc:
            print(f"  FAILED   {path}: {exc}")
            failed += 1

    print(f"\nFreed {human(total)}." if not failed else f"\n{failed} item(s) could not be removed.")
    report_leftovers()
    print(f"Finally, delete the repository itself:\n\n    rm -rf {REPO_ROOT}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
