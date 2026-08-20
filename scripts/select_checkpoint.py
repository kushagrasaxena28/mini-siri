"""Evaluate every saved LoRA checkpoint and report which is actually best.

Ranks on the VALIDATION split. Selecting on test would make the headline number
a training signal rather than a held-out one -- the checkpoint would have been
chosen because it scored well on exactly the rows later used to report accuracy.

The last checkpoint is NOT reliably the best one. In the first training run of
this project, validation loss went 0.118 (iter 200) -> 0.072 (iter 300) ->
0.083 (iter 400), and held-out exact-match followed it down: iteration 300
scored 78.4% while the final iteration 400 scored 72.5%. Taking the final
adapter would have shipped a measurably worse model.

    uv run python scripts/select_checkpoint.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ADAPTER_DIR = Path("adapters")
# Exact-match points within which two checkpoints are considered tied. ~100
# validation rows means one row is ~1 pp.
NOISE_BAND_PP = 3.0
STAGING_DIR = Path("adapters_eval")


def find_checkpoints(adapter_dir: Path) -> list[tuple[int, Path]]:
    """Return [(iteration, path)] sorted by iteration, plus the final adapter."""
    checkpoints = []
    for path in adapter_dir.glob("*_adapters.safetensors"):
        match = re.match(r"(\d+)_adapters\.safetensors", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    return checkpoints


def evaluate(checkpoint: Path, adapter_dir: Path, test_path: str) -> dict | None:
    """Stage one checkpoint as adapters.safetensors and run the evaluator.

    Reads the evaluator's own JSON rather than scraping its stdout: a change to
    a print format should not silently turn every metric into NaN.
    """
    STAGING_DIR.mkdir(exist_ok=True)
    shutil.copy(adapter_dir / "adapter_config.json", STAGING_DIR / "adapter_config.json")
    shutil.copy(checkpoint, STAGING_DIR / "adapters.safetensors")
    metrics_path = STAGING_DIR / "metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_slm.py",
            "--adapter",
            str(STAGING_DIR),
            "--test",
            test_path,
            "--out",
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"    eval failed: {result.stderr[-300:]}")
        return None

    raw = json.loads(metrics_path.read_text())
    n = raw["n"]
    return {
        "schema_valid": 100 * raw["schema_valid"] / n,
        "intent_acc": 100 * raw["intent_acc"] / n,
        "exact_match": 100 * raw["exact_match"] / n,
        "unknown_precision": 100 * raw["unknown_precision"],
        "unknown_recall": 100 * raw["unknown_recall"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    # VALIDATION, not test: picking a checkpoint by test score is model selection
    # on the held-out set, and the number it then reports is no longer held out.
    parser.add_argument("--eval-on", default="datasets/intent/valid.jsonl")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    checkpoints = find_checkpoints(adapter_dir)
    if not checkpoints:
        print(f"no numbered checkpoints in {adapter_dir}")
        return 1

    print(f"evaluating {len(checkpoints)} checkpoints on {args.eval_on}\n")
    print(f"{'iter':>6} {'valid':>8} {'intent':>8} {'exact':>8} {'unk_P':>8} {'unk_R':>8}")
    print("-" * 52)

    results = []
    for iteration, path in checkpoints:
        metrics = evaluate(path, adapter_dir, args.eval_on)
        if metrics is None:
            continue
        metrics["iteration"] = iteration
        results.append(metrics)
        print(
            f"{iteration:>6} {metrics['schema_valid']:>7.1f}% {metrics['intent_acc']:>7.1f}% "
            f"{metrics['exact_match']:>7.1f}% {metrics['unknown_precision']:>7.1f}% "
            f"{metrics['unknown_recall']:>7.1f}%"
        )

    if not results:
        return 1

    # Rank by exact match, breaking ties toward balanced unknown handling: a
    # model that never says "unknown" scores well on precision and is unusable.
    def unknown_f1(row: dict) -> float:
        precision, recall = row["unknown_precision"], row["unknown_recall"]
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # A validation set this size cannot resolve small differences: at ~100 rows
    # one row is ~1 pp, so a 2 pp gap is noise. Chasing it picked iteration 700
    # over 600 on +1.8 pp exact -- and 700 answered "unknown" to "Open Terminal."
    # while 600 got every core command right. Within the band, prefer the EARLIER
    # checkpoint: same measured quality, less overfitting.
    # BOTH metrics are noisy at this sample size, so both are banded. A
    # checkpoint only loses if it is measurably worse on one of them.
    best_exact = max(row["exact_match"] for row in results)
    contenders = [row for row in results if row["exact_match"] >= best_exact - NOISE_BAND_PP]
    best_f1 = max(unknown_f1(row) for row in contenders)
    contenders = [row for row in contenders if unknown_f1(row) >= best_f1 - NOISE_BAND_PP]
    best = min(contenders, key=lambda row: row["iteration"])
    print(
        f"\nBEST: iteration {best['iteration']} "
        f"(exact {best['exact_match']:.1f}%, unknown F1 {unknown_f1(best):.1f})"
    )
    if len(contenders) > 1:
        tied = ", ".join(str(row["iteration"]) for row in contenders)
        print(f"  tied within {NOISE_BAND_PP:.0f} pp on {len(contenders)} checkpoints ({tied});")
        print("  broke toward the earliest, since that difference is not measurable here.")
    print(
        f"\nTo use it:\n  cp {adapter_dir}/{best['iteration']:07d}_adapters.safetensors "
        f"{adapter_dir}/adapters.safetensors"
    )
    print("\nThen score it ONCE on the test set:")
    print("  uv run python scripts/eval_slm.py --adapter adapters")

    Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/results/checkpoint_selection.json").write_text(
        json.dumps({"results": results, "best_iteration": best["iteration"]}, indent=2)
    )
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
