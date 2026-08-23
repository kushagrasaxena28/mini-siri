"""Evaluate an SLM checkpoint on the held-out intent test set.

Reports the metrics separately, because they fail differently and a single
accuracy number hides which:

  schema_valid   -- fraction that parse and validate at all (fail-closed rate)
  intent_acc     -- right action chosen. STRICT: a validator rejection never
                    counts as correct here, even when declining was right
  declined_ok    -- took no action when it should not have. A rejection counts,
                    because no action is no action however it was reached
  args_acc       -- right slots, GIVEN the right intent
  exact_match    -- both, strictest
  unknown_p/r    -- the safety-critical class for an always-listening system

    uv run python scripts/eval_slm.py                          # zero-shot baseline
    uv run python scripts/eval_slm.py --adapter adapters       # LoRA adapter
    uv run python scripts/eval_slm.py --model models/fused     # fused checkpoint
    uv run python scripts/eval_slm.py --test datasets/intent/test_asr_real.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from _datasets import require
from mini_siri_local.config import SlmConfig
from mini_siri_local.schema.intents import IntentCall, validate
from mini_siri_local.slm.parser import SlmParser


def load_test(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def transcripts_from_prompt(prompt: str) -> list[str]:
    """Every user turn in a rendered prompt, in order.

    Multi-turn rows carry the turns that came BEFORE the one being scored, and
    their label depends on them -- "make it louder" is set_volume after a volume
    command and unknown after "what day is it". Taking only the last user
    message scored those rows against context the model was never given, which
    made a correct answer look like an error.
    """
    marker = "<|im_start|>user\n"
    out = []
    for chunk in prompt.split(marker)[1:]:
        end = chunk.find("<|im_end|>")
        out.append(chunk[: end if end != -1 else None].strip())
    return out or [prompt]


def transcript_from_prompt(prompt: str) -> str:
    """The turn being scored -- the last user message."""
    return transcripts_from_prompt(prompt)[-1]


def redact_home(text: str) -> str:
    """Replace this machine's home directory with ~ before anything is written.

    validate() normalises `"downloads"` into an ABSOLUTE path, so a raw dump of
    an error record bakes the runner's home directory -- and username -- into a
    committed artifact. Redact at the boundary rather than scrubbing after.
    """
    return text.replace(str(Path.home()), "~")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model id or local path")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--test", default="datasets/intent/test.jsonl")
    ap.add_argument(
        "--fewshot",
        action="store_true",
        help="use the long few-shot prompt (for zero-shot baselines)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write metrics JSON here instead of the default")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()
    require(args.test)

    rows = load_test(Path(args.test))
    if args.limit:
        rows = rows[: args.limit]

    # No history: every held-out row is an independent classification, and
    # letting turn N-1 leak into turn N would not be an accuracy measurement.
    # History ON so multi-turn rows can be replayed; reset before every row keeps
    # single-turn rows independent.
    overrides = {"fewshot_prompt": args.fewshot, "keep_history": True}
    if args.model:
        overrides["model_id"] = args.model
    overrides["adapter_path"] = args.adapter  # None for the base model
    parser = SlmParser(SlmConfig(**overrides))
    parser.warmup()

    label = args.adapter or args.model or "base"
    prompt_kind = "few-shot" if args.fewshot else "short"
    print(f"model={label}  prompt={prompt_kind}  test={args.test}  n={len(rows)}\n")

    n_valid = n_intent = n_args = n_exact = 0
    latencies: list[float] = []
    errors: list[tuple[str, str, str]] = []
    # unknown confusion counters
    tp = fp = fn = 0
    per_intent: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]

    for row in rows:
        turns = transcripts_from_prompt(row["prompt"])
        transcript = turns[-1]
        gold = json.loads(row["completion"])
        gold_intent = gold["intent"]
        # Compare validated-to-validated. The validator NORMALISES: set_timer
        # {value, unit} becomes {duration_seconds}, open_path "downloads" becomes
        # an absolute path. Comparing its output against the raw label scored
        # every one of those rows as an args failure even when the model was right.
        gold_validated = validate(row["completion"])
        gold_args = (
            gold_validated.args
            if isinstance(gold_validated, IntentCall)
            else gold.get("args", {})
        )

        # Replay any earlier turns so a multi-turn row is scored with the
        # context its label assumes, then time only the turn under test.
        parser.reset_conversation()
        for earlier in turns[:-1]:
            parser.parse(earlier)

        result = parser.parse(transcript)
        latencies.append(result.latency_ms)
        validated = validate(result.raw)

        per_intent[gold_intent][1] += 1

        if not isinstance(validated, IntentCall):
            errors.append(
                (
                    transcript,
                    redact_home(f"REJECTED: {validated.reason}"),
                    redact_home(json.dumps(gold, separators=(",", ":"))),
                )
            )
            # A rejection IS a decline. The user-visible outcome is identical to
            # `unknown` -- no action, spoken decline -- so for the safety metric
            # it counts as correctly declining, not as a miss. It still counts
            # against schema_valid, because the output genuinely did not parse.
            if gold_intent == "unknown":
                tp += 1
            else:
                fp += 1
            continue

        n_valid += 1
        intent_ok = validated.intent.value == gold_intent

        args_ok = validated.args == gold_args

        if intent_ok:
            n_intent += 1
            per_intent[gold_intent][0] += 1
            if args_ok:
                n_args += 1
                n_exact += 1
        if not (intent_ok and args_ok):
            errors.append(
                (
                    transcript,
                    redact_home(
                        json.dumps(
                            {"intent": validated.intent.value, "args": validated.args},
                            separators=(",", ":"),
                        )
                    ),
                    redact_home(json.dumps(gold, separators=(",", ":"))),
                )
            )

        # unknown confusion
        pred_unknown = validated.intent.value == "unknown"
        gold_unknown = gold_intent == "unknown"
        if pred_unknown and gold_unknown:
            tp += 1
        elif pred_unknown and not gold_unknown:
            fp += 1
        elif not pred_unknown and gold_unknown:
            fn += 1

    n = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"  schema_valid  {n_valid}/{n}  {n_valid / n:6.1%}")
    print(f"  intent_acc    {n_intent}/{n}  {n_intent / n:6.1%}")
    print(f"  args_acc      {n_args}/{n_intent}  {n_args / n_intent if n_intent else 0:6.1%}"
          "   (right slots, GIVEN the right intent)")
    print(f"  exact_match   {n_exact}/{n}  {n_exact / n:6.1%}")
    print(f"  declined_prec {prec:6.1%}   (declines only when it should)")
    print(f"  declined_rec  {rec:6.1%}   (catches out-of-scope speech; a")
    print("                          validator rejection counts as a decline)")
    print(
        f"  latency       median {statistics.median(latencies):.0f} ms  "
        f"p90 {sorted(latencies)[int(len(latencies) * 0.9)]:.0f} ms"
    )

    print("\n  per-intent accuracy:")
    for intent in sorted(per_intent):
        correct, total = per_intent[intent]
        flag = "  <--" if total and correct < total else ""
        print(f"    {intent:<15} {correct}/{total}{flag}")

    if errors and args.show_errors:
        print(f"\n  {len(errors)} errors:")
        for transcript, got, want in errors:
            print(f"    {transcript!r}")
            print(f"      got : {got}")
            print(f"      want: {want}")

    tag = (args.adapter or args.model or "base").replace("/", "_")
    out_path = Path(args.out or f"benchmarks/results/eval_{tag}_{Path(args.test).stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": label,
                "prompt": prompt_kind,
                "test": args.test,
                "n": n,
                "schema_valid": n_valid,
                "intent_acc": n_intent,
                "args_acc": n_args,
                "exact_match": n_exact,
                "unknown_precision": prec,
                "unknown_recall": rec,
                "latency_median_ms": statistics.median(latencies),
                "errors": [{"transcript": t, "got": g, "want": w} for t, g, w in errors],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
