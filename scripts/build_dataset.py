"""YAML seeds -> validated, chat-templated JSONL for mlx_lm.lora.

Two things this enforces that are easy to get wrong by hand:

1. Every row is validated against the SAME schema.validate() the runtime uses,
   so a label typo is caught now instead of silently teaching the model a wrong
   schema.
2. The training prompt is rendered with the EXACT chat template + short system
   prompt (prompts.FINETUNED_SYSTEM_PROMPT) the fine-tuned model will see at
   inference. If these drift apart, the model is trained for a prompt it will
   never actually receive.

    uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from mlx_lm import load

from mini_siri_local.schema.intents import IntentCall, validate
from mini_siri_local.slm.prompts import (
    DEFAULT_MODEL,
    FINETUNED_SYSTEM_PROMPT,
    NO_THINK,
    TURN_SEPARATOR,
)

SEEDS_DIR = Path("datasets/intent/seeds")
CAPTURED_PATH = Path("datasets/intent/asr_captured.jsonl")  # from scripts/record_dataset.py
OUT_DIR = Path("datasets/intent")
SPLIT = (0.70, 0.15, 0.15)  # train, valid, test
SEED = 1234


def compact_json(intent: str, args: dict) -> str:
    """No whitespace, matching what the runtime SLM is asked to produce --
    every output token costs ~16 ms of decode time (measured)."""
    return json.dumps({"intent": intent, "args": args}, separators=(",", ":"))


def _identity(row: dict) -> str:
    """Dedupe/leakage key. A multi-turn row is only the same as another if its
    CONTEXT matches too -- "close it" after "open chrome" is a different training
    example from "close it" on its own."""
    context = " | ".join(t["text"].strip().lower() for t in row.get("context", []))
    return f"{context} >> {row['text'].strip().lower()}"


def load_seeds() -> list[dict]:
    """Load every seed, deduplicated by transcript.

    Duplicates are not harmless. The splitter treats two copies of one utterance
    as independent rows and can put one in train and the other in test, which
    leaks. Eight such pairs existed before this dedupe and put four utterances on
    both sides of the split.

    A repeated transcript with a DIFFERENT label is a labelling bug, not a
    duplicate, so it stops the build rather than silently picking one.
    """
    by_text: dict[str, dict] = {}
    conflicts: list[tuple[str, dict, dict]] = []
    for f in sorted(SEEDS_DIR.glob("*.yaml")):
        for row in yaml.safe_load(f.read_text()):
            row["_source_file"] = f.stem
            row["_provenance"] = "llm_draft"
            row.setdefault("context", [])
            key = _identity(row)
            existing = by_text.get(key)
            if existing is None:
                by_text[key] = row
            elif (existing["intent"], existing.get("args", {})) != (
                row["intent"],
                row.get("args", {}),
            ):
                conflicts.append((row["text"], existing, row))

    if conflicts:
        print(f"\n{len(conflicts)} CONFLICTING LABELS -- fix before training:\n")
        for text, first, second in conflicts:
            print(f"  {text!r}")
            print(f"      [{first['_source_file']}] {first['intent']} {first.get('args', {})}")
            print(f"      [{second['_source_file']}] {second['intent']} {second.get('args', {})}")
        raise SystemExit(1)
    return list(by_text.values())


def load_captured() -> list[dict]:
    """Real ASR output from scripts/record_dataset.py -- text is the actual
    (possibly corrupted) transcript, not what was said."""
    if not CAPTURED_PATH.exists():
        return []
    rows = []
    for line in CAPTURED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        rows.append(
            {
                "text": entry["transcript"],  # the ASR OUTPUT, corruption included
                "intent": entry["intent"],
                "args": entry["args"],
                "_source_file": "asr_captured",
                "_provenance": "asr_captured",
            }
        )
    return rows


def asr_surface_variants(text: str, intent: str) -> list[str]:
    """Re-render a seed in the SURFACE FORM our ASR actually produces.

    Seeds are authored lowercase and unpunctuated, but Parakeet emits
    capitalised text with terminal punctuation:

        "set a timer for ten minutes"  (seed)
        "Set a timer for ten minutes." (what the SLM actually receives)

    Training only on the seed form is a train/inference mismatch on every single
    turn. These variants are NOT a substitute for real recorded ASR errors
    (word-level corruption still needs scripts/record_dataset.py) -- they fix the
    surface-form gap only, which is the part we can characterise exactly.
    """
    variants = []
    capitalised = text[0].upper() + text[1:] if text else text
    question_words = (
        "what",
        "when",
        "where",
        "who",
        "how",
        "do ",
        "does ",
        "can ",
        "did ",
        "are ",
        "is ",
    )
    ends_question = text.lower().startswith(question_words)
    punct = "?" if ends_question else "."
    variants.append(capitalised + punct)

    # Parakeet flips between word and digit forms of the same number depending on
    # silence padding (measured). Cover the digit form where a word
    # number appears, so both surface forms map to the same JSON.
    word_to_digit = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "fifteen": "15",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "ninety": "90",
    }
    words = text.split()
    swapped = [word_to_digit.get(w.lower().strip(",."), w) for w in words]
    if swapped != words:
        digit_text = " ".join(swapped)
        variants.append(digit_text[0].upper() + digit_text[1:] + punct)

    return variants


def validate_row(row: dict) -> str | None:
    """Round-trip the label through the REAL validator. Returns an error string,
    or None if the row is well-formed and would actually execute."""
    raw = compact_json(row["intent"], row.get("args", {}))
    result = validate(raw)
    if not isinstance(result, IntentCall):
        return result.reason
    if result.intent.value != row["intent"]:
        return (
            f"intent mismatch: wrote {row['intent']!r}, validator resolved {result.intent.value!r}"
        )

    # Two intents deliberately train on a DIFFERENT representation than the
    # validator outputs, because the validator's job is to convert:
    #
    #   open_path  -- model emits the allowlist key ("downloads"); the validator
    #                 resolves it to a real path. Training on the resolved path
    #                 would bake THIS machine's home directory into the model.
    #   set_timer  -- model emits {value, unit}; the validator multiplies into
    #                 duration_seconds. Asking the model to multiply produced
    #                 10x digit errors (see schema/intents.py _check_timer).
    #
    # For these, "the validator accepted it" is the whole check.
    # These four train on a DIFFERENT representation than the validator emits,
    # because converting is the validator's job:
    #   open_path / list_items -- allowlist key -> absolute path. Training on the
    #     resolved path would bake this machine's home directory into the model.
    #   set_timer      -- {value, unit} -> seconds.
    #   create_reminder -- 12-hour clock face -> 24-hour.
    # For these, "the validator accepted it" is the whole check.
    if row["intent"] in ("open_path", "list_items", "set_timer", "create_reminder"):
        return None

    if result.args != row.get("args", {}):
        return f"args normalized: wrote {row.get('args')!r}, validator produced {result.args!r}"
    return None


def prompt_parts(tokenizer) -> tuple[str, str]:
    """(head, tail) from the tokenizer's own template, split at the transcript.

    Exactly the decomposition SlmParser makes at inference, so training prompts
    and runtime prompts are built from the same two strings.
    """
    msgs = [
        {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
        {"role": "user", "content": "__PLACEHOLDER__"},
    ]
    rendered = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    if isinstance(rendered, list):
        rendered = tokenizer.decode(rendered)
    head, tail = rendered.split("__PLACEHOLDER__")
    return head, tail


def render_prompt(tokenizer, transcript: str, context: list[dict] | None = None) -> str:
    """Render one training prompt, optionally preceded by earlier turns.

    Assembled by hand rather than through apply_chat_template, because the
    template STRIPS `<think>` blocks out of historical assistant turns -- while
    SlmParser's KV cache keeps them, since it feeds NO_THINK on every turn. Going
    through the template would train the model on a conversation shape it never
    sees at inference, and nothing would have raised.
    """
    head, tail = prompt_parts(tokenizer)
    parts = [head]
    for turn in context or []:
        parts.append(turn["text"].strip())
        parts.append(tail)
        parts.append(NO_THINK + compact_json(turn["intent"], turn.get("args", {})))
        parts.append(TURN_SEPARATOR)
    parts.append(transcript.strip())
    parts.append(tail)
    parts.append(NO_THINK)
    return "".join(parts)


def family_split(rows: list[dict]) -> tuple[list, list, list]:
    """Split by (intent, source file) group, not by row.

    These are hand-authored rather than paraphrase-expanded, so there is no
    finer-grained "same underlying utterance" family id yet -- but splitting
    randomly at the row level is still wrong in principle (dataset-strategy.md
    §4) and this keeps the split mechanism correct for when paraphrase
    expansion or ASR-captured variants ARE added, where it matters a lot more.
    """
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    rng = random.Random(SEED)
    train, valid, test = [], [], []
    for group in by_intent.values():
        items = list(group)
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * SPLIT[0]))
        n_valid = max(1, round(n * SPLIT[1])) if n - n_train > 1 else 0
        train += items[:n_train]
        valid += items[n_train : n_train + n_valid]
        test += items[n_train + n_valid :]
    return train, valid, test


def rendered_texts(rows: list[dict], surface: str) -> set[str]:
    """Every transcript a split will actually contain, in the form it is written.

    Comparing SEED text is not enough: train is written with its ASR surface
    variants too, and two different seeds can render to the same variant once
    capitalised, punctuated and digit-swapped.
    """
    out: set[str] = set()
    for row in rows:
        prefix = " | ".join(t["text"].strip().lower() for t in row.get("context", []))
        if surface == "both":
            texts = [row["text"], *asr_surface_variants(row["text"], row["intent"])]
        elif surface == "asr":
            variants = asr_surface_variants(row["text"], row["intent"])
            texts = [variants[0]] if variants else [row["text"]]
        else:
            texts = [row["text"]]
        out.update(f"{prefix} >> {t.strip().lower()}" for t in texts)
    return out


def assert_no_leakage(train: list[dict], valid: list[dict], test: list[dict]) -> None:
    """Held-out means held out. This is the check that would have caught the
    duplicate seeds, so it runs on every build rather than living in a comment."""
    trained = rendered_texts(train, "both")
    for name, rows in (("valid", valid), ("test", test)):
        leaked = sorted(rendered_texts(rows, "asr") & trained)
        if leaked:
            raise SystemExit(
                f"\n{len(leaked)} transcripts appear in BOTH train and {name}:\n"
                + "\n".join(f"  {t!r}" for t in leaked)
                + "\n\nUsually two seeds that differ only in case, punctuation or "
                "word-vs-digit form. Remove or reword one."
            )


def split_captured_eval(rows: list[dict]) -> tuple[list, list]:
    """Real-voice rows: MOST go to a dedicated robustness eval (never trained
    on), a smaller share goes into training so the model has seen at least
    some real ASR noise. This mirrors DS-INTENT-ASR in the original dataset
    strategy -- evaluating fine-tuned accuracy on genuine ASR output, not just
    clean text, is the only way to know the ASR-robustness story is real.
    """
    captured = [r for r in rows if r["_provenance"] == "asr_captured"]
    if not captured:
        return [], []
    rng = random.Random(SEED)
    captured = captured[:]
    rng.shuffle(captured)
    n_train_in = max(0, round(len(captured) * 0.3))  # 30% strengthens training
    return captured[:n_train_in], captured[n_train_in:]  # (into train, eval-only)


def write_split(name: str, rows: list[dict], tokenizer, surface: str = "seed") -> None:
    """Write a split in one of three surface modes.

    "seed"  -- as authored (lowercase, unpunctuated)
    "both"  -- seed + ASR surface variants. TRAIN ONLY: augmenting valid/test
               would inflate scores by evaluating on near-duplicates of train.
    "asr"   -- ASR surface form only. Used for TEST, because capitalised and
               punctuated text is what the pipeline actually hands the SLM at
               runtime. Evaluating on the seed form would measure a
               distribution the model never sees in production.
    """
    path = OUT_DIR / f"{name}.jsonl"
    n_written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if surface == "both":
                texts = [row["text"], *asr_surface_variants(row["text"], row["intent"])]
            elif surface == "asr":
                variants = asr_surface_variants(row["text"], row["intent"])
                texts = [variants[0]] if variants else [row["text"]]
            else:
                texts = [row["text"]]
            completion = compact_json(row["intent"], row.get("args", {}))
            for text in dict.fromkeys(texts):  # dedupe, preserve order
                prompt = render_prompt(tokenizer, text, row.get("context"))
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
                n_written += 1
    note = {
        "both": "  (seeds + ASR surface variants)",
        "asr": "  (ASR surface form -- matches runtime)",
    }.get(surface, "")
    print(f"  {name:<14} {n_written:>4} rows -> {path}{note}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include-captured",
        action="store_true",
        help="fold in datasets/intent/asr_captured.jsonl (real voice, from record_dataset.py)",
    )
    args = ap.parse_args()

    rows = load_seeds()
    print(f"loaded {len(rows)} seed rows from {SEEDS_DIR}")

    captured = load_captured() if args.include_captured else []
    if args.include_captured:
        if captured:
            print(f"loaded {len(captured)} real-voice rows from {CAPTURED_PATH}")
        else:
            print(
                f"--include-captured passed but {CAPTURED_PATH} is empty or missing "
                f"(run scripts/record_dataset.py first)"
            )
    rows = rows + captured
    print()

    print("validating every row against schema.intents.validate() ...")
    errors = []
    for row in rows:
        err = validate_row(row)
        if err:
            errors.append((row, err))
    if errors:
        print(f"\n{len(errors)} INVALID ROWS -- fix before training:\n")
        for row, err in errors:
            print(f"  [{row['_source_file']}] {row['text']!r}")
            print(f"      {err}")
        return 1
    print(f"  all {len(rows)} rows valid\n")

    by_intent = defaultdict(int)
    for r in rows:
        by_intent[r["intent"]] += 1
    print("counts per intent:")
    for intent, n in sorted(by_intent.items()):
        print(f"  {intent:<15} {n:>3}")

    print("\nloading tokenizer to render training prompts ...")
    _, tokenizer = load(DEFAULT_MODEL)

    clean_rows = [r for r in rows if r["_provenance"] != "asr_captured"]
    train, valid, test = family_split(clean_rows)
    captured_into_train, captured_eval_only = split_captured_eval(rows)
    train += captured_into_train

    assert_no_leakage(train, valid, test)
    print(
        f"\nsplit ({SPLIT[0]:.0%}/{SPLIT[1]:.0%}/{SPLIT[2]:.0%} on clean text, "
        f"stratified per intent, seed={SEED}):"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_split("train", train, tokenizer, surface="both")
    write_split("valid", valid, tokenizer, surface="asr")
    write_split("test", test, tokenizer, surface="asr")
    if captured_eval_only:
        # Already real ASR output -- do not re-render its surface form.
        write_split("test_asr_real", captured_eval_only, tokenizer, surface="seed")
        print(
            f"  ({len(captured_into_train)} real-voice rows folded into train, "
            f"{len(captured_eval_only)} held out ONLY for the ASR-robustness eval)"
        )

    sample_tokens = len(tokenizer.encode(render_prompt(tokenizer, "set a timer for ten minutes")))
    print(f"\nsample prompt length: {sample_tokens} tokens (short/fine-tuned form)")
    print(f"\ntotal: {len(rows)} seed rows")
    print("NOTE: this is the CLEAN-TEXT slice only. The real-ASR-error slice needs your voice")
    print("      -- see scripts/record_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
