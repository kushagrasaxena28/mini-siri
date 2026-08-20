"""Correctness of the SLM's KV-cache reuse.

Two silent failure modes are possible here, and both would be invisible in
normal use until latency or accuracy quietly degraded:

  1. The cache is not trimmed back after a turn, so it grows unboundedly and
     every turn gets slower than the last.
  2. Tokenising the transcript SEPARATELY from the cached system prompt yields
     different tokens than tokenising the whole prompt at once, so the cached
     path produces different output than the uncached path.

These tests load a real model, so they are slow. Run with:
    uv run pytest tests/test_slm_cache.py -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

def _held_out_transcripts() -> list[str]:
    import json
    from pathlib import Path

    marker = "<|im_start|>user\n"
    out = []
    for line in Path("datasets/intent/test.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        prompt = json.loads(line)["prompt"]
        start = prompt.rfind(marker) + len(marker)
        out.append(prompt[start : prompt.find("<|im_end|>", start)].strip())
    return out


TRANSCRIPTS = [
    "Set a timer for ten minutes.",
    "Open terminal.",
    "What time is it?",
    "Turn the volume down.",
]


# The SHIPPED configuration: LoRA adapter, short prompt. Testing the base model
# here would guard a checkpoint nobody runs.
SHIPPED = {"adapter_path": "adapters", "fewshot_prompt": False, "keep_history": False}


@pytest.fixture(scope="module")
def cached_parser():
    from mini_siri_local.config import SlmConfig
    from mini_siri_local.slm.parser import SlmParser

    parser = SlmParser(SlmConfig(**SHIPPED, reuse_prompt_cache=True))
    parser.warmup()
    return parser


@pytest.fixture(scope="module")
def uncached_parser():
    from mini_siri_local.config import SlmConfig
    from mini_siri_local.slm.parser import SlmParser

    return SlmParser(SlmConfig(**SHIPPED, reuse_prompt_cache=False))


def test_cache_does_not_grow_across_turns(cached_parser):
    """Offset must return to the prefix length after every parse.

    If trim_prompt_cache silently failed, this would climb turn over turn and
    latency would degrade without any error.
    """
    prefix_len = cached_parser._prefix_len
    for transcript in TRANSCRIPTS * 3:
        cached_parser.parse(transcript)
        assert cached_parser._cache[0].offset == prefix_len, (
            f"cache grew to {cached_parser._cache[0].offset} (prefix is {prefix_len}) -- "
            "trim is not working, latency will degrade every turn"
        )


def test_cached_and_uncached_paths_mostly_agree(cached_parser, uncached_parser):
    """Reusing the prefix KV cache is NOT bit-identical to a fresh prefill.

    Prefilling the head alone and prefilling head+transcript together are the
    same computation in different shapes, and in float32 on a quantised model
    they land a few ulps apart -- enough to flip a borderline classification.
    Measured on the held-out set: 78/79 agree; `"No stop."` splits cancel vs
    unknown (the cached path is the one that matches the label).

    So this is a rate, not an identity. It is asserted because the 400 ms/turn
    the cache saves is only worth having if the disagreement stays rare, and the
    only way to know is to keep measuring it.
    """
    transcripts = _held_out_transcripts()
    disagreements = [
        t for t in transcripts if cached_parser.parse(t).raw != uncached_parser.parse(t).raw
    ]
    rate = len(disagreements) / len(transcripts)
    assert rate <= 0.05, f"cached and uncached diverged on {len(disagreements)}: {disagreements}"


def test_decode_matches_mlx_lm_generate(uncached_parser):
    """Our greedy loop replaced mlx_lm.generate for ~125 ms/turn. That is only a
    valid substitution if it produces the SAME tokens.

    Both sides get a fresh cache, so this measures the DECODER and nothing else;
    prompt-cache reuse is a separate effect with its own test below.

    argmax(logits) and argmax(logits - logsumexp(logits)) are the same ranking
    but not the same floating-point operation, so an exact tie could in
    principle break either way. On the shipped checkpoint all 79 held-out rows
    agree; the base model with the few-shot prompt splits one.
    """
    from mlx_lm import generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    from mini_siri_local.slm.prompts import JSON_PRIME

    sampler = make_sampler(temp=0.0)
    for transcript in _held_out_transcripts():
        reference = generate(
            uncached_parser.model,
            uncached_parser.tokenizer,
            prompt=uncached_parser._prompt_for(transcript),
            max_tokens=uncached_parser.max_tokens,
            sampler=sampler,
            prompt_cache=make_prompt_cache(uncached_parser.model),
            verbose=False,
        ).strip()
        if not reference.startswith("{"):
            reference = JSON_PRIME + reference
        assert uncached_parser.parse(transcript).raw == reference, (
            f"decode diverged from mlx_lm.generate on {transcript!r}"
        )


def test_output_is_stable_over_many_turns(cached_parser):
    """The reused prefix must not decay. Drift here would show up as accuracy
    quietly falling during a long session, with nothing to point at."""
    import mlx.core as mx

    transcripts = _held_out_transcripts()
    prefix_before = mx.array(cached_parser._cache[0].keys[:, :, : cached_parser._prefix_len, :])
    mx.eval(prefix_before)

    first = [cached_parser.parse(t).raw for t in transcripts]
    second = [cached_parser.parse(t).raw for t in transcripts]

    prefix_after = mx.array(cached_parser._cache[0].keys[:, :, : cached_parser._prefix_len, :])
    mx.eval(prefix_after)
    assert float(mx.abs(prefix_after - prefix_before).max()) == 0.0, "cached prefix drifted"
    assert first == second


def test_repeated_parses_are_deterministic(cached_parser):
    """Greedy decoding plus a correctly restored cache means identical input
    yields identical output. Divergence indicates cache contamination."""
    first = cached_parser.parse("Set a timer for ten minutes.").raw
    for _ in range(4):
        assert cached_parser.parse("Set a timer for ten minutes.").raw == first
