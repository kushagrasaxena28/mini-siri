"""Transcript -> JSON intent, via a small local model.

The model's ONLY job is: noisy transcript in, one JSON object out. No planning,
no tool chaining. Code orchestrates; the model classifies and extracts.

Previous turns are carried so "close it" resolves, but they live in the KV cache
rather than being re-prefilled -- see keep_history and discard_last_turn.

Qwen3 note: `enable_thinking=False` is a NO-OP on Qwen3-1.7B-MLX-4bit (verified:
the rendered prompt is byte-identical with or without it). Left unhandled, every
call emits <think>... and blows the latency budget 5-15x. The working mechanism is
to prime the assistant turn with an empty, pre-closed think block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from ..config import SlmConfig
from .prompts import (
    FEWSHOT_SYSTEM_PROMPT,
    FINETUNED_SYSTEM_PROMPT,
    JSON_PRIME,
    NO_THINK,
    TURN_SEPARATOR,
)

# History lives in the KV cache, so it costs no prefill -- but it does cost
# memory, and an always-listening process runs for days. At the cap the
# conversation resets rather than growing without bound.
MAX_HISTORY_TOKENS = 3000

# A degenerate generation repeats one token forever. Observed: "Play." produced
# {"action":"playpausepausepausepause... , burning the whole 160-token budget
# (~2.5 s) to emit unparseable JSON. Real completions never repeat a token this
# often, so stopping early costs nothing and caps the damage at a fast decline.
MAX_TOKEN_REPEATS = 8


@dataclass(frozen=True)
class SlmResult:
    raw: str
    latency_ms: float


class SlmParser:
    """Parses transcripts to JSON, reusing the system prompt's KV cache.

    The system prompt is ~500 tokens and never changes; the transcript is ~10.
    Re-prefilling the whole thing every turn costs ~790 ms, almost all of it
    prefill. Prefilling once at startup and reusing the cache removes that from
    the per-turn path.
    """

    def __init__(self, config: SlmConfig | None = None):
        """Everything tunable lives in SlmConfig, so the constructor cannot drift
        away from it -- the two used to mirror each other by hand."""
        self.config = config or SlmConfig()
        self.model_id = self.config.model_id
        self.max_tokens = self.config.max_tokens
        self.use_cache = self.config.reuse_prompt_cache
        self.keep_history = self.config.keep_history and self.use_cache
        self.adapter_path = self.config.adapter_path
        self.system_prompt = (
            FEWSHOT_SYSTEM_PROMPT if self.config.fewshot_prompt else FINETUNED_SYSTEM_PROMPT
        )
        load_kwargs = {"adapter_path": self.adapter_path} if self.adapter_path else {}
        self.model, self.tokenizer = load(self.model_id, **load_kwargs)
        self._eos_ids = set(self.tokenizer.eos_token_ids)

        rendered = self._render_template()
        # Split at the transcript: everything before it is fixed and cacheable.
        self._head, self._tail = rendered.split("__PLACEHOLDER__")
        self._cache = None
        self._prefix_len = 0
        self._turn_start = 0
        if self.use_cache:
            self._prefill()

    def _render_template(self) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "__PLACEHOLDER__"},
        ]
        rendered = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        if isinstance(rendered, list):
            rendered = self.tokenizer.decode(rendered)
        return rendered

    def _prefill(self) -> None:
        """Run the fixed head through the model once and keep its KV cache."""
        self._cache = make_prompt_cache(self.model)
        head_ids = mx.array(self.tokenizer.encode(self._head))[None]
        self.model(head_ids, cache=self._cache)
        mx.eval([c.state for c in self._cache])
        self._prefix_len = head_ids.shape[1]
        self._turn_start = self._prefix_len

    def _trim_to(self, offset: int) -> None:
        excess = self._cache[0].offset - offset
        if excess > 0:
            trim_prompt_cache(self._cache, excess)

    @property
    def history_tokens(self) -> int:
        """Conversation currently carried in the cache, beyond the system prompt."""
        return 0 if self._cache is None else self._cache[0].offset - self._prefix_len

    def reset_conversation(self) -> None:
        """Forget every previous turn. The system prompt stays prefilled."""
        if self._cache is not None:
            self._trim_to(self._prefix_len)

    def discard_last_turn(self) -> None:
        """Drop the turn just parsed from the conversation.

        Called for anything that produced no action. An always-listening mic
        hears a lot of speech it correctly declines, and keeping that in context
        would poison the next real command -- history should be what the user
        actually asked the assistant to do, not everything it overheard.
        """
        if self.keep_history and self._cache is not None:
            self._trim_to(self._turn_start)

    def _prompt_for(self, transcript: str) -> str:
        """Full prompt (no cache) -- used when use_cache=False, and for token counts."""
        return self._head + transcript.strip() + self._tail + NO_THINK + JSON_PRIME

    def _decode(self, prompt: str, cache: list) -> str:
        """Greedy decode against `cache`, which it extends in place.

        Deliberately not `mlx_lm.generate`: that wrapper costs ~125 ms per turn
        on this model -- a full 152k-vocab logsumexp plus sampler machinery on
        every step, for what temperature 0 makes an argmax -- and produces
        byte-identical output here (checked over the whole held-out test set by
        tests/test_slm_cache.py).
        """
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        logits = self.model(tokens, cache=cache)[:, -1, :]

        generated: list[int] = []
        repeats = 0
        for _ in range(self.max_tokens):
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
            value = token.item()
            if value in self._eos_ids:
                break

            repeats = repeats + 1 if generated and value == generated[-1] else 0
            if repeats >= MAX_TOKEN_REPEATS:
                break  # degenerate loop; let validation reject what we have

            generated.append(value)
            logits = self.model(token[None], cache=cache)[:, -1, :]
        return self.tokenizer.decode(generated)

    def warmup(self) -> float:
        t0 = time.monotonic()
        self.parse("what time is it")
        return (time.monotonic() - t0) * 1000

    def parse(self, transcript: str) -> SlmResult:
        started = time.monotonic()
        if self.use_cache:
            self._turn_start = self._cache[0].offset
            # Continuing a conversation: close the previous assistant turn and
            # open a new user one. On the first turn the head already ends
            # mid-user-message, so nothing is prepended.
            opener = TURN_SEPARATOR if self._turn_start > self._prefix_len else ""
            suffix = opener + transcript.strip() + self._tail + NO_THINK + JSON_PRIME
            text = self._decode(suffix, self._cache)

            if not self.keep_history:
                # Roll back to the fixed head so the next turn starts clean.
                self._trim_to(self._prefix_len)
            elif self.history_tokens > MAX_HISTORY_TOKENS:
                self.reset_conversation()
        else:
            text = self._decode(self._prompt_for(transcript), make_prompt_cache(self.model))

        # Generation started INSIDE the object, so restore the opening we primed.
        raw = text.strip()
        if not raw.startswith("{"):
            raw = JSON_PRIME + raw
        return SlmResult(raw, (time.monotonic() - started) * 1000)
