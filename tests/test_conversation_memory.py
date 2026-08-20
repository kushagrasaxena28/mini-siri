"""Conversation history, carried in the KV cache.

Roughly a quarter of a real session is the user referring back -- "close it",
"the app", "make it louder". With no history the model invented a target:
"Close the app for me." produced app_name "MyApp" and tried to quit it.

History lives in the KV cache rather than being re-prefilled each turn, so it
costs memory but almost no latency. That makes two things load-bearing and
silent if wrong:

  * the prompt the trainer builds must be byte-identical to the one the cache
    reconstructs, or the model is trained for a shape it never sees;
  * declined turns must be dropped, or an always-listening mic fills the
    context with overheard speech.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

CONTEXT = [{"text": "open apple music for me", "intent": "open_app", "args": {"app_name": "Music"}}]


@pytest.fixture(scope="module")
def parser():
    from mini_siri_local.config import SlmConfig
    from mini_siri_local.slm.parser import SlmParser

    p = SlmParser(SlmConfig(keep_history=True))
    p.warmup()
    p.reset_conversation()
    return p


def test_training_prompt_matches_what_the_cache_reconstructs(parser):
    """The mismatch this catches is invisible: apply_chat_template STRIPS
    <think> blocks out of historical assistant turns, while the cache keeps
    them, so going through the template would train on a different prompt."""
    from build_dataset import render_prompt
    from mini_siri_local.slm.prompts import JSON_PRIME, NO_THINK, TURN_SEPARATOR

    training = render_prompt(parser.tokenizer, "close the app for me", CONTEXT)
    inference = (
        parser._head
        + CONTEXT[0]["text"]
        + parser._tail
        + NO_THINK
        + JSON_PRIME
        + 'intent":"open_app","args":{"app_name":"Music"}}'
        + TURN_SEPARATOR
        + "close the app for me"
        + parser._tail
        + NO_THINK
    )
    assert training == inference


def test_a_pronoun_resolves_against_the_previous_turn(parser):
    from mini_siri_local.schema.intents import IntentCall, validate

    parser.reset_conversation()
    parser.parse("Open Google Chrome.")
    result = validate(parser.parse("Close it.").raw)

    assert isinstance(result, IntentCall)
    assert result.args.get("app_name") == "Google Chrome"


def test_history_accumulates_and_resets(parser):
    parser.reset_conversation()
    assert parser.history_tokens == 0

    parser.parse("Open Google Chrome.")
    after_one = parser.history_tokens
    assert after_one > 0

    parser.parse("Close it.")
    assert parser.history_tokens > after_one

    parser.reset_conversation()
    assert parser.history_tokens == 0


def test_declined_turns_are_dropped_from_history(parser):
    """An always-listening mic hears a lot it correctly ignores. Keeping that
    in context would make the next real command answer to background noise."""
    parser.reset_conversation()
    parser.parse("Open Google Chrome.")
    before = parser.history_tokens

    parser.parse("So anyway I was thinking about dinner.")
    parser.discard_last_turn()

    assert parser.history_tokens == before


def test_discarding_leaves_earlier_turns_resolvable(parser):
    """Dropping the overheard turn must not damage the history before it."""
    from mini_siri_local.schema.intents import IntentCall, validate

    parser.reset_conversation()
    parser.parse("Open Google Chrome.")
    parser.parse("the weather is nice today isn't it")
    parser.discard_last_turn()

    result = validate(parser.parse("Close it.").raw)
    assert isinstance(result, IntentCall)
    assert result.args.get("app_name") == "Google Chrome"


def test_history_is_off_when_not_requested():
    """eval and the regression probe need independent turns; history there
    would make an accuracy number meaningless."""
    from mini_siri_local.config import SlmConfig
    from mini_siri_local.slm.parser import SlmParser

    p = SlmParser(SlmConfig(keep_history=False))
    p.parse("Open Google Chrome.")
    assert p.history_tokens == 0
