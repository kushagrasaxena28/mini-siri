"""Prompt templates -- the single source of truth shared by inference and dataset
generation.

Two variants, matched to the two models they run against:

  FEWSHOT_SYSTEM_PROMPT  -- the zero-shot baseline. Needs worked examples because
                            the base model has never seen this schema.
  FINETUNED_SYSTEM_PROMPT -- what the LoRA-tuned model uses. No examples: the
                            mapping is now in the weights, not the prompt. This
                            is also what scripts/build_dataset.py trains against,
                            so training and inference can never drift apart.

The few-shot version is ~500 prompt tokens; the short one ~224. Prefix-KV reuse
absorbs the difference at inference (253 ms vs 252 ms), so the short prompt is
about matching what the fine-tune was trained against, not about speed. If this
file and the dataset generator disagree, the model is trained for a prompt it
never receives.
"""

from __future__ import annotations

SCHEMA_BLOCK = """Schema: {"intent": <intent>, "args": {...}}

intents and args:
- set_timer      {"value": int, "unit": "seconds"|"minutes"|"hours"}
- open_app       {"app_name": str}
- close_app      {"app_name": str}
- open_path      {"path": "downloads"|"documents"|"desktop"|"home"|"applications"}
- set_volume     {"direction": "up"|"down"|"mute"|"unmute"|"set", "level": int 0-100 (only for "set")}
- media_control  {"action": "play"|"pause"|"next"|"previous"}
- capture_note   {"text": str}
- search_notes   {"query": str}
- find_file      {"query": str}
- get_time       {}
- get_status     {"item": "battery"|"wifi"|"bluetooth"|"storage"|"chip"|"device"|"date"|"day"}
- list_items     {"what": "notes"|"files", "where": folder (only for files)}
- open_folder    {"name": str}
- create_reminder {"text": str, "hour": int, "minute": int, "meridiem": "am"|"pm", "day": "today"|"tomorrow"}
- cancel         {}
- unknown        {}

Rules:
- Volume of the SYSTEM is set_volume. Playback control is media_control.
- Convert spoken numbers to integers. For timers, report the number and unit as
  SPOKEN ("three hours" -> value 3, unit "hours"); never multiply it yourself.
- If the command is not one of the intents above, use unknown.
- Speech that is chatter, background talk, or an unfinished sentence is unknown.
- "open X" is open_app; "close/quit/turn off X" is close_app. Never confuse them.
- search_notes looks in the user's NOTES. find_file looks for FILES on disk.
- An incomplete command with the key detail missing is unknown, not a guess.
- get_time is the CLOCK only. The date, the day, or any time arithmetic
  ("what was the time five minutes ago") is get_status or unknown, never get_time.
- open_path is for the standard folders (downloads, documents, ...). Any other
  folder name is open_folder.
- Alarms and reminders are both create_reminder. Report the clock face you heard;
  never convert it yourself.
- If earlier turns are shown, resolve "it", "that" and "the app" against them.
  With nothing to resolve against, the command is unknown."""

FEWSHOT_SYSTEM_PROMPT = f"""You convert a spoken command into JSON. Output ONLY the JSON object.

{SCHEMA_BLOCK}

Examples:
set a timer for ten minutes -> {{"intent":"set_timer","args":{{"value":10,"unit":"minutes"}}}}
give me a 30 second timer -> {{"intent":"set_timer","args":{{"value":30,"unit":"seconds"}}}}
set a timer for three hours -> {{"intent":"set_timer","args":{{"value":3,"unit":"hours"}}}}
open Terminal -> {{"intent":"open_app","args":{{"app_name":"Terminal"}}}}
close WhatsApp -> {{"intent":"close_app","args":{{"app_name":"WhatsApp"}}}}
quit Spotify -> {{"intent":"close_app","args":{{"app_name":"Spotify"}}}}
find my resume pdf -> {{"intent":"find_file","args":{{"query":"resume"}}}}
open my downloads folder -> {{"intent":"open_path","args":{{"path":"downloads"}}}}
turn the volume down -> {{"intent":"set_volume","args":{{"direction":"down"}}}}
mute -> {{"intent":"set_volume","args":{{"direction":"mute"}}}}
set volume to 40 -> {{"intent":"set_volume","args":{{"direction":"set","level":40}}}}
pause the music -> {{"intent":"media_control","args":{{"action":"pause"}}}}
skip this song -> {{"intent":"media_control","args":{{"action":"next"}}}}
note check the cache config tomorrow -> {{"intent":"capture_note","args":{{"text":"check the cache config tomorrow"}}}}
what did I note about the encoder -> {{"intent":"search_notes","args":{{"query":"encoder"}}}}
what time is it -> {{"intent":"get_time","args":{{}}}}
never mind -> {{"intent":"cancel","args":{{}}}}
so anyway I was thinking we should -> {{"intent":"unknown","args":{{}}}}
set a timer for -> {{"intent":"unknown","args":{{}}}}
open -> {{"intent":"unknown","args":{{}}}}"""

FINETUNED_SYSTEM_PROMPT = f"""You convert a spoken command into JSON. Output ONLY the JSON object.

{SCHEMA_BLOCK}"""

# Primes the assistant past Qwen3's reasoning step -- enable_thinking=False is a
# verified no-op on this checkpoint. See slm/parser.py module docstring.
NO_THINK = "<think>\n\n</think>\n\n"

# Opening of the JSON object, appended to the prompt so generation starts INSIDE it.
#
# Required after LoRA training with --mask-prompt. With next-token prediction the
# target for the first completion token is predicted FROM the last prompt
# position -- which masking excludes from the loss. The model therefore never
# learns to emit `{"` and starts at `intent":...`, producing unparseable output
# with otherwise perfect content. (Tokenisation is NOT the cause: `{"` encodes as
# a single clean token and the prompt prefix is preserved under joint encoding --
# both verified.)
#
# Priming also helps the un-finetuned model (it constrains the opening) and saves
# one decode step, worth ~7-8 ms.
JSON_PRIME = '{"'

# Closes the assistant's turn and opens the next user turn, so a conversation can
# be continued in place. Verified byte-identical against the tokenizer's own
# multi-turn rendering -- getting this wrong corrupts every follow-up silently.
TURN_SEPARATOR = "<|im_end|>\n<|im_start|>user\n"

DEFAULT_MODEL = "Qwen/Qwen3-1.7B-MLX-4bit"
