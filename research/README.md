# Research

Two experiments that are **not** on the runtime path and are kept because the
result is worth having, not because the code ships. Nothing in
`src/mini_siri_local/` imports either of them; the runtime uses
`parakeet-mlx` directly via `asr/offline.py`.

Both are pinned by tests, so they cannot rot silently:

```bash
uv run pytest tests/test_conformer_numerical.py -v -m slow   # 1
uv run pytest tests/test_streaming_cache.py    -v -m slow    # 2
```

`pyproject.toml` puts `research/` on the pytest path. It is deliberately not part
of the installed wheel.

---

## 1. `conformer_from_scratch/` — the encoder, reimplemented

**What was built from scratch.** A complete Conformer encoder block in MLX:
relative-position multi-head attention (`attention.py`), the depthwise
convolution module (`convolution.py`), the macaron feed-forward pair
(`feedforward.py`), and the block that composes them with the correct residual
scaling and normalisation order (`block.py`).

**What is third-party.** The *weights*. The block is loaded with the pretrained
tensors from `mlx-community/parakeet-tdt-0.6b-v3` (NVIDIA Parakeet TDT 0.6B) and
compared element-wise against that checkpoint's own encoder layer.

**Why third-party weights.** Training a 0.6B speech encoder is not feasible on
one laptop, and it is not the question being asked. The question is whether the
architecture is understood exactly — and borrowing the weights is what makes that
question answerable. A from-scratch module with from-scratch weights proves
nothing, because there is nothing to disagree with.

**What it demonstrates.** Writing a Conformer-*shaped* module is a tutorial
exercise. Making it a numerically exact drop-in for a pretrained checkpoint is
not: it forces every detail to be right — macaron residual scale of ½,
pre-norm ordering, relative-position index arithmetic, BatchNorm held in eval
mode, and the exact pretrained weight layout.

**Measured** (`benchmarks/datasets/synth/timer.wav` fixtures, float32):

| check | max abs diff vs pretrained |
|---|---|
| feed-forward 1, attention, convolution, feed-forward 2 | each < 1e-4 |
| **full block** | **1.907e-05** |
| across sequence lengths 8 / 16 / 64 / 128 | 1.9e-05 – 3.4e-05 |

Exact equality is not expected and not asserted: the attention is written as a
four-term expansion where the reference groups terms differently, so the
difference is float accumulation order, at noise level.

**Status: kept, not shipped.** The runtime calls `parakeet-mlx` because it is
maintained, decoded end-to-end, and faster to keep current. This block exists as
evidence and as the base the streaming loop below was built on.

---

## 2. `streaming_encoder/` — cache-aware streaming, and why it is not used

The project does **not** stream speech recognition. This is the experiment that
established why, and the negative result is the reason the directory is kept.

**What was attempted.** Run the Conformer encoder incrementally over chunks of
audio, carrying per-layer attention and convolution state forward so each chunk
is processed as though the whole utterance were available, minus the future.

**What worked — the implementation is correct.** Feeding an entire utterance
through the streaming loop as a single chunk reproduces the offline encoder to
**5.513e-07**. The per-layer KV caches, the convolution hand-off and the absolute
position offsets are all right.

**The non-obvious finding: the subsampling stem needs its own cache.** The
strided convolutions in the frontend have a receptive field of their own, and
cutting it at a chunk boundary corrupts features before any encoder layer runs.
Measured at the stem output, 320 ms chunks versus whole-utterance:

| mel frames of left context | max abs diff | mean abs diff |
|---|---|---|
| 0 | **2759.60** | 28.53 |
| 4 | 9260.95 | 215.78 |
| **8 (80 ms)** | **0.0049** | 0.00006 |
| 16 | 0.0049 | 0.00006 |
| 32 | 0.0049 | 0.00006 |

Roughly 560,000x error reduction from 80 ms of overlap, and nothing beyond 8
frames — which pins the stem's receptive field at ≤ 8 mel frames. Four frames is
*worse* than zero: enough to shift the alignment, not enough to cover the field.

**What did not work — and it is the model, not the code.** With the loop proven
correct, chunked streaming still collapses on this checkpoint. Transcripts,
offline versus streaming at 320 ms chunks with the same TDT decoder:

| clip | offline | streaming |
|---|---|---|
| timer | "Set a timer for 10 minutes." | "ten minutes." |
| open_app | "Open terminal." | "Open." |
| volume | "Turn the volume down." | "Mm." |
| time | "What time is it?" | "Voy a time." |

**0/8 identical** at 320 ms and at 640 ms. Not degradation — collapse.

**The control that rules out an implementation bug.** `parakeet-mlx`'s own
`transcribe_stream()` on the same checkpoint and clips returns **0/6 identical**,
most of them empty. Two independent implementations fail the same way, so the
cause is upstream of both.

**The blocker.** `att_context_size = [-1, -1]`, read from the checkpoint's own
config. Parakeet TDT v3 was trained seeing the entire utterance with
bidirectional attention. Chunked streaming forces each frame to attend only to
its past — a condition it has never seen — and 1–3 second commands do not supply
enough left context to compensate.

**Why the blocker could not be resolved.** It is a property of the trained
weights, not a parameter. No amount of cache correctness recovers context the
model was never trained to do without. A knob that withheld the tail of each
chunk until the next arrived was tried and removed: the frames were already
computed without the future, so delaying their release delayed output without
improving it (identical error at lookahead 0, 1 and 2). Recovering future context
requires **re-encoding**, which costs real compute.

**The decision, pre-registered before measuring:**

| measured | action |
|---|---|
| within ~2% of offline | ship streaming |
| 2–10% worse | add right-context lookahead, re-measure |
| **>10% worse** | **fall back to offline ASR** |

0/8 is far past the third row, so the pre-committed fallback was taken.

**What it cost the product: nothing.** Recognition is already hidden inside the
endpoint wait (see `ENGINEERING.md` §4) — it runs during the silence the
endpointer spends confirming the turn is over, so it contributes ~0 ms to the
measured turn. Streaming would have optimised a stage that no longer appears in
the latency budget.

**What should happen if the blocker is removed.** Port a checkpoint *trained*
with limited context — `stt_en_fastconformer_hybrid_large_streaming_multi` is the
obvious candidate — to MLX. No MLX port exists today; that port, not this loop,
is the real work. If such a checkpoint lands, this loop is the harness to
evaluate it with, and `tests/test_streaming_cache.py` already asserts the
divergence: if chunked streaming ever stops diverging, either the model changed
or the test stopped streaming.
