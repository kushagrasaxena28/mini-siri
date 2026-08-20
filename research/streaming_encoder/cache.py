"""Per-layer state carried between streaming chunks.

This is the heart of cache-aware streaming. Each Conformer layer needs two
pieces of history to process a new chunk as if it had seen the whole utterance:

  KV cache   -- past keys/values, so attention can look backwards without
                recomputing the entire history every chunk. This is the
                difference between O(n) and O(n^2) work over an utterance.
  conv cache -- the last (kernel-1) frames, because the depthwise convolution
                needs its left neighbours. Without it, every chunk boundary
                would be treated as the start of the audio.

WHY THIS PROJECT KEEPS UNBOUNDED LEFT CONTEXT
---------------------------------------------
NVIDIA's cache-aware configs bound the left context too, because they stream
hour-long audio and cannot grow state forever. Our utterances are 1-3 seconds --
roughly 25-40 encoder frames at 80 ms each -- so keeping ALL past frames costs
almost nothing and matches how this checkpoint was trained (full context).

That leaves exactly ONE source of train/inference mismatch: missing FUTURE
context. One variable is measurable; two tangled together are not. See
`right_context_frames` in encoder.py for the knob that trades latency against
recovering that future context.
"""

from __future__ import annotations

import mlx.core as mx


class LayerCache:
    """KV + convolution history for one Conformer layer."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.conv_tail: mx.array | None = None

    def extend_attention(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Append this chunk's keys/values and return the full history.

        Shapes are (batch, heads, time, head_dim). Concatenating on the time
        axis is what lets a chunk attend over everything that came before it.
        """
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values

    def pad_convolution(self, x: mx.array, padding: int) -> mx.array:
        """Prepend the previous chunk's tail so the conv sees its left neighbours.

        On the first chunk there is no history, so we zero-pad -- which is
        correct, since the utterance genuinely starts there.

        Note this pads only on the LEFT from cache; the right side is zero-padded
        within the chunk. That is what makes the convolution causal across chunk
        boundaries while staying symmetric inside a chunk.
        """
        if padding == 0:
            return x

        batch, _, channels = x.shape
        if self.conv_tail is None:
            left = mx.zeros((batch, padding, channels), x.dtype)
        else:
            left = self.conv_tail

        # Keep this chunk's tail for the NEXT chunk before returning.
        self.conv_tail = x[:, -padding:, :] if x.shape[1] >= padding else x
        return mx.concatenate([left, x], axis=1)

    def trim_attention_to(self, max_frames: int) -> None:
        """Bound the KV cache. Only needed for very long sessions.

        Unused for short commands (see module docstring) but required if this is
        ever pointed at long-form audio, where an unbounded cache would grow
        without limit.
        """
        if self.keys is not None and self.keys.shape[2] > max_frames:
            self.keys = self.keys[:, :, -max_frames:, :]
            self.values = self.values[:, :, -max_frames:, :]

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.conv_tail = None


class EncoderCache:
    """One LayerCache per encoder layer, plus the absolute stream position.

    `frames_seen` is load-bearing: relative-position attention must be indexed
    by ABSOLUTE position in the stream, not position within the current chunk.
    Getting that wrong is the classic off-by-one -- it does not crash, it makes
    transcripts drift as the utterance gets longer.
    """

    def __init__(self, n_layers: int) -> None:
        self.layers = [LayerCache() for _ in range(n_layers)]
        self.frames_seen = 0

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> LayerCache:
        return self.layers[index]

    def advance(self, n_frames: int) -> None:
        self.frames_seen += n_frames

    def reset(self) -> None:
        """Clear everything. MUST be called at every turn boundary -- leaked
        state from the previous utterance corrupts the next one silently."""
        for layer in self.layers:
            layer.reset()
        self.frames_seen = 0
