"""Cache-aware streaming encoder loop.

THE ARTIFACT. Runs the Conformer encoder incrementally over chunks of audio,
carrying per-layer attention and convolution state forward so each chunk is
processed as though the whole utterance were available -- except for the future.

    offline:    encode(all frames)              once, sees past AND future
    streaming:  encode(chunk) -> encode(chunk)  each sees all past, limited future

WHAT DIFFERS FROM OFFLINE, AND WHY
----------------------------------
With unbounded left context (see cache.py) the past is identical to offline. The
only difference is that a frame near the end of a chunk has not yet seen the
frames after it, while offline it would have.

There is no lookahead knob. Withholding the tail of a chunk until the next one
arrives was tried and does nothing -- see encode_chunk. Recovering future
context requires RE-ENCODING, which this loop deliberately does not do.

This checkpoint was trained full-context (att_context_size [-1,-1], read from its
own config), so some mismatch is unavoidable -- it was never trained to cope with
a missing future. The job here is to MEASURE that cost, not to pretend it is zero.
See ENGINEERING.md section 6.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .cache import EncoderCache

# Mel frames of left context the subsampling stem needs. Measured empirically:
# 8 frames (80 ms) drops the boundary error from 2759 to 0.005; more adds nothing.
SUBSAMPLING_CONTEXT_MEL_FRAMES = 8


class StreamingEncoder:
    """Drives a Conformer encoder incrementally, one chunk at a time.

    `encoder` is any module exposing `.layers` of Conformer blocks -- the
    pretrained one, or the from-scratch stack in research/conformer_from_scratch/.
    """

    def __init__(
        self,
        encoder,
        *,
        max_left_context_frames: int | None = None,
        subsampling_factor: int = 8,
        subsampling_context_mel: int = SUBSAMPLING_CONTEXT_MEL_FRAMES,
    ):
        self.encoder = encoder
        self.max_left_context_frames = max_left_context_frames
        self.subsampling_factor = subsampling_factor
        self.subsampling_context_mel = subsampling_context_mel

        self.cache = EncoderCache(len(encoder.layers))
        # Trailing mel frames kept so the NEXT chunk's subsampling has left context.
        self._mel_context: mx.array | None = None

    def reset(self) -> None:
        """Clear all state. Call at every turn boundary."""
        self.cache.reset()
        self._mel_context = None

    def _subsample(self, mel_chunk: mx.array) -> mx.array:
        """Subsample with left context carried from the previous chunk.

        THE SUBSAMPLING STEM NEEDS ITS OWN CACHE. This is easy to miss -- the
        obvious reading of "cache-aware streaming" is that attention and conv
        caches inside the layers are the whole story. They are not: the strided
        convolutions in the frontend have their own receptive field, and cutting
        it at a chunk boundary corrupts the features before the layers ever see
        them.

        Measured on a 320 ms chunk, subsampled features vs whole-utterance:

            0 mel frames of context   max diff 2759.6    mean 27.5
            8 mel frames of context   max diff    0.0049 mean  0.00006

        A ~560,000x reduction from 80 ms of overlap. Beyond 8 frames there is no
        further gain, which pins the receptive field at <= 8 mel frames.
        """
        if self._mel_context is not None:
            context_frames = self._mel_context.shape[1]
            mel_chunk = mx.concatenate([self._mel_context, mel_chunk], axis=1)
        else:
            context_frames = 0

        # Keep this chunk's tail for the next call before encoding. The explicit
        # zero case matters: x[:, -0:, :] is the WHOLE array, not an empty one.
        keep = self.subsampling_context_mel
        self._mel_context = mel_chunk[:, -keep:, :] if keep else None

        lengths = mx.array([mel_chunk.shape[1]], dtype=mx.int64)
        features, _ = self.encoder.pre_encode(mel_chunk, lengths)

        # Drop the output frames that correspond to the prepended context.
        drop = context_frames // self.subsampling_factor
        return features[:, drop:, :] if drop else features

    def _run_layers(self, mel_chunk: mx.array) -> mx.array:
        """Subsample, add positional encoding, then run the cached layer stack.

        Uses the model's OWN `pre_encode` and `pos_enc` rather than
        reimplementing them, so any difference from offline comes from the cache
        logic under test and not from a divergent frontend.

        `pos_enc` takes an `offset` -- this is where absolute stream position
        enters. Passing 0 every chunk would restart positions at each boundary,
        which does not crash but makes transcripts drift as the utterance grows.
        """
        features = self._subsample(mel_chunk)

        pos_emb = None
        if getattr(self.encoder, "pos_enc", None) is not None:
            features, pos_emb = self.encoder.pos_enc(features, offset=self.cache.frames_seen)

        x = features
        for index, layer in enumerate(self.encoder.layers):
            x = self._run_one_layer(layer, x, pos_emb, self.cache[index])
        return x

    def _run_one_layer(self, layer, x: mx.array, pos_emb: mx.array, cache) -> mx.array:
        """Macaron block with cached attention and convolution.

        Mirrors the verified block in conformer_from_scratch/block.py; the only
        additions are the two cache hand-offs.
        """
        x = x + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(x))

        normed = layer.norm_self_att(x)
        x = x + self._cached_attention(layer.self_attn, normed, pos_emb, cache)

        x = x + self._cached_convolution(layer.conv, layer.norm_conv(x), cache)
        x = x + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(x))
        return layer.norm_out(x)

    def _cached_attention(self, attn, x: mx.array, pos_emb: mx.array, cache) -> mx.array:
        """Attention over (cached history + this chunk)."""
        batch, length, _ = x.shape
        # parakeet-mlx names these `n_head`; conformer_from_scratch/ uses `n_heads`.
        # Accept either so the loop drives the pretrained encoder AND ours.
        n_heads = getattr(attn, "n_heads", None) or attn.n_head
        head_dim = attn.head_dim

        def split(tensor):
            return tensor.reshape(batch, -1, n_heads, head_dim).transpose(0, 2, 1, 3)

        q = split(attn.linear_q(x))
        k_new = split(attn.linear_k(x))
        v_new = split(attn.linear_v(x))

        # The cache hand-off: this chunk attends over everything seen so far.
        k, v = cache.extend_attention(k_new, v_new)
        if self.max_left_context_frames is not None:
            cache.trim_attention_to(self.max_left_context_frames)
            k, v = cache.keys, cache.values

        p = split(attn.linear_pos(pos_emb))

        q_u = (q.transpose(0, 2, 1, 3) + attn.pos_bias_u).transpose(0, 2, 1, 3)
        q_v = (q.transpose(0, 2, 1, 3) + attn.pos_bias_v).transpose(0, 2, 1, 3)

        matrix_ac = q_u @ k.transpose(0, 1, 3, 2)
        matrix_bd = attn.rel_shift(q_v @ p.transpose(0, 1, 3, 2))[:, :, :, : k.shape[2]]

        scores = (matrix_ac + matrix_bd) * attn.scale
        weights = mx.softmax(scores, axis=-1, precise=True)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return attn.linear_out(out)

    def _cached_convolution(self, conv, x: mx.array, cache) -> mx.array:
        """Convolution with the previous chunk's tail prepended."""
        x = conv.pointwise_conv1(x)
        x = nn.glu(x, axis=2)

        padding = conv.padding
        x = cache.pad_convolution(x, padding)
        x = mx.pad(x, ((0, 0), (0, padding), (0, 0)))

        x = conv.depthwise_conv(x)
        x = conv.batch_norm(x)
        x = conv.activation(x)
        return conv.pointwise_conv2(x)

    def encode_chunk(self, mel_chunk: mx.array) -> mx.array:
        """Encode one chunk and return its encoder frames.

        NOTE ON RIGHT CONTEXT: an earlier version tried to gain future context by
        WITHHOLDING the tail of each chunk and releasing it once the next chunk
        arrived. That does nothing -- the frames were already computed without
        the future, so delaying their release delays output without improving it.
        Measured: identical error at right_context 0, 1 and 2.

        Recovering future context genuinely requires RE-ENCODING those frames
        once the lookahead audio exists, which costs compute and latency. Given
        the measured transcript impact (see stage-05-results.md) that trade was
        not worth taking here, so the loop runs with zero lookahead.
        """
        features = self._run_layers(mel_chunk)
        self.cache.advance(features.shape[1])
        return features
