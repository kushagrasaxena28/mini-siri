"""Relative-position multi-head self-attention (Transformer-XL style).

Conformer uses RELATIVE rather than absolute positions (Gulati et al. 2020,
§2.1, following Dai et al. 2019). The attention score between query i and key j
decomposes into four terms:

    (a) content x content     q . k
    (b) content x position    q . pos
    (c) global content bias   u . k
    (d) global position bias  v . pos

Terms (a)+(c) and (b)+(d) are computed by biasing the query with `pos_bias_u`
and `pos_bias_v`, then two matmuls -- the standard formulation.

Why this matters for the rest of the project: because positions are relative,
the SAME weights work at any sequence length, which is what makes a streaming
window possible at all. But the position embedding must be indexed by ABSOLUTE
stream position, not chunk-local position. Getting that wrong is the classic
off-by-one that produces transcripts which degrade over a long utterance
instead of failing outright.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import ConformerConfig


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        bias = config.use_bias
        self.linear_q = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.linear_k = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.linear_v = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.linear_out = nn.Linear(config.d_model, config.d_model, bias=bias)
        # Projects the sinusoidal position embedding; never biased.
        self.linear_pos = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learned global biases -- the u and v of terms (c) and (d).
        self.pos_bias_u = mx.zeros((self.n_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((self.n_heads, self.head_dim))

    @staticmethod
    def rel_shift(x: mx.array) -> mx.array:
        """Turn absolute position offsets into relative ones.

        The position matmul produces scores indexed by absolute offset; this
        pad-reshape-slice trick re-indexes them so entry (i, j) means "j relative
        to i". Purely an indexing manoeuvre -- no arithmetic on the values.
        """
        batch, heads, n_queries, n_positions = x.shape
        x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)])
        x = x.reshape(batch, heads, n_positions + 1, n_queries)
        x = x[:, :, 1:, :]
        return x.reshape(batch, heads, n_queries, n_positions)

    def _split_heads(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape

        q = self._split_heads(self.linear_q(x))  # (B, H, T, D)
        k = self._split_heads(self.linear_k(x))
        v = self._split_heads(self.linear_v(x))
        p = self._split_heads(self.linear_pos(pos_emb))

        # Bias the query two different ways instead of computing four separate
        # score matrices -- algebraically identical, half the matmuls.
        q_with_u = (q.transpose(0, 2, 1, 3) + self.pos_bias_u).transpose(0, 2, 1, 3)
        q_with_v = (q.transpose(0, 2, 1, 3) + self.pos_bias_v).transpose(0, 2, 1, 3)

        matrix_ac = q_with_u @ k.transpose(0, 1, 3, 2)  # content terms (a)+(c)
        matrix_bd = q_with_v @ p.transpose(0, 1, 3, 2)  # position terms (b)+(d)
        matrix_bd = self.rel_shift(matrix_bd)[:, :, :, :length]

        scores = (matrix_ac + matrix_bd) * self.scale
        if mask is not None:
            scores = mx.where(mask, mx.array(-mx.inf, scores.dtype), scores)

        weights = mx.softmax(scores, axis=-1, precise=True)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.linear_out(out)
