"""One Conformer encoder block, assembled in macaron order.

    x = x + 0.5 * FFN1(norm(x))          <- half-weighted
    x = x +       MHSA(norm(x))
    x = x +       Conv(norm(x))
    x = x + 0.5 * FFN2(norm(x))          <- half-weighted
    out = norm(x)

Pre-norm throughout (normalise the input to each sublayer, add the raw residual),
which is what makes a 24-layer stack trainable without warmup tricks.

The 0.5 factors are the macaron structure (Gulati et al. 2020, §2.2). They are
easy to omit and produce no error when omitted -- the block still runs, output
still looks like plausible features, and the error compounds silently across all
24 layers. `tests/test_conformer_numerical.py` checks for exactly this.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .attention import RelPositionMultiHeadAttention
from .config import ConformerConfig
from .convolution import ConvolutionModule
from .feedforward import FeedForward

MACARON_RESIDUAL_SCALE = 0.5


class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        self.norm_feed_forward1 = nn.LayerNorm(config.d_model)
        self.feed_forward1 = FeedForward(config.d_model, config.d_ff, config.use_bias)

        self.norm_self_att = nn.LayerNorm(config.d_model)
        self.self_attn = RelPositionMultiHeadAttention(config)

        self.norm_conv = nn.LayerNorm(config.d_model)
        self.conv = ConvolutionModule(config)

        self.norm_feed_forward2 = nn.LayerNorm(config.d_model)
        self.feed_forward2 = FeedForward(config.d_model, config.d_ff, config.use_bias)

        self.norm_out = nn.LayerNorm(config.d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        x = x + MACARON_RESIDUAL_SCALE * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb=pos_emb, mask=mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + MACARON_RESIDUAL_SCALE * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)
