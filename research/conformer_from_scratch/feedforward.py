"""Macaron feed-forward module.

Two of these sandwich the attention and convolution modules, each contributing a
HALF-weighted residual -- the "macaron" structure from Conformer (Gulati et al.
2020, §2.2). Dropping the 0.5 is the classic silent error: the block still runs
and the output still looks like speech features, but every layer is subtly wrong
and it compounds over 24 of them. The scaling lives in the block (block.py), not
here, so this module stays a plain FFN.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


class FeedForward(nn.Module):
    """Linear -> Swish -> Linear. Expansion factor 4 in this configuration."""

    def __init__(self, d_model: int, d_ff: int, use_bias: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()  # Swish; SiLU is the same function
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))
