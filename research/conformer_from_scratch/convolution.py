"""Conformer convolution module.

    pointwise(d -> 2d) -> GLU -> depthwise(k=9) -> BatchNorm -> Swish -> pointwise(d -> d)

This is what gives Conformer its local modelling, complementing attention's
global view (Gulati et al. 2020, §2.1).

Two details that cause silent corruption if missed:

1. BATCHNORM MUST BE IN EVAL MODE. Batch statistics computed over a single
   streaming utterance are meaningless; inference must use the frozen running
   stats. MLX modules default to eval unless `.train()` is called, and
   `assert_eval_mode()` below makes that checkable rather than assumed.

2. PADDING IS SYMMETRIC HERE (non-causal): the module sees `padding` frames on
   both sides. A causal/streaming variant pads (kernel-1) on the LEFT only. The
   checkpoint we load was trained non-causally, so we match that -- see
   ENGINEERING.md §6.

Layout note: MLX Conv1d expects (batch, time, channels), unlike PyTorch's
(batch, channels, time). Weights ported from PyTorch need transposing.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .config import ConformerConfig


class ConvolutionModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config
        self.padding = config.conv_padding

        # Doubles the channels so GLU can halve them back, gating one half by
        # the other.
        self.pointwise_conv1 = nn.Conv1d(
            config.d_model, config.d_model * 2, kernel_size=1, bias=config.use_bias
        )
        # groups=d_model makes this depthwise: each channel is convolved
        # independently, which is what keeps the parameter count linear in d_model.
        self.depthwise_conv = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.conv_kernel_size,
            groups=config.d_model,
            bias=config.use_bias,
        )
        self.batch_norm = nn.BatchNorm(config.d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            config.d_model, config.d_model, kernel_size=1, bias=config.use_bias
        )

    def assert_eval_mode(self) -> None:
        """Fail loudly if BatchNorm is in training mode.

        In training mode BatchNorm would normalise using the statistics of the
        current utterance rather than the frozen running stats, which changes the
        output without raising anything.
        """
        if self.batch_norm.training:
            raise RuntimeError(
                "BatchNorm is in training mode; inference would use batch statistics "
                "instead of the frozen running stats. Call .eval() on the model."
            )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, d_model)
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)  # (batch, time, 2d) -> (batch, time, d)
        x = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.pointwise_conv2(x)
