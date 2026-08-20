"""Conformer encoder geometry.

Values match `mlx-community/parakeet-tdt-0.6b-v3` as loaded from its own config,
so a from-scratch block can be checked against the pretrained weights.

Two of these are the structural facts the whole project is built around:

    subsampling_factor = 8, with a 10 ms mel hop  ->  80 ms per encoder frame
    att_context_size   = (-1, -1)                 ->  FULL context, not streaming

The second is why this checkpoint is not a cache-aware streaming model: it was
trained seeing the entire utterance. See ENGINEERING.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConformerConfig:
    feat_in: int = 128  # mel bins
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 8
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    use_bias: bool = False
    pos_emb_max_len: int = 5000

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ff_expansion_factor

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def conv_padding(self) -> int:
        """Symmetric padding that keeps sequence length unchanged.

        Non-causal: the module sees `conv_padding` frames on BOTH sides. A
        streaming/causal variant pads (kernel-1) on the left only -- that
        difference is the whole reason streaming needs its own training.
        """
        return (self.conv_kernel_size - 1) // 2

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if (self.conv_kernel_size - 1) % 2:
            raise ValueError(f"conv_kernel_size {self.conv_kernel_size} must be odd")
