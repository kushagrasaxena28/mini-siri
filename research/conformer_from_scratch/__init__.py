"""A Conformer encoder block written from scratch in MLX.

Built to be a numerically verified drop-in for the pretrained
`mlx-community/parakeet-tdt-0.6b-v3` encoder: same architecture, same weights,
outputs checked element-wise against the reference implementation. Writing a
Conformer-shaped module is a tutorial exercise; matching the pretrained weights
to 1e-4 is evidence that every detail is right.

See tests/test_conformer_numerical.py for the verification.
"""

from .attention import RelPositionMultiHeadAttention
from .block import ConformerBlock
from .config import ConformerConfig
from .convolution import ConvolutionModule
from .feedforward import FeedForward

__all__ = [
    "ConformerBlock",
    "ConformerConfig",
    "ConvolutionModule",
    "FeedForward",
    "RelPositionMultiHeadAttention",
]
