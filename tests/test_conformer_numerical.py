"""Numerical verification of the from-scratch Conformer block.

Writing a Conformer-SHAPED module is a tutorial exercise. Proving it is a
numerically exact drop-in for the pretrained encoder is what shows every detail
is right -- residual scaling, normalisation order, relative-position indexing,
BatchNorm mode, and the weight layout.

Slow: loads the real 0.6B checkpoint. Run with:
    uv run pytest tests/test_conformer_numerical.py -v -m slow
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from conformer_from_scratch import ConformerBlock, ConformerConfig
from conformer_from_scratch.block import MACARON_RESIDUAL_SCALE

pytestmark = pytest.mark.slow

# Attention accumulates in a different order than the reference (four-term
# formulation vs the reference's grouping), so exact equality is not expected --
# only float-noise-level agreement.
TOLERANCE = 1e-4
SEQ_LEN = 32

# BOTH blocks are forced to float32 before comparing. The checkpoint ships in bfloat16,
# which carries ~3 significant digits: run the comparison in bf16 and the difference is
# ~7.5e-01, not ~2e-05. Passing float32 inputs to bf16 weights makes MLX promote
# implicitly, and that promotion varies by MLX version and by GPU -- which is how this
# test passed on an M1 Pro and failed on an M5 with a diff of 8e-02.
#
# Being explicit makes the test measure ALGORITHMIC equivalence, which is what it claims,
# rather than how bf16 rounding happens to land on the machine running it. Do not loosen
# TOLERANCE to accommodate bf16: that hides the very thing this test exists to catch.
COMPARE_DTYPE = mx.float32


@pytest.fixture(scope="module")
def reference_block():
    from parakeet_mlx import from_pretrained

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    block = model.encoder.layers[0]
    block.set_dtype(COMPARE_DTYPE)
    block.eval()
    return block


@pytest.fixture(scope="module")
def our_block(reference_block):
    """Our implementation, loaded with the PRETRAINED weights."""
    block = ConformerBlock(ConformerConfig())
    block.update(tree_unflatten(list(tree_flatten(reference_block.parameters()))))
    block.set_dtype(COMPARE_DTYPE)
    block.eval()
    return block


@pytest.fixture(scope="module")
def inputs():
    mx.random.seed(0)
    config = ConformerConfig()
    x = mx.random.normal((1, SEQ_LEN, config.d_model)).astype(mx.float32)
    # Relative positions span -(T-1) .. +(T-1), hence 2T-1 embeddings.
    pos = mx.random.normal((1, 2 * SEQ_LEN - 1, config.d_model)).astype(mx.float32)
    return x, pos


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


# --- structural equivalence --------------------------------------------------


def test_parameter_names_and_shapes_match_pretrained(our_block, reference_block):
    """A name or shape mismatch means the weights would not load -- or worse,
    would load into the wrong tensor."""
    ours = {k: v.shape for k, v in tree_flatten(our_block.parameters())}
    ref = {k: v.shape for k, v in tree_flatten(reference_block.parameters())}
    assert set(ours) == set(ref), f"name mismatch: {set(ours) ^ set(ref)}"
    assert ours == ref, "shape mismatch on identically-named tensors"


# --- per-submodule verification ----------------------------------------------


def test_feedforward1_matches(our_block, reference_block, inputs):
    x, _ = inputs
    ours = our_block.feed_forward1(our_block.norm_feed_forward1(x))
    ref = reference_block.feed_forward1(reference_block.norm_feed_forward1(x))
    assert max_abs_diff(ours, ref) < TOLERANCE


def test_attention_matches(our_block, reference_block, inputs):
    """The subtlest module: relative-position indexing and the four-term
    score decomposition both have to be right."""
    x, pos = inputs
    normed = our_block.norm_self_att(x)
    ours = our_block.self_attn(normed, pos_emb=pos)
    ref = reference_block.self_attn(normed, normed, normed, pos_emb=pos)
    assert max_abs_diff(ours, ref) < TOLERANCE


def test_convolution_matches(our_block, reference_block, inputs):
    x, _ = inputs
    ours = our_block.conv(our_block.norm_conv(x))
    ref = reference_block.conv(reference_block.norm_conv(x))
    assert max_abs_diff(ours, ref) < TOLERANCE


def test_feedforward2_matches(our_block, reference_block, inputs):
    x, _ = inputs
    ours = our_block.feed_forward2(our_block.norm_feed_forward2(x))
    ref = reference_block.feed_forward2(reference_block.norm_feed_forward2(x))
    assert max_abs_diff(ours, ref) < TOLERANCE


def test_full_block_matches(our_block, reference_block, inputs):
    """The headline assertion: our block is a drop-in for the pretrained one."""
    x, pos = inputs
    ours = our_block(x, pos_emb=pos)
    ref = reference_block(x, pos_emb=pos)
    assert max_abs_diff(ours, ref) < TOLERANCE


def test_block_matches_across_sequence_lengths(our_block, reference_block):
    """Relative-position attention should be length-agnostic. If the position
    indexing were subtly wrong, longer sequences would diverge more."""
    config = ConformerConfig()
    mx.random.seed(1)
    for length in (8, 16, 64, 128):
        x = mx.random.normal((1, length, config.d_model)).astype(mx.float32)
        pos = mx.random.normal((1, 2 * length - 1, config.d_model)).astype(mx.float32)
        diff = max_abs_diff(our_block(x, pos_emb=pos), reference_block(x, pos_emb=pos))
        assert diff < TOLERANCE, f"diverged at length {length}: {diff:.2e}"


# --- the errors that do not raise --------------------------------------------


def test_macaron_scale_is_one_half():
    """Omitting the 0.5 residual scaling produces no error -- the block runs and
    the output still looks like features -- but every layer is wrong and it
    compounds across all 24."""
    assert MACARON_RESIDUAL_SCALE == 0.5


def test_batchnorm_eval_mode_is_enforced(our_block):
    """Batch statistics over a single utterance are meaningless; inference must
    use the frozen running stats. Training mode would change the output silently."""
    our_block.conv.assert_eval_mode()

    our_block.conv.train()
    with pytest.raises(RuntimeError, match="training mode"):
        our_block.conv.assert_eval_mode()
    our_block.conv.eval()


def test_convolution_preserves_sequence_length(our_block, inputs):
    """Symmetric padding keeps length unchanged. A causal variant would pad only
    on the left -- that difference is why streaming needs its own training."""
    x, _ = inputs
    out = our_block.conv(our_block.norm_conv(x))
    assert out.shape == x.shape
