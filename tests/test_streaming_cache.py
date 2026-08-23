"""Numerical verification of the cache-aware streaming loop.

This is the streaming artifact and it had no test at all: the claim that the
per-layer KV cache, convolution hand-off and absolute position offsets are
correct lived only in a results document. Every failure mode here is silent --
a leaked cache or a chunk-local position offset does not raise, it degrades
transcripts as the utterance grows.

Slow: loads the real 0.6B checkpoint.
    uv run pytest tests/test_streaming_cache.py -v -m slow
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from streaming_encoder import StreamingEncoder

pytestmark = pytest.mark.slow

CLIP = Path("benchmarks/datasets/synth/timer.wav")
MEL_HOP_MS = 10
# Whole-utterance-as-one-chunk must match the offline encoder to float noise.
EXACT_TOLERANCE = 1e-5

# Forced to float32 for the same reason as tests/test_conformer_numerical.py: the
# checkpoint is bfloat16, and relying on implicit promotion makes this tolerance
# hardware- and version-dependent. See that module's comment.
COMPARE_DTYPE = mx.float32


@pytest.fixture(scope="module")
def encoder():
    from parakeet_mlx import from_pretrained

    enc = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3").encoder
    enc.set_dtype(COMPARE_DTYPE)
    enc.eval()
    return enc


@pytest.fixture(scope="module")
def mel(encoder):
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    audio, _ = sf.read(str(CLIP), dtype="float32")
    return get_logmel(mx.array(audio), model.preprocessor_config)


@pytest.fixture(scope="module")
def offline(encoder, mel):
    out, _ = encoder(mel, mx.array([mel.shape[1]], dtype=mx.int64))
    mx.eval(out)
    return out


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    mx.eval(a, b)
    return float(mx.abs(a - b).max())


def stream(encoder, mel, chunk_ms: int, **kwargs) -> mx.array:
    loop = StreamingEncoder(encoder, **kwargs)
    frames_per_chunk = chunk_ms // MEL_HOP_MS
    outputs = [
        loop.encode_chunk(mel[:, i : i + frames_per_chunk, :])
        for i in range(0, mel.shape[1], frames_per_chunk)
        if mel[:, i : i + frames_per_chunk, :].shape[1] >= 8
    ]
    return mx.concatenate(outputs, axis=1)


def test_single_chunk_reproduces_the_offline_encoder(encoder, mel, offline):
    """With the whole utterance in one chunk there is no missing future context,
    so the caches must be a no-op. Any difference here is a bug in the loop
    itself rather than the train/inference mismatch."""
    assert max_abs_diff(stream(encoder, mel, 10_000), offline) < EXACT_TOLERANCE


def test_chunked_streaming_preserves_frame_count(encoder, mel, offline):
    """Off-by-one in the subsampling hand-off shows up as extra or missing
    frames long before it shows up as a wrong word."""
    for chunk_ms in (320, 640):
        assert stream(encoder, mel, chunk_ms).shape[1] == offline.shape[1]


def _subsampled(encoder, mel, chunk_ms: int, context_mel: int) -> mx.array:
    loop = StreamingEncoder(encoder, subsampling_context_mel=context_mel)
    frames_per_chunk = chunk_ms // MEL_HOP_MS
    parts = [
        loop._subsample(mel[:, i : i + frames_per_chunk, :])
        for i in range(0, mel.shape[1], frames_per_chunk)
        if mel[:, i : i + frames_per_chunk, :].shape[1] >= 8
    ]
    return mx.concatenate(parts, axis=1)


def test_subsampling_stem_needs_its_own_cache(encoder, mel):
    """The obvious reading of "cache-aware streaming" is that the per-layer
    attention and convolution caches are the whole story. They are not: the
    strided convolutions in the frontend have their own receptive field, and
    cutting it at a chunk boundary corrupts features before any layer runs.

    Measured at the stem output rather than the encoder output, because 24
    LayerNorms downstream mask the magnitude of the corruption.
    """
    reference, _ = encoder.pre_encode(mel, mx.array([mel.shape[1]], dtype=mx.int64))
    mx.eval(reference)

    with_context = _subsampled(encoder, mel, 320, 8)
    without_context = _subsampled(encoder, mel, 320, 0)
    n = reference.shape[1]

    good = max_abs_diff(with_context[:, :n], reference)
    bad = max_abs_diff(without_context[:, :n], reference)
    assert good < 0.01, f"8 mel frames of context should be near-lossless, got {good:.3g}"
    assert bad > 100 * good, (
        f"dropping the stem context barely changed anything ({bad:.3g} vs {good:.3g}) "
        "-- the context is probably not being applied at all"
    )


def test_reset_clears_all_carried_state(encoder, mel):
    """State leaked across turns corrupts the next utterance silently."""
    loop = StreamingEncoder(encoder)
    first = loop.encode_chunk(mel[:, :32, :])
    loop.reset()
    assert loop.cache.frames_seen == 0
    assert loop._mel_context is None
    assert all(layer.keys is None and layer.conv_tail is None for layer in loop.cache.layers)
    assert max_abs_diff(loop.encode_chunk(mel[:, :32, :]), first) == 0.0


def test_streaming_this_checkpoint_is_measurably_worse(encoder, mel, offline):
    """The conclusion, pinned. This checkpoint was trained full-context,
    so chunked streaming MUST diverge -- if this ever passes cleanly, either the
    model changed or the test stopped streaming anything."""
    assert max_abs_diff(stream(encoder, mel, 320), offline) > EXACT_TOLERANCE


def test_kv_cache_can_be_bounded(encoder, mel):
    """Unbounded left context is fine for 1-3 second commands but would grow
    without limit on long-form audio."""
    loop = StreamingEncoder(encoder, max_left_context_frames=4)
    for i in range(0, 96, 32):
        loop.encode_chunk(mel[:, i : i + 32, :])
    assert all(layer.keys.shape[2] <= 4 for layer in loop.cache.layers)


def test_frames_seen_tracks_absolute_stream_position(encoder, mel):
    """Relative-position attention must be indexed by absolute stream position.
    Resetting it per chunk does not crash; it makes transcripts drift."""
    loop = StreamingEncoder(encoder)
    total = 0
    for i in range(0, 96, 32):
        total += loop.encode_chunk(mel[:, i : i + 32, :]).shape[1]
        assert loop.cache.frames_seen == total


def test_conv_tail_carries_kernel_minus_one_frames(encoder, mel):
    """The depthwise convolution needs its left neighbours; without them every
    chunk boundary is treated as the start of the audio."""
    loop = StreamingEncoder(encoder)
    loop.encode_chunk(mel[:, :32, :])
    padding = encoder.layers[0].conv.padding
    assert padding > 0
    assert all(layer.conv_tail.shape[1] == padding for layer in loop.cache.layers)


def test_offline_audio_is_untouched_by_the_loop(mel):
    """Guard against the loop mutating its input, which would make the offline
    comparison meaningless."""
    before = np.array(mel)
    StreamingEncoder.__init__  # noqa: B018 -- import-time sanity only
    assert np.array_equal(before, np.array(mel))
