"""Silero VAD (ONNX). Fixed 512-sample frames at 16 kHz -- not configurable.

TWO pieces of state must be carried between frames and reset between turns:

  1. LSTM state (2, 1, 128)
  2. A 64-sample CONTEXT window of the previous frame's tail

The model input is (1, 576) = 64 context + 512 chunk. Feeding a bare 512 samples
does NOT error -- it returns near-zero probability on loud speech. Measured on a
clip peaking at -0.9 dBFS: max prob 0.524 / mean 0.018 without context, versus
max 1.000 / mean 0.996 with it. Silent corruption, so it is asserted in tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import SAMPLE_RATE_HZ, VAD_FRAME_SAMPLES, SileroConfig

# The model input is 64 context samples + the 512-sample chunk (576 total).
CONTEXT_SAMPLES = 64
LSTM_STATE_SHAPE = (2, 1, 128)


class SileroVad:
    def __init__(self, config: SileroConfig | None = None, model_path: Path | str | None = None):
        self.config = config or SileroConfig()
        path = Path(model_path) if model_path else self.config.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"silero model missing at {path}\nRun:  uv run python scripts/download_models.py"
            )
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1  # one 32 ms frame; threads only add overhead
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sample_rate = np.array(SAMPLE_RATE_HZ, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear LSTM state AND the context window. Call at every turn boundary.

        Leaking either across turns biases the next utterance -- the classic
        symptom is the first frames of a new turn being misclassified.
        """
        self.state = np.zeros(LSTM_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame."""
        if frame.shape[0] != VAD_FRAME_SAMPLES:
            raise ValueError(
                f"silero needs exactly {VAD_FRAME_SAMPLES} samples, got {frame.shape[0]}"
            )
        frame = frame.astype(np.float32, copy=False)
        # Prepending the previous frame's tail is REQUIRED. Feeding a bare 512
        # does not error -- it silently returns ~0 on loud speech (measured:
        # max prob 0.524 without context vs 1.000 with it).
        model_input = np.concatenate([self.context, frame]).reshape(1, -1)
        output, self.state = self.session.run(
            None, {"input": model_input, "state": self.state, "sr": self._sample_rate}
        )
        self.context = frame[-CONTEXT_SAMPLES:].copy()
        return float(output[0][0])
