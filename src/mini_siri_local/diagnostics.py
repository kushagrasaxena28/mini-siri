"""Environment diagnostics -- `mini-siri-local --check`.

Checks the things that fail SILENTLY on macOS, in the order they bite:

  1. Python version   -- misaki (Kokoro's G2P) caps the project at 3.12
  2. Packages         -- a missing optional extra degrades TTS without erroring
  3. Metal            -- MLX falling back off the GPU is a 10x slowdown, not a crash
  4. Models present   -- a missing model is a confusing crash at first use
  5. Audio devices    -- virtual devices (Teams, Zoom) record digital silence
  6. Microphone       -- macOS denies CLI processes access WITHOUT prompting

Every failure prints the fix, not just the symptom.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import MODELS_DIR, SAMPLE_RATE_HZ, AudioConfig

# Substrings identifying virtual/loopback drivers rather than real microphones.
VIRTUAL_DEVICE_HINTS = ("teams", "zoom", "blackhole", "soundflower", "loopback", "virtual")

REQUIRED_MODELS = {
    "models--mlx-community--parakeet-tdt-0.6b-v3": "speech recognition",
    "models--Qwen--Qwen3-1.7B-MLX-4bit": "intent understanding",
    "models--mlx-community--Kokoro-82M-bf16": "speech synthesis",
}

REQUIRED_PACKAGES = ("mlx", "mlx_lm", "parakeet_mlx", "sounddevice", "numpy", "onnxruntime")
# Only needed for spoken confirmations; the pipeline runs without them.
OPTIONAL_PACKAGES = ("mlx_audio", "misaki")

SILENCE_THRESHOLD_DBFS = -60.0
QUIET_THRESHOLD_DBFS = -35.0
RECORD_SECONDS = 1.5


class Report:
    """Collects pass/warn/fail results and prints remediation."""

    def __init__(self) -> None:
        self.failed = False
        self.warned = False

    def ok(self, label: str, detail: str = "") -> None:
        print(f"[  ok  ] {label}" + (f"  --  {detail}" if detail else ""))

    def warn(self, label: str, detail: str = "", fix: str = "") -> None:
        self.warned = True
        print(f"[ warn ] {label}" + (f"  --  {detail}" if detail else ""))
        self._print_fix(fix)

    def fail(self, label: str, detail: str = "", fix: str = "") -> None:
        self.failed = True
        print(f"[ FAIL ] {label}" + (f"  --  {detail}" if detail else ""))
        self._print_fix(fix)

    @staticmethod
    def _print_fix(fix: str) -> None:
        for line in fix.strip().splitlines():
            if line.strip():
                print(f"         {line.strip()}")


def check_python(report: Report) -> None:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) == (3, 12):
        report.ok("python", text)
    else:
        report.fail(
            "python",
            f"{text} (need 3.12.x)",
            "misaki (Kokoro's grapheme-to-phoneme) requires <3.13.\nRun:  uv sync --extra tts",
        )


def check_packages(report: Report) -> None:
    import importlib  # noqa: PLC0415

    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            report.fail(name, str(exc)[:70], "Run:  uv sync")
        else:
            report.ok(name, getattr(module, "__version__", "present"))

    for name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            report.warn(f"{name} (tts)", "missing", "Run:  uv sync --extra tts")
        else:
            report.ok(f"{name} (tts)", "present")


def check_metal(report: Report) -> None:
    """MLX silently falls back off the GPU rather than failing, and the
    difference is ~10x. A timed matmul is the cheapest way to see it."""
    import time  # noqa: PLC0415

    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError as exc:
        report.fail("mlx", str(exc)[:70], "Run:  uv sync")
        return

    started = time.monotonic_ns()
    square = mx.random.normal((1024, 1024))
    mx.eval(square @ square)  # MLX is lazy: without eval this times graph building
    elapsed_ms = (time.monotonic_ns() - started) / 1e6

    info = mx.device_info()
    working_set_gb = info.get("max_recommended_working_set_size", 0) / 1e9
    report.ok("metal", f"{info.get('device_name', '?')}, working set {working_set_gb:.1f} GB")
    report.ok("matmul 1024x1024", f"{elapsed_ms:.1f} ms")


# espeak-ng stores its data directory in a fixed 160-char buffer (N_PATH_HOME). A longer
# path is silently rejected and it falls back to the location compiled into the wheel --
# a GitHub Actions build path that does not exist -- then exits inside native code before
# Python can catch anything. Only reachable from a deeply nested checkout, but the failure
# is fatal and the error message names someone else's CI machine, so flag it here.
ESPEAK_PATH_LIMIT = 150


def check_espeak_data_path(report: Report) -> None:
    try:
        import espeakng_loader  # noqa: PLC0415 -- only present with the tts extra
    except ImportError:
        report.warn("espeak-ng data", "not installed", "uv sync --extra tts")
        return

    path = str(espeakng_loader.get_data_path())
    if len(path) > ESPEAK_PATH_LIMIT:
        report.fail(
            "espeak-ng data path",
            f"{len(path)} chars, over the {ESPEAK_PATH_LIMIT} espeak-ng accepts",
            "Speech synthesis will abort with an error naming a path under /Users/runner.\n"
            "Move the checkout somewhere shorter, e.g. ~/mini-siri-local, and re-run setup.sh.\n"
            "Or install without synthesis:  ./setup.sh --lite",
        )
    else:
        report.ok("espeak-ng data path", f"{len(path)} chars")


def check_models(report: Report) -> None:
    hub = Path.home() / ".cache/huggingface/hub"
    for directory, purpose in REQUIRED_MODELS.items():
        path = hub / directory
        if path.exists():
            report.ok(directory.split("--")[-1], purpose)
        else:
            report.fail(
                directory.split("--")[-1],
                "missing",
                "Run:  uv run python scripts/download_models.py",
            )

    vad_model = MODELS_DIR / "silero_vad.onnx"
    if vad_model.exists():
        report.ok("silero_vad.onnx", "voice activity detection")
    else:
        report.fail("silero_vad.onnx", "missing", "Run:  uv run python scripts/download_models.py")


def check_audio(report: Report, *, test_microphone: bool = True) -> None:
    import numpy as np  # noqa: PLC0415 -- optional at import time
    import sounddevice as sd  # noqa: PLC0415

    from .audio.devices import select_input  # noqa: PLC0415

    inputs = [
        (index, device)
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]
    if not inputs:
        report.fail("input devices", "none found", "Check System Settings > Sound > Input")
        return

    for index, device in inputs:
        is_virtual = any(hint in device["name"].lower() for hint in VIRTUAL_DEVICE_HINTS)
        detail = "virtual device -- would record silence" if is_virtual else "available"
        (report.warn if is_virtual else report.ok)(f"[{index}] {device['name']}", detail)

    device_index, device_name = select_input(AudioConfig())
    report.ok("selected input", f"{device_name} (chosen by name, not index)")

    if not test_microphone:
        return

    try:
        print(f"         recording {RECORD_SECONDS}s from '{device_name}' -- please speak")
        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE_HZ),
            samplerate=SAMPLE_RATE_HZ,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception as exc:
        report.fail(
            "microphone",
            str(exc)[:70],
            "Grant microphone access to your terminal:\n"
            "System Settings > Privacy & Security > Microphone",
        )
        return

    peak = float(np.max(np.abs(recording)))
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else -120.0

    if peak_dbfs < SILENCE_THRESHOLD_DBFS:
        report.fail(
            "microphone",
            f"peak {peak_dbfs:.1f} dBFS (silence)",
            "macOS is denying microphone access, or this is a virtual device.\n"
            "1. System Settings > Privacy & Security > Microphone -> enable your terminal\n"
            "2. Run from Terminal.app or iTerm, NOT an editor's embedded shell",
        )
    elif peak_dbfs < QUIET_THRESHOLD_DBFS:
        report.warn(
            "microphone",
            f"peak {peak_dbfs:.1f} dBFS (quiet)",
            "Works, but low. Speech usually peaks between -6 and -15 dBFS.\n"
            "Raise input gain in System Settings > Sound > Input.",
        )
    else:
        report.ok("microphone", f"peak {peak_dbfs:.1f} dBFS")


def check_disk(report: Report) -> None:
    free_gb = shutil.disk_usage("/").free / 1e9
    report.ok("free disk", f"{free_gb:.0f} GB") if free_gb > 5 else report.warn(
        "free disk", f"{free_gb:.0f} GB (models need ~4 GB)"
    )


def run_diagnostics(*, test_microphone: bool = True) -> int:
    print("mini-siri-local -- environment check\n")
    report = Report()

    print("Runtime")
    check_python(report)
    print("\nPackages")
    check_packages(report)
    print("\nMetal")
    check_metal(report)
    print("\nModels")
    check_models(report)
    check_espeak_data_path(report)
    print("\nAudio")
    check_audio(report, test_microphone=test_microphone)
    print("\nDisk")
    check_disk(report)

    print()
    if report.failed:
        print("RESULT: FAILED -- fix the items above, then re-run `mini-siri-local --check`")
        return 1
    suffix = " (with warnings)" if report.warned else ""
    print(f"RESULT: OK{suffix} -- run `mini-siri-local`")
    return 0
