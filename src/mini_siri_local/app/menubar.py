"""Menu-bar app: a visible listening indicator and a real mute switch.

An always-listening microphone with no visible state is a trust problem. This
gives the user two things at all times: what the assistant is currently doing,
and a one-click way to stop it hearing anything.

THREADING
---------
macOS demands the main thread for its UI run loop, and the audio callback has
hard timing requirements. So the pipeline runs on a worker thread and publishes
state through a lock-protected snapshot; the UI polls that snapshot on a timer
and never blocks the pipeline. Nothing in the audio path waits on the UI.

MUTE
----
Mute closes the input device, it does not merely discard frames. A flag that let
the microphone keep streaming would leave macOS showing the recording indicator,
which is precisely what the control exists to disprove.
"""

from __future__ import annotations

import threading

import rumps

from ..config import SlmConfig, TtsConfig, VadConfig
from ..executor import macos
from ..pipeline.assistant import Assistant

# Emoji rather than icon files: no assets to ship, and legible in both light and
# dark menu bars.
ICON_STARTING = "🎙️"
ICON_LISTENING = "🎙️"
ICON_SPEECH = "🗣️"
ICON_THINKING = "⚙️"
ICON_MUTED = "🔇"
ICON_ERROR = "⚠️"

UI_POLL_SECONDS = 0.25


class AssistantState:
    """Thread-safe snapshot of what the pipeline is doing.

    The pipeline writes; the UI reads. A lock rather than a queue because the UI
    only ever wants the LATEST state, never the history.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "starting"
        self._last_transcript = ""
        self._last_action = ""
        self._turns = 0

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def record_turn(self, transcript: str, action: str) -> None:
        with self._lock:
            self._last_transcript = transcript
            self._last_action = action
            self._turns += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "transcript": self._last_transcript,
                "action": self._last_action,
                "turns": self._turns,
            }


class MenuBarApp(rumps.App):
    # Parameters mirror the CLI flags one-for-one; bundling them into a config object
    # would only move the same count behind a wrapper.
    def __init__(  # noqa: PLR0913
        self,
        vad_config: VadConfig | None = None,
        slm_config: SlmConfig | None = None,
        tts_config: TtsConfig | None = None,
        *,
        device: str | None = None,
        enable_tts: bool = True,
        notify: bool = True,
    ) -> None:
        super().__init__("mini-siri-local", icon=None, title=ICON_STARTING, quit_button=None)
        self.notify = notify

        self.vad_config = vad_config or VadConfig()
        self.slm_config = slm_config or SlmConfig()
        self.tts_config = tts_config or TtsConfig()
        self.device = device
        self.enable_tts = enable_tts

        self.state = AssistantState()
        self.assistant: Assistant | None = None
        self._muted = threading.Event()
        self._worker: threading.Thread | None = None

        self.status_item = rumps.MenuItem("Starting…")
        self.transcript_item = rumps.MenuItem("—")
        self.mute_item = rumps.MenuItem("Mute", callback=self.toggle_mute)
        self.menu = [
            self.status_item,
            self.transcript_item,
            None,
            self.mute_item,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self._start_worker()

    # -- pipeline worker ---------------------------------------------------

    def _start_worker(self) -> None:
        self._worker = threading.Thread(target=self._run_pipeline, daemon=True, name="pipeline")
        self._worker.start()

    def _run_pipeline(self) -> None:
        """Load models and run the listen loop. Never touches the UI directly."""
        try:
            self.assistant = Assistant.build(
                self.vad_config, self.slm_config, self.tts_config, enable_tts=self.enable_tts
            )
            self.assistant.on_status = self.state.set_status
            self.assistant.on_turn = self._on_turn
            self.state.set_status("listening")
            self.assistant.run(device=self.device, replay=None, is_muted=self._muted.is_set)
        except Exception as exc:
            self.state.set_status(f"error: {type(exc).__name__}")

    def _on_turn(self, transcript: str, action: str) -> None:
        """Record the turn, and banner it if something actually happened.

        Declines are deliberately silent: the microphone is always on, so notifying on
        every overheard sentence would make the assistant unusable.
        """
        self.state.record_turn(transcript, action)
        if self.notify and action != "declined":
            macos.notify(f"{action.replace('_', ' ')}", transcript)

    # -- UI ----------------------------------------------------------------

    @rumps.timer(UI_POLL_SECONDS)
    def refresh(self, _timer) -> None:
        """Poll the snapshot. Runs on the main thread; must stay cheap."""
        if self._muted.is_set():
            self.title = ICON_MUTED
            self.status_item.title = "Muted — not listening"
            return

        snapshot = self.state.snapshot()
        status = snapshot["status"]

        self.title = {
            "starting": ICON_STARTING,
            "listening": ICON_LISTENING,
            "speech": ICON_SPEECH,
            "thinking": ICON_THINKING,
        }.get(status, ICON_ERROR if status.startswith("error") else ICON_LISTENING)

        self.status_item.title = {
            "starting": "Loading models…",
            "listening": f"Listening · {snapshot['turns']} turns",
            "speech": "Hearing you…",
            "thinking": "Working…",
        }.get(status, status)

        transcript = snapshot["transcript"]
        self.transcript_item.title = (
            f"“{transcript[:40]}” → {snapshot['action']}" if transcript else "—"
        )

    def toggle_mute(self, sender) -> None:
        if self._muted.is_set():
            self._muted.clear()
            sender.title = "Mute"
            self.state.set_status("listening")
        else:
            self._muted.set()
            sender.title = "Unmute"

    def quit_app(self, _sender) -> None:
        if self.assistant is not None:
            self.assistant.close()
        rumps.quit_application()


def main(  # noqa: PLR0913 -- mirrors the CLI flags; see MenuBarApp.__init__
    vad_config: VadConfig | None = None,
    slm_config: SlmConfig | None = None,
    tts_config: TtsConfig | None = None,
    *,
    device: str | None = None,
    enable_tts: bool = True,
    notify: bool = True,
) -> int:
    MenuBarApp(
        vad_config, slm_config, tts_config, device=device, enable_tts=enable_tts, notify=notify
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
