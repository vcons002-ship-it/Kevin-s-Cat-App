"""The background detection loop, started/stopped by the web GUI.

Reads the saved config, watches the chosen camera for a person, and on each
permitted detection rolls the die and casts the sound on a treat. Runs in a
daemon thread so the Flask GUI stays responsive; exposes live status (running
state, last roll, last treat) back to the GUI.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from . import config as config_mod
from . import dice
from .caster import Caster, SoundServer
from .detector import PersonDetector

log = logging.getLogger("d20app.loop")


@dataclass
class Status:
    running: bool = False
    last_error: str = ""
    last_roll: str = ""          # human-readable, e.g. "rolled 18/d20 vs DC18 -> TREAT!"
    last_roll_at: float | None = None
    treats: int = 0
    rolls: int = 0


class DetectionLoop:
    """Owns the worker thread and shared state for one detection session."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.status = Status()
        self._sound_server: SoundServer | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> bool:
        """Start the loop from the current saved config. No-op if running."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self.status = Status(running=True)
            self._thread = threading.Thread(
                target=self._run, name="detection-loop", daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> bool:
        with self._lock:
            if not (self._thread and self._thread.is_alive()):
                self.status.running = False
                return False
            self._stop.set()
        self._thread.join(timeout=10)
        self.status.running = False
        return True

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- the worker ----------------------------------------------------------
    def _run(self) -> None:
        cfg = config_mod.load()
        if not cfg.camera_url:
            self.status.last_error = "No camera selected — choose one in the GUI."
            self.status.running = False
            return
        if not cfg.speaker_name:
            self.status.last_error = "No speaker selected — choose one in the GUI."
            self.status.running = False
            return

        if self._sound_server is None:
            self._sound_server = SoundServer(port=cfg.file_server_port)
        caster = Caster(self._sound_server)

        detector = PersonDetector(
            source=_camera_source(cfg),
            confidence=cfg.person_confidence,
            roi=cfg.roi,
        )
        gate = dice.RollGate(cfg.cooldown_seconds)

        log.info("Detection loop started (camera=%s, speaker=%s)",
                 cfg.camera_name or cfg.camera_url, cfg.speaker_name)
        try:
            self._loop_body(cfg, detector, gate, caster)
        except Exception as exc:  # keep the GUI informed rather than dying silently
            log.exception("Detection loop crashed")
            self.status.last_error = str(exc)
        finally:
            detector.release()
            self.status.running = False
            log.info("Detection loop stopped")

    def _loop_body(self, cfg, detector, gate, caster) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                person = detector.read_and_detect()
            except FileNotFoundError as exc:
                self.status.last_error = str(exc)
                return
            except Exception as exc:
                self.status.last_error = f"camera error: {exc}"
                time.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)
                continue
            backoff = 1.0

            if not person:
                time.sleep(0.05)        # ~20 fps ceiling; cheap when idle
                continue

            result = dice.attempt_roll(gate, cfg.dice_sides, cfg.dc)
            if not result.rolled:
                continue        # within cooldown window

            self.status.rolls += 1
            self.status.last_roll = result.describe()
            self.status.last_roll_at = time.time()
            log.info("Person detected: %s", result.describe())

            if result.treat:
                self.status.treats += 1
                try:
                    caster.play_sound(
                        cfg.speaker_name,
                        cfg.sound_file,
                        dont_interrupt=cfg.dont_interrupt_playback,
                    )
                except Exception as exc:
                    self.status.last_error = f"cast error: {exc}"
                    log.warning("Failed to cast sound: %s", exc)

    # -- one-off test --------------------------------------------------------
    def test_cast(self) -> None:
        """Force a treat sound on the configured speaker (GUI 'Test' button)."""
        cfg = config_mod.load()
        if not cfg.speaker_name:
            raise ValueError("No speaker selected.")
        if self._sound_server is None:
            self._sound_server = SoundServer(port=cfg.file_server_port)
        Caster(self._sound_server).play_sound(cfg.speaker_name, cfg.sound_file)


def _camera_source(cfg) -> str:
    """Inject username/password into an rtsp:// URL if provided separately."""
    url = cfg.camera_url
    if cfg.camera_username and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        cred = cfg.camera_username
        if cfg.camera_password:
            cred += f":{cfg.camera_password}"
        return f"{scheme}://{cred}@{rest}"
    return url
