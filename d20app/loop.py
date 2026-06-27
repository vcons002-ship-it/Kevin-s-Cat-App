"""The background detection loop, started/stopped by the web GUI.

Reads the saved config, watches the chosen camera for a person, and on each
permitted detection rolls the die and casts the sound on a treat. Runs in a
daemon thread so the Flask GUI stays responsive; exposes live status (running
state, last roll, last treat) back to the GUI.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import quote

from . import config as config_mod
from . import dice
from .activitylog import ActivityLog
from .caster import Caster, SoundServer
from .cats import CatTracker
from .detector import PersonDetector, mask_credentials
from .snapshots import SnapshotStore

log = logging.getLogger("d20app.loop")

# Don't log every frame of a wandering cat — at most one motion note this often.
_MOTION_LOG_INTERVAL = 10.0


def _parse_hhmm(value: str):
    """Parse 'HH:MM' to a datetime.time, or None if blank/invalid."""
    try:
        h, m = value.strip().split(":")
        return datetime.time(int(h), int(m))
    except (ValueError, AttributeError):
        return None


def in_quiet_window(now: datetime.time, start: str, end: str) -> bool:
    """True if ``now`` falls in the [start, end) quiet window.

    Handles a window that wraps past midnight (e.g. 22:00 → 07:00). If either
    bound is blank/invalid, quiet time is disabled and this returns False.
    """
    s, e = _parse_hhmm(start), _parse_hhmm(end)
    if s is None or e is None or s == e:
        return False
    if s < e:
        return s <= now < e
    return now >= s or now < e        # wraps midnight


@dataclass
class Status:
    running: bool = False
    last_error: str = ""
    last_roll: str = ""          # human-readable, e.g. "rolled 18/d20 vs DC18 -> TREAT!"
    last_roll_at: float | None = None
    treats: int = 0
    rolls: int = 0


def _cooldown_resume_delay(cfg) -> float:
    """Seconds before the cooldown ends at which to resume the neural net.

    Enough lead to rebuild the confirm-frames streak (``confirm_frames`` frames at
    ``scan_fps``) plus a small margin for stream warmup, so the next treat window
    isn't missed when detection was paused to save CPU.
    """
    per_frame = 1.0 / max(1.0, float(cfg.scan_fps))
    return max(3.0, int(cfg.confirm_frames) * per_frame + 1.0)


class DetectionLoop:
    """Owns the worker thread and shared state for one detection session."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.status = Status()
        self.activity = ActivityLog()
        self.snapshots = SnapshotStore()
        self.cats = CatTracker()        # cat sightings, for the "show cat" feature
        self._sound_server: SoundServer | None = None
        self._caster: Caster | None = None
        self._detector: PersonDetector | None = None   # live while running, for the GUI feed

    def _caster_for(self, cfg) -> Caster:
        """A single long-lived Caster so speaker connections stay open."""
        if self._sound_server is None:
            self._sound_server = SoundServer(port=cfg.file_server_port)
        if self._caster is None:
            self._caster = Caster(self._sound_server)
        return self._caster

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

    def live_jpeg(self) -> bytes | None:
        """Current annotated frame from the running detector, for the live feed.

        ``None`` when the loop isn't running or hasn't read a frame yet.
        """
        det = self._detector
        return det.live_jpeg() if det is not None else None

    def live_version(self) -> int:
        """Frame/box version of the running detector (0 if not running)."""
        det = self._detector
        return det.live_version() if det is not None else 0

    def cat_present(self) -> bool:
        """True if a cat is on camera right now (for the flashing Show-cat button)."""
        det = self._detector
        return det.cat_present() if det is not None else False

    def set_smooth(self, on: bool) -> None:
        """Request smooth-feed on/off on the running detector.

        Only sets the desired flag; the loop thread reconciles it on its next
        frame, so the camera is never read by two threads at once. No-op if the
        loop isn't running (the saved config takes effect on the next start).
        """
        det = self._detector
        if det is not None:
            det._smooth_desired = bool(on)

    # -- the worker ----------------------------------------------------------
    def _run(self) -> None:
        cfg = config_mod.load()
        if not cfg.camera_url:
            self.status.last_error = "No camera selected — choose one in the GUI."
            self.activity.add("error", "Can't start: no camera selected.")
            self.status.running = False
            return
        targets = config_mod.speaker_targets(cfg)
        if not targets:
            self.status.last_error = "No speaker selected — choose one in the GUI."
            self.activity.add("error", "Can't start: no speaker selected.")
            self.status.running = False
            return

        caster = self._caster_for(cfg)
        if cfg.keep_speakers_warm:
            # Loop silence so the receiver stays loaded — no "connecting" chime.
            caster.start_keepalive(targets)

        detector = PersonDetector(
            source=_camera_source(cfg),
            confidence=cfg.person_confidence,
            roi=cfg.roi,
            detect_size=cfg.detect_size,
            label_floor=cfg.label_floor,
            motion_min_area_frac=cfg.motion_min_area_frac,
            motion_diff_threshold=cfg.motion_diff_threshold,
            motion_min_blob_px=cfg.motion_min_blob_px,
            model=cfg.detector_model,
            accelerator=cfg.accelerator,
            smooth_feed=cfg.smooth_live_feed,
        )
        self._detector = detector          # expose for the live GUI feed
        gate = dice.RollGate(cfg.cooldown_seconds)

        # Never echo a password: prefer the friendly name, else the URL with
        # any embedded credentials masked.
        cam_label = cfg.camera_name or mask_credentials(cfg.camera_url)
        speakers_label = ", ".join(targets)
        log.info("Detection loop started (camera=%s, speakers=%s)",
                 cam_label, speakers_label)
        self.activity.add(
            "info",
            f"▶ Started watching {cam_label} "
            f"(speakers: {speakers_label}, treat on d{cfg.dice_sides} ≥ {cfg.dc}).",
        )
        try:
            self._loop_body(cfg, detector, gate, caster, targets, speakers_label)
        except Exception as exc:  # keep the GUI informed rather than dying silently
            log.exception("Detection loop crashed")
            self.status.last_error = str(exc)
            self.activity.add("error", f"Detection loop crashed: {exc}")
        finally:
            self._detector = None   # stop serving the live feed once we're done
            detector.release()
            caster.close()          # drop held speaker connections when we stop
            self.status.running = False
            log.info("Detection loop stopped")
            self.activity.add("info", "■ Stopped watching.")

    def _loop_body(self, cfg, detector, gate, caster, targets, speakers_label) -> None:
        backoff = 1.0
        cam_label = cfg.camera_name or mask_credentials(cfg.camera_url)
        last_cam_error = ""        # so a flaky camera doesn't flood the log
        connected = False          # log once when the first frame actually reads
        motion_gate = dice.RollGate(_MOTION_LOG_INTERVAL)   # throttle motion notes
        streak = 0                 # consecutive person frames (false-positive guard)
        confirm_frames = max(1, int(cfg.confirm_frames))
        interval = 1.0 / max(1.0, float(cfg.scan_fps))      # seconds between reads
        resume_at = 0.0            # monotonic time to resume the net after a roll (0 = not paused)
        while not self._stop.is_set():
            # During the post-roll cooldown the net can't trigger anything, so
            # skip it to save CPU — but keep reading frames so the stream stays
            # warm and a dead camera is still noticed. Resumes (with lead time)
            # just before the cooldown window reopens.
            paused = bool(
                cfg.pause_during_cooldown and resume_at and time.monotonic() < resume_at
            )
            try:
                outcome = detector.read_and_detect(detect=not paused)
            except FileNotFoundError as exc:
                self.status.last_error = str(exc)
                self.activity.add("error", str(exc))
                return
            except Exception as exc:
                self.status.last_error = f"camera error: {exc}"
                if str(exc) != last_cam_error:        # log only on change
                    self.activity.add("error", f"Camera problem: {exc} (retrying…)")
                    last_cam_error = str(exc)
                time.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)
                continue
            if last_cam_error:
                self.activity.add("info", "Camera stream recovered.")
                last_cam_error = ""
            backoff = 1.0

            # Confirm — once — that frames are actually flowing, so a running
            # loop that simply hasn't seen a person yet isn't silent.
            if not connected and detector.frame_size:
                w, h = detector.frame_size
                self.activity.add(
                    "info", f"📷 Camera connected ({w}×{h}) — watching for people."
                )
                connected = True

            if not outcome.motion:
                streak = 0          # nothing moving — reset the person streak
                time.sleep(interval)    # scan-rate ceiling; cheap when idle
                continue

            # Motion, but not a person: report what moved (the cats!), throttled
            # so a pacing pet doesn't flood the log, with an annotated snapshot.
            # A cat is also recorded as a sighting (when + where) for "show cat".
            if not outcome.person:
                streak = 0
                if motion_gate.allow():
                    snap = self.snapshots.save(detector.annotated_jpeg())
                    cat = detector.best_box("cat") if "cat" in outcome.labels else None
                    if cat is not None:
                        score, box = cat
                        sighting = self.cats.record(
                            cam_label, box, detector.frame_size, score, image=snap
                        )
                        where = f" ({sighting['region']})" if sighting["region"] else ""
                        self.activity.add(
                            "motion",
                            f"🐱 Cat seen{where} on {cam_label} — tracked, no roll.",
                            image=snap,
                        )
                    else:
                        what = outcome.labels[0] if outcome.labels else "something"
                        self.activity.add(
                            "motion",
                            f"Non-human motion — {what} moved (no person, no roll).",
                            image=snap,
                        )
                time.sleep(interval)
                continue

            # A person was seen — require it to persist across several frames
            # before acting. Single-frame false positives never reach the count.
            streak += 1
            if streak < confirm_frames:
                time.sleep(interval)
                continue

            result = dice.attempt_roll(gate, cfg.dice_sides, cfg.dc)
            if not result.rolled:
                time.sleep(interval)
                continue        # within cooldown window

            # A roll just happened, so the cooldown gate is now closed for
            # cfg.cooldown_seconds. Pause the net until shortly before it reopens.
            if cfg.pause_during_cooldown and cfg.cooldown_seconds > 0:
                resume_at = time.monotonic() + max(
                    0.0, cfg.cooldown_seconds - _cooldown_resume_delay(cfg)
                )
                self.activity.add(
                    "info",
                    f"Detection paused ~{round(cfg.cooldown_seconds / 60)} min "
                    "for cooldown (saving CPU) — resumes before the next window.",
                )

            self.status.rolls += 1
            self.status.last_roll = result.describe()
            self.status.last_roll_at = time.time()
            log.info("Person detected: %s", result.describe())
            image = self.snapshots.save(detector.annotated_jpeg())

            roll_desc = f"rolled {result.value} on d{cfg.dice_sides} (need ≥ {cfg.dc})"
            if not result.treat:
                self.activity.add(
                    "roll", f"Person detected — {roll_desc}: no treat.", image=image
                )
                continue

            # A treat — but stay silent during quiet time.
            now = datetime.datetime.now().time()
            if in_quiet_window(now, cfg.quiet_start, cfg.quiet_end):
                self.activity.add(
                    "roll",
                    f"Person detected — {roll_desc}: TREAT, but it's quiet "
                    f"time ({cfg.quiet_start}–{cfg.quiet_end}) — chime suppressed.",
                    image=image,
                )
                continue

            self.status.treats += 1
            self._cast_for_treat(cfg, caster, targets, speakers_label,
                                 result, roll_desc, image)

    def _cast_for_treat(self, cfg, caster, targets, speakers_label,
                        result, roll_desc, image) -> None:
        """Cast the chime/speech for a won roll and log the outcome.

        Split out from the loop body so it takes its speaker arguments explicitly
        (``targets`` / ``speakers_label``) rather than reaching for ``_run``'s
        locals — that cross-method reference was the old ``NameError`` crash.
        """
        what = "Spoke the message on" if cfg.use_speech else "Chime sent to"
        try:
            if cfg.use_speech:
                cast = caster.say(targets, cfg.speech_text,
                                  dont_interrupt=cfg.dont_interrupt_playback)
            else:
                cast = caster.play_sound(targets, cfg.sound_file,
                                         dont_interrupt=cfg.dont_interrupt_playback)
            if cast:
                self.activity.add(
                    "treat",
                    f"Person detected — {roll_desc}: TREAT! 🎉 "
                    f"{what} {speakers_label}.",
                    image=image,
                )
            else:
                self.activity.add(
                    "roll",
                    f"Person detected — {roll_desc}: TREAT, but the "
                    f"speaker(s) were already playing — skipped.",
                    image=image,
                )
        except Exception as exc:
            self.status.last_error = f"cast error: {exc}"
            log.warning("Failed to cast sound: %s", exc)
            self.activity.add(
                "error",
                f"Rolled a treat ({result.value}) but couldn't reach "
                f"{speakers_label}: {exc}",
                image=image,
            )

    # -- one-off test --------------------------------------------------------
    def test_cast(self) -> None:
        """Play the chime (or speak the message) on the configured speakers."""
        cfg = config_mod.load()
        targets = config_mod.speaker_targets(cfg)
        if not targets:
            raise ValueError("No speaker selected.")
        caster = self._caster_for(cfg)
        label = ", ".join(targets)
        try:
            if cfg.use_speech:
                caster.say(targets, cfg.speech_text)
                self.activity.add("info", f"🗣 Spoke the message on {label}.")
            else:
                caster.play_sound(targets, cfg.sound_file)
                self.activity.add("info", f"🔊 Test sound played on {label}.")
        except Exception as exc:
            self.activity.add("error", f"Test failed on {label}: {exc}")
            raise


def _camera_source(cfg) -> str:
    """Inject username/password into an rtsp:// URL if provided separately.

    Credentials are percent-encoded so a password containing URL-significant
    characters (``@ : / ? #``) doesn't corrupt the URL and cause a spurious
    401 / connection failure.
    """
    url = cfg.camera_url
    if cfg.camera_username and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        cred = quote(cfg.camera_username, safe="")
        if cfg.camera_password:
            cred += ":" + quote(cfg.camera_password, safe="")
        return f"{scheme}://{cred}@{rest}"
    return url
