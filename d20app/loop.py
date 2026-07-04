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

from . import config as config_mod
from . import dice
from .activitylog import ActivityLog
from .caster import Caster, SoundServer
from .cats import CatTracker, zone_for
from .detector import PersonDetector, mask_credentials
from .snapshots import SnapshotStore

log = logging.getLogger("d20app.loop")

# Don't log every frame of a wandering cat — at most one motion note this often.
_MOTION_LOG_INTERVAL = 10.0

# How long "Show cat" forces continuous detection on the camera it jumps to, so the
# live feed keeps drawing a box around the cat (even a still one) while the user looks.
_CAT_BOOST_SECONDS = 20.0

# How long a camera's newest recorded sighting stays on the live feed as the grey
# "last known location" box (0.42.0). Long-ish on purpose: stale-but-labelled beats
# absent — the box carries its own age, so 25 minutes old reads as 25 minutes old.
_LAST_KNOWN_TTL = 1800.0

# How long a camera stays pinned-active after the GUI last fetched a frame of it, so a
# camera you're watching never sleeps under round-robin (refreshed each streamed frame).
_VIEW_TTL = 8.0


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


def _shown_label(label: str) -> str:
    """What a counted-as-the-cat sighting is CALLED in the log (#98): "cat".

    A no-dog household opts into counting the model's dog-mislabels as the cat
    (locator_classes = ["cat", "dog"]) — so the log saying "Still dog seen"
    under a 🐱 surfaces exactly the misclassification the user asked us to
    absorb. The stored sighting keeps the RAW label (data stays honest); only
    the sentence says cat. Anything that reaches these log lines was already a
    locator hit, so mapping every non-"cat" label is correct by construction."""
    return "cat"


def _cooldown_resume_delay(cfg) -> float:
    """Seconds before the cooldown ends at which to resume the neural net.

    Enough lead to rebuild the confirm-frames streak (``confirm_frames`` frames at
    ``scan_fps``) plus a small margin for stream warmup, so the next treat window
    isn't missed when detection was paused to save CPU.
    """
    per_frame = 1.0 / max(1.0, float(cfg.scan_fps))
    return max(3.0, int(cfg.confirm_frames) * per_frame + 1.0)


def _cat_flash_ttl(cfg) -> float:
    """How long (sec) a cat counts as "present" after the net last saw it.

    Spans the gap between periodic forced scans so a still cat keeps flashing the
    button. With always-on / disabled scanning the net refreshes at frame rate (or
    only on motion), so a short window is enough.
    """
    iv = float(getattr(cfg, "cat_scan_interval", 30.0))
    return iv + 2.0 if iv > 0 else 2.0


def _locator_hit(detector, outcome):
    """``(label, score, box)`` for the strongest locator-class detection this frame
    (the cat — possibly a "dog" standing in for it), or ``None``. Picks the best box
    across ``detector.locator_classes``."""
    best = None
    for cls in detector.locator_classes:
        if cls in outcome.labels:
            bb = detector.best_box(cls)
            if bb and (best is None or bb[0] > best[1]):
                best = (cls, bb[0], bb[1])     # (label, score, box)
    return best


def _cat_scan_due(cfg, track_cats: bool, last_scan: float, now: float) -> bool:
    """Whether a cat-tracking camera should force a still-cat scan this iteration.

    ``cat_scan_interval``: ``<0`` off (motion only), ``0`` always-on (every frame),
    ``>0`` every N seconds since the net last ran.
    """
    if not track_cats:
        return False
    iv = float(getattr(cfg, "cat_scan_interval", 30.0))
    if iv < 0:
        return False
    if iv == 0:
        return True
    return (now - last_scan) >= iv


class DetectionLoop:
    """Owns the worker thread and shared state for one detection session."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None   # the orchestrator thread
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.status = Status()
        self.activity = ActivityLog()
        self.snapshots = SnapshotStore()
        self.cats = CatTracker()        # cat sightings, for the "show cat" feature
        self._sound_server: SoundServer | None = None
        self._caster: Caster | None = None
        # Multi-camera: one PersonDetector + one worker thread per watched camera.
        # _detectors is built once at start and rebound to {} after all joins, so
        # the web thread reads it lock-free via a stable reference.
        self._detectors: dict[str, PersonDetector] = {}
        self._threads: list[threading.Thread] = []
        self._gate: dice.RollGate | None = None         # SHARED cooldown gate
        self._roll_lock = threading.Lock()              # guards gate + roll bookkeeping
        self._status_lock = threading.Lock()            # guards Status mutation
        self._cam_lock = threading.Lock()               # guards _cam_status
        self._resume_at = 0.0                           # SHARED cooldown-pause deadline
        self._cam_status: dict[str, dict] = {}          # name -> {connected,last_error,roll,track_cats}
        self._live_name: str | None = None              # camera the GUI streams by default
        self._cat_flash_ttl = 2.0                       # how long a cat stays "present" between scans
        self._cat_boost: dict[str, float] = {}          # name -> monotonic deadline for forced detection
        self._viewing: dict[str, float] = {}            # name -> deadline; the GUI is streaming this camera
        self._active_cams: set[str] | None = None       # round-robin: which cameras detect now (None = all)
        self._scan_last: dict[str, dict] = {}           # name -> {ts, found}: last still-scan (#94)

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

    def _pick(self, name: str | None) -> PersonDetector | None:
        if name and name in self._detectors:
            return self._detectors[name]
        return self._detectors.get(self._live_name)   # fall back to the default feed

    def get_detector(self, name: str) -> PersonDetector | None:
        """The named camera's running detector, or None. Lock-free: ``_detectors``
        is built once per start and rebound (never mutated), same as :meth:`_pick`.
        Used by the on-demand escalation endpoint (#66) — NOT a fallback lookup."""
        return self._detectors.get(name)

    def live_jpeg(self, name: str | None = None, trail: bool = False,
                  last_known: bool = True) -> bytes | None:
        """Annotated frame from a camera's detector (defaults to the streamed one).

        ``trail=True`` composites the live cat-trail overlay (0.39.0).
        ``last_known`` (default on, 0.42.0) draws the camera's newest recorded
        sighting as a grey, age-labelled box — "where was she last?" stays
        answered even when nothing is detected right now. ``None`` when the loop
        isn't running or that camera hasn't read a frame.
        """
        det = self._pick(name)
        if det is None:
            return None
        return det.live_jpeg(trail=trail,
                             last_known=self._last_known(name) if last_known else None)

    def _last_known(self, name: str | None) -> dict | None:
        """The streamed camera's newest confirmed cat position as an overlay
        payload, or None (nothing known, older than :data:`_LAST_KNOWN_TTL`, or
        no camera).

        Prefers the detector's live confirmation track (0.42.1) — updated on
        every ≥cat_confidence detection, so the grey box points at the *true*
        last confirmed spot instead of lagging the throttled sightings log by
        up to a record interval. The log is the fallback (it survives a loop
        restart; the live track doesn't)."""
        cam = name if name and name in self._detectors else self._live_name
        if not cam:
            return None
        det = self._detectors.get(cam)
        if det is not None:
            live = det.last_confirmed()
            if live and live["age_s"] <= _LAST_KNOWN_TTL:
                return {"box": live["box"], "label": live["label"],
                        "age_s": live["age_s"]}
        s = self.cats.last_for(cam)
        if not s or not s.get("box"):
            return None
        age = time.time() - s.get("ts", 0.0)
        if not 0 <= age <= _LAST_KNOWN_TTL:
            return None
        return {"box": s["box"], "label": s.get("label", "cat"), "age_s": age}

    def live_version(self, name: str | None = None) -> int:
        """Frame/box version of a camera's detector (0 if not running)."""
        det = self._pick(name)
        return det.live_version() if det is not None else 0

    def cats_present_cameras(self) -> list:
        """Names of **cat-tracking** cameras seeing a cat now, newest sighting first.

        A cat is "present" if the net saw one within ``_cat_flash_ttl`` — a window
        sized to the scan interval so a *still* cat re-found by the periodic forced
        scan keeps flashing the button (and stays in the Show-cat rotation) between
        scans, not just for the 1.5 s box-TTL.
        """
        status = self._cam_status
        now = time.monotonic()
        seen = []
        for cam_name, det in self._detectors.items():
            if not status.get(cam_name, {}).get("track_cats"):
                continue
            last = det.cat_last_seen()
            if last and now - last <= self._cat_flash_ttl:
                seen.append((last, cam_name))
        seen.sort(reverse=True)        # most-recent sighting first
        return [name for _, name in seen]

    def cat_present(self) -> bool:
        """True if **any cat-tracking** camera has a cat on it right now."""
        return bool(self.cats_present_cameras())

    def note_viewing(self, name: str | None = None) -> None:
        """Mark a camera as actively viewed by the GUI (refreshed each streamed
        frame), so round-robin keeps it active while you watch it."""
        target = name or self._live_name
        if target:
            self._viewing[target] = time.monotonic() + _VIEW_TTL

    @staticmethod
    def _rr_window(rotatable: list, always_on: set, start: int, size: int) -> set:
        """The round-robin active set: the always-on cameras plus a window of
        ``size`` rotatable cameras starting at ``start`` (wrapping)."""
        window = set(always_on)
        n = len(rotatable)
        for i in range(min(size, n)):
            window.add(rotatable[(start + i) % n])
        return window

    def _is_pinned(self, name: str, now: float) -> bool:
        """A viewed or boosted camera stays active regardless of the rotation."""
        return self._viewing.get(name, 0.0) > now or self._cat_boost.get(name, 0.0) > now

    def _camera_active(self, name: str) -> bool:
        """Whether a camera should detect now (round-robin gate). True for all when
        round-robin is off (``_active_cams is None``)."""
        active = self._active_cams
        if active is None or name in active:
            return True
        return self._is_pinned(name, time.monotonic())

    def _record_fused(self, name: str, cam_label: str, spec: dict, detector) -> dict | None:
        """Claim and record a temporal-fusion confirmation (0.37.0), or None.

        The fused hit is pure YOLO evidence accumulated across frames — a moving
        cat no single frame could confirm — so it's recorded as an ordinary
        sighting, honestly tagged ``source="track"`` with the mean weak score.
        """
        take = getattr(detector, "take_fused_hit", None)   # absent on test stubs
        fused = take() if take else None
        if not fused:
            return None
        snap = self.snapshots.save(detector.annotated_jpeg())
        sighting = self.cats.record(
            name, tuple(fused["box"]), detector.frame_size, fused["score"],
            image=snap, label="cat", source="track",
            zone=zone_for(fused["box"], spec.get("zones"), spec.get("roi")))
        spot = sighting.get("zone") or sighting["region"]
        where = f" ({spot})" if spot else ""
        self.activity.add(
            "motion",
            f"🐱 Moving cat confirmed by track fusion ({fused['n']} weak hits over "
            f"{fused['span_s']}s){where} on {cam_label} — tracked, no roll.",
            image=snap)
        return sighting

    def boost_detection(self, name: str, seconds: float | None = None,
                        box=None) -> bool:
        """Run the net continuously on ``name`` for ``seconds`` (default
        :data:`_CAT_BOOST_SECONDS`), so the live feed keeps drawing a box around the
        cat — even a motionless one between periodic scans — while the user looks.

        ``box`` (0.42.0) makes the boost **targeted**: forced scans additionally
        zoom a full-resolution crop around that spot and run the heaviest model
        on disk there — for leads that name a *place* (a VLM "yes" box). The
        feed draws the box as "checking (lead)" while it's live.

        Returns False if that camera isn't currently being watched. The worker reads
        the deadline lock-free (one float key per camera; the GIL makes the dict
        get/set atomic), the same build-once/read-many pattern as ``_detectors``.
        """
        if not name or name not in self._detectors:
            return False
        dur = _CAT_BOOST_SECONDS if seconds is None else max(0.0, float(seconds))
        self._cat_boost[name] = time.monotonic() + dur
        if box is not None:
            self._detectors[name].set_boost_hint(box, dur)
        return True

    def cam_status(self) -> list:
        """Per-camera {name, connected, last_error, roll, track_cats} for the GUI.

        Enriched with what the net ACTUALLY runs on (#90): ``ran_on`` (the
        effective accelerator) and ``fallback`` (why, when it isn't the request)
        — a degraded camera must be visible, not just a buried log line."""
        with self._cam_lock:
            rows = [{"name": n, **dict(s)} for n, s in self._cam_status.items()]
        detectors = self._detectors
        for row in rows:
            det = detectors.get(row["name"])
            probe = getattr(det, "effective_accelerator", None)   # stub-tolerant
            if probe is not None:
                eff, why = probe()
                if eff:
                    row["ran_on"] = eff
                    if why:
                        row["fallback"] = why
            scan = self._scan_last.get(row["name"])
            if scan:
                row["scan_ago_s"] = round(time.time() - scan["ts"], 1)
                row["scan_found"] = bool(scan["found"])
        return rows

    def set_smooth(self, on: bool) -> None:
        """Request smooth-feed on/off on every running detector.

        Only sets the desired flag; each loop thread reconciles it on its next
        frame, so a camera is never read by two threads at once.
        """
        for det in self._detectors.values():
            det._smooth_desired = bool(on)

    def _fail_start(self, err: str, log_msg: str) -> None:
        with self._status_lock:
            self.status.last_error = err
            self.status.running = False
        self.activity.add("error", log_msg)

    # -- the orchestrator ----------------------------------------------------
    def _run(self) -> None:
        """Build one detector + worker thread per watched camera, then supervise.

        This thread does no detection itself — it owns the worker threads and the
        shared state so start/stop/join has a single supervisory thread, and so
        ``_detectors``/``_threads`` have exactly one writer (no concurrent-mutation
        race against the web thread's lock-free reads).
        """
        cfg = config_mod.load()
        specs = config_mod.camera_targets(cfg)
        if not specs:
            self._fail_start("No camera selected — choose one in the GUI.",
                             "Can't start: no camera selected.")
            return
        targets = config_mod.speaker_targets(cfg)
        if not targets:
            self._fail_start("No speaker selected — choose one in the GUI.",
                             "Can't start: no speaker selected.")
            return

        caster = self._caster_for(cfg)
        if cfg.keep_speakers_warm:
            caster.start_keepalive(targets)

        self._gate = dice.RollGate(cfg.cooldown_seconds)   # SHARED across cameras
        self._resume_at = 0.0
        # Keep a cat "present" (flashing button / Show-cat rotation) across the gap
        # between forced scans, so a still cat scanned every N s doesn't flicker.
        self._cat_flash_ttl = _cat_flash_ttl(cfg)
        speakers_label = ", ".join(targets)

        # Build all detectors first, then publish them and spawn workers.
        detectors: dict[str, PersonDetector] = {}
        cam_status: dict[str, dict] = {}
        for spec in specs:
            name = spec["name"]
            detectors[name] = PersonDetector(
                source=spec["source"],
                confidence=spec["person_confidence"],
                roi=spec["roi"],
                label_floor=spec["label_floor"],
                motion_min_area_frac=spec["motion_min_area_frac"],
                motion_diff_threshold=spec["motion_diff_threshold"],
                motion_min_blob_px=spec["motion_min_blob_px"],
                motion_gate=str(spec.get("motion_sensitivity", "")) != "off",
                model=spec["model"],
                accelerator=spec["accelerator"],
                smooth_feed=spec["smooth_feed"],
                gamma=spec["gamma"],
                brightness=spec["brightness"],
                contrast=spec["contrast"],
                saturation=spec["saturation"],
                cat_scan_tiling=spec["cat_scan_tiling"],
                cat_scan_tile_overlap=spec["cat_scan_tile_overlap"],
                cat_scan_imgsz=spec["cat_scan_imgsz"],
                cat_scan_frames=spec["cat_scan_frames"],
                cat_scan_model=spec.get("cat_scan_model", ""),
                track_fusion=spec["track_fusion"],
                cat_confidence=spec["cat_confidence"],
                locator_classes=spec["locator_classes"],
            )
            cam_status[name] = {"connected": False, "last_error": "",
                                "roll": bool(spec["roll"]),
                                "track_cats": bool(spec["track_cats"]),
                                "always_watch": bool(spec["always_watch"]),
                                "resting": False}
        self._detectors = detectors            # built once; rebound to {} after joins
        with self._cam_lock:
            self._cam_status = cam_status
        self._live_name = specs[0]["name"]

        cam_names = ", ".join(s["name"] for s in specs)
        log.info("Detection loop started (cameras=%s, speakers=%s)",
                 cam_names, speakers_label)
        self.activity.add(
            "info",
            f"▶ Watching {len(specs)} camera(s): {cam_names} "
            f"(speakers: {speakers_label}, treat on d{cfg.dice_sides} ≥ {cfg.dc}).",
        )

        # Round-robin: rotate which cameras detect at once to cap CPU. Off (or when
        # every rotatable camera already fits in one set) → all active (None).
        rr_size = max(1, int(cfg.round_robin_size))
        rotatable = [s["name"] for s in specs if not s["always_watch"]]
        always_on = {s["name"] for s in specs if s["always_watch"]}
        rotating = bool(cfg.round_robin) and len(rotatable) > rr_size
        self._active_cams = self._rr_window(rotatable, always_on, 0, rr_size) if rotating else None
        if rotating:
            self.activity.add(
                "info",
                f"Round-robin: {rr_size} of {len(rotatable)} cameras at a time, "
                f"rotating every {round(float(cfg.round_robin_interval))}s.")

        threads = []
        for spec in specs:
            t = threading.Thread(
                target=self._camera_worker,
                args=(spec, detectors[spec["name"]], cfg, caster, targets, speakers_label),
                name=f"cam-{spec['name']}", daemon=True,
            )
            t.start()
            threads.append(t)
        self._threads = threads

        try:
            if not rotating:
                self._stop.wait()  # supervise until stop() is called or a fatal error
            else:
                interval = max(0.2, float(cfg.round_robin_interval))
                start = 0
                while not self._stop.is_set():
                    self._stop.wait(interval)
                    if self._stop.is_set():
                        break
                    start = (start + rr_size) % len(rotatable)
                    self._active_cams = self._rr_window(rotatable, always_on, start, rr_size)
        finally:
            for t in threads:
                t.join(timeout=10)
            for det in detectors.values():
                try:
                    det.release()
                except Exception:      # noqa: BLE001 — never let cleanup raise
                    log.exception("error releasing a detector")
            self._detectors = {}       # stop serving the live feed
            self._threads = []
            self._cat_boost = {}       # drop any pending detection boosts
            self._scan_last = {}
            self._viewing = {}
            self._active_cams = None
            caster.close()             # drop held speaker connections when we stop
            with self._status_lock:
                self.status.running = False
            log.info("Detection loop stopped")
            self.activity.add("info", "■ Stopped watching.")

    # -- per-camera status helpers (thread-safe) -----------------------------
    def _cam_set(self, name: str, **fields) -> None:
        with self._cam_lock:
            if name in self._cam_status:
                self._cam_status[name].update(fields)

    def _cam_error(self, name: str, err: str) -> None:
        """Record a camera's error and roll it up into the global Status."""
        with self._cam_lock:
            if name in self._cam_status:
                self._cam_status[name]["last_error"] = err
                if err:
                    self._cam_status[name]["connected"] = False
        if err:
            with self._status_lock:
                self.status.last_error = f"{name}: {err}"

    # -- the per-camera worker ----------------------------------------------
    def _camera_worker(self, spec, detector, cfg, caster, targets, speakers_label) -> None:
        """Watch one camera. Role-gated: rolls for treats and/or tracks cats.

        Fully isolated — any failure here exits this thread only; it never touches
        the other workers or the orchestrator, and only the orchestrator releases
        the detector.
        """
        name = spec["name"]
        cam_label = mask_credentials(name)
        roll_enabled = bool(spec["roll"])
        track_cats = bool(spec["track_cats"])
        backoff = 1.0
        last_cam_error = ""
        connected = False
        motion_gate = dice.RollGate(_MOTION_LOG_INTERVAL)   # throttle motion notes
        streak = 0
        confirm_frames = max(1, int(spec["confirm_frames"]))
        interval = 1.0 / max(1.0, float(spec["scan_fps"]))
        last_scan = 0.0          # monotonic time the net last ran (motion or forced)
        cat_seen_still = False   # was a cat present on the previous *forced* still scan?
        resting = False          # round-robin: currently asleep (capture released)
        # A camera may skip the net during the shared cooldown only if it has
        # nothing to do then: it rolls (so the closed gate blocks it anyway) AND
        # it doesn't track cats (which must keep detecting). A cat-tracking camera
        # never pauses, so it stays watching for cats during another camera's cooldown.
        can_pause = roll_enabled and not track_cats
        try:
            while not self._stop.is_set():
                # Round-robin gate: when it's not this camera's turn, release the
                # capture (stop decoding → the CPU win) and idle until it is. A
                # viewed/boosted camera is never rested (see _camera_active).
                if not self._camera_active(name):
                    if not resting:
                        resting = True
                        # Keep `connected` as-is: a rotation rest isn't an error, so the
                        # camera shouldn't re-log its "connected" heartbeat on every wake.
                        try:
                            detector.release()
                        except Exception:      # noqa: BLE001 — never let rest raise
                            log.exception("error resting detector %s", name)
                        self._cam_set(name, resting=True)
                    self._stop.wait(0.2)   # fine poll so a camera wakes promptly on its turn
                    continue
                if resting:
                    resting = False
                    self._cam_set(name, resting=False)

                # Shared cooldown-pause: once any roll-camera rolls, eligible cameras
                # skip the net until just before the window reopens (read lock-free;
                # the deadline is written inside _roll_lock).
                now = time.monotonic()
                paused = bool(can_pause and cfg.pause_during_cooldown and self._resume_at
                              and now < self._resume_at)
                # Periodic still-cat scan: a sleeping cat makes no motion, so on a
                # cat-tracking camera force the net every cat_scan_interval seconds.
                # A "Show cat" boost forces it continuously for a short window so the
                # live feed keeps boxing the cat while the user looks (any camera).
                boost = now < self._cat_boost.get(name, 0.0)
                force_scan = _cat_scan_due(cfg, track_cats, last_scan, now) or boost
                try:
                    outcome = detector.read_and_detect(detect=not paused, force=force_scan)
                except FileNotFoundError as exc:
                    # Missing MODEL files are global & unrecoverable — stop everything.
                    with self._status_lock:
                        self.status.last_error = str(exc)
                    self.activity.add("error", str(exc))
                    self._stop.set()
                    return
                except Exception as exc:        # noqa: BLE001 — recoverable camera error
                    if str(exc) != last_cam_error:
                        self.activity.add("error",
                                          f"Camera problem on {cam_label}: {exc} (retrying…)")
                        last_cam_error = str(exc)
                        self._cam_error(name, str(exc))
                    self._stop.wait(min(backoff, 30))
                    backoff = min(backoff * 2, 30)
                    continue
                if last_cam_error:
                    self.activity.add("info", f"Camera {cam_label} recovered.")
                    last_cam_error = ""
                    self._cam_set(name, last_error="")
                backoff = 1.0

                if not connected and detector.frame_size:
                    w, h = detector.frame_size
                    self.activity.add("info", f"📷 {cam_label} connected ({w}×{h}).")
                    connected = True
                    self._cam_set(name, connected=True)

                if force_scan or outcome.motion:
                    last_scan = now      # the net ran; defer the next forced scan

                # Temporal fusion (0.37.0): a string of weak YOLO hits that chained
                # and MOVED was confirmed as one cat — record it like any sighting.
                if track_cats:
                    self._record_fused(name, cam_label, spec, detector)

                # Forced scan (periodic still-cat check or a Show-cat boost) with no
                # real motion: the net ran anyway and may have found a sleeping cat.
                # Record it on the rising edge (so a long nap logs once, not every
                # scan); the live flash/rotation are driven by cats_present_cameras().
                # Never rolls — a no-motion frame breaks the consecutive-motion person
                # streak, same as any idle frame.
                gate_off = str(spec.get("motion_sensitivity", "")) == "off"
                if (force_scan or gate_off) and not outcome.motion:
                    streak = 0
                    cat = _locator_hit(detector, outcome) if track_cats else None
                    if force_scan:
                        # glanceable proof the scan is firing (#94)
                        self._scan_last[name] = {"ts": time.time(),
                                                 "found": cat is not None}
                    if cat is not None:
                        if not cat_seen_still:
                            label, score, box = cat
                            snap = self.snapshots.save(detector.annotated_jpeg())
                            sighting = self.cats.record(
                                name, box, detector.frame_size, score, image=snap, label=label,
                                zone=zone_for(box, spec.get("zones"), spec.get("roi")))
                            spot = sighting.get("zone") or sighting["region"]
                            where = f" ({spot})" if spot else ""
                            self.activity.add(
                                "motion",
                                f"🐱 Still {_shown_label(label)} seen{where} on {cam_label} — tracked, no roll.",
                                image=snap)
                        cat_seen_still = True
                    else:
                        cat_seen_still = False
                    self._stop.wait(interval)
                    continue

                if not outcome.motion:
                    # Idle (no motion, no forced scan): the net didn't run, so leave
                    # the still-scan edge intact — only real motion resets it below.
                    streak = 0
                    self._stop.wait(interval)
                    continue
                cat_seen_still = False   # a real-motion frame supersedes the still-scan edge

                # Motion, not a person: record a cat sighting (if this camera tracks
                # cats) or just note the mover, throttled, with a snapshot.
                if not outcome.person:
                    streak = 0
                    if motion_gate.allow():
                        snap = self.snapshots.save(detector.annotated_jpeg())
                        cat = _locator_hit(detector, outcome) if track_cats else None
                        if cat is not None:
                            label, score, box = cat
                            sighting = self.cats.record(
                                name, box, detector.frame_size, score, image=snap, label=label,
                                zone=zone_for(box, spec.get("zones"), spec.get("roi")))
                            spot = sighting.get("zone") or sighting["region"]
                            where = f" ({spot})" if spot else ""
                            self.activity.add(
                                "motion",
                                f"🐱 {_shown_label(label).capitalize()} seen{where} on {cam_label} — tracked, no roll.",
                                image=snap)
                        else:
                            what = outcome.labels[0] if outcome.labels else "something"
                            self.activity.add(
                                "motion",
                                f"Non-human motion on {cam_label} — {what} moved.",
                                image=snap)
                    self._stop.wait(interval)
                    continue

                # A person. Only roll-enabled cameras act on it.
                if not roll_enabled:
                    self._stop.wait(interval)
                    continue
                streak += 1
                if streak < confirm_frames:
                    self._stop.wait(interval)
                    continue

                # --- shared roll critical section (fast: gate + counters only) ---
                pause_note = False
                with self._roll_lock:
                    result = dice.attempt_roll(self._gate, cfg.dice_sides, cfg.dc)
                    if result.rolled:
                        if cfg.pause_during_cooldown and cfg.cooldown_seconds > 0:
                            self._resume_at = time.monotonic() + max(
                                0.0, cfg.cooldown_seconds - _cooldown_resume_delay(cfg))
                            pause_note = True
                        with self._status_lock:
                            self.status.rolls += 1
                            self.status.last_roll = result.describe()
                            self.status.last_roll_at = time.time()
                if not result.rolled:
                    self._stop.wait(interval)
                    continue        # within the shared cooldown window

                # Slow work runs OUTSIDE the roll lock so a network cast on one
                # camera never blocks another camera's gate check.
                if pause_note:
                    self.activity.add(
                        "info",
                        f"Detection paused ~{round(cfg.cooldown_seconds / 60)} min for "
                        "cooldown (saving CPU) — resumes before the next window.")
                log.info("Person detected on %s: %s", cam_label, result.describe())
                image = self.snapshots.save(detector.annotated_jpeg())
                roll_desc = f"rolled {result.value} on d{cfg.dice_sides} (need ≥ {cfg.dc})"
                if not result.treat:
                    self.activity.add(
                        "roll", f"Person on {cam_label} — {roll_desc}: no treat.", image=image)
                    continue
                now = datetime.datetime.now().time()
                if in_quiet_window(now, cfg.quiet_start, cfg.quiet_end):
                    self.activity.add(
                        "roll",
                        f"Person on {cam_label} — {roll_desc}: TREAT, but it's quiet time "
                        f"({cfg.quiet_start}–{cfg.quiet_end}) — chime suppressed.", image=image)
                    continue
                with self._status_lock:
                    self.status.treats += 1
                self._cast_for_treat(cfg, caster, targets, speakers_label,
                                     result, roll_desc, image)
        except Exception as exc:        # noqa: BLE001 — isolate: this thread only
            log.exception("camera worker %s crashed", name)
            self._cam_error(name, f"worker crashed: {exc}")

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
            with self._status_lock:
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
    """Credential-injected source for the legacy single active camera.

    Kept for back-compat (webapp imports it); delegates to
    :func:`config.camera_source`. Multi-camera uses per-spec sources from
    :func:`config.camera_targets`.
    """
    return config_mod.camera_source(
        cfg.camera_url, cfg.camera_username, cfg.camera_password)
