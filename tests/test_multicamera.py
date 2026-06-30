"""Multi-camera: per-camera specs/roles, role-gated rolling/tracking, shared
cooldown across cameras, failure isolation, and the per-camera endpoints.

The detection loop is driven with a fake detector so no real cameras are needed.
"""

import time

import pytest

import d20app.config as config_mod
import d20app.loop as loopmod
from d20app.cats import CatTracker
from d20app.config import Config
from d20app.detector import CameraError, FrameOutcome
from d20app.webapp import create_app


# ---- config: per-camera specs ---------------------------------------------
def test_camera_targets_multi_and_roles():
    cfg = Config(
        cameras=[
            {"name": "A", "url": "rtsp://a/s", "username": "u", "password": "p@ss",
             "roll": True, "track_cats": False, "scan_fps": 5, "roi": [1, 2, 3, 4]},
            {"name": "B", "url": "usb:0", "roll": False, "track_cats": True},
        ],
        active_cameras=["A", "B", "Ghost"],   # Ghost isn't saved → dropped
    )
    specs = config_mod.camera_targets(cfg)
    assert [s["name"] for s in specs] == ["A", "B"]
    a, b = specs
    assert a["source"] == "rtsp://u:p%40ss@a/s" and a["roll"] and not a["track_cats"]
    assert a["scan_fps"] == 5 and a["roi"] == [1, 2, 3, 4]
    assert b["source"] == "usb:0" and not b["roll"] and b["track_cats"]
    # Missing per-camera settings inherit the global defaults.
    assert b["model"] == cfg.detector_model and b["confirm_frames"] == cfg.confirm_frames


def test_camera_targets_dedupes_active_names():
    # Duplicate names must not produce two specs (→ two threads on one detector).
    cfg = Config(cameras=[{"name": "X", "url": "rtsp://x/s"}],
                 active_cameras=["X", "X", "X"])
    specs = config_mod.camera_targets(cfg)
    assert [s["name"] for s in specs] == ["X"]


def test_coerce_camera_keeps_explicit_roi_none():
    # An explicit roi=None means whole-frame, not "inherit the global ROI".
    cfg = Config(roi=[0, 0, 99, 99])
    assert config_mod.coerce_camera({"name": "A", "roi": None}, cfg)["roi"] is None
    # ...but a camera that omits roi inherits the default.
    assert config_mod.coerce_camera({"name": "A"}, cfg)["roi"] == [0, 0, 99, 99]


def test_camera_targets_legacy_single_fallback():
    cfg = Config(camera_url="rtsp://x/s", camera_name="Solo")
    specs = config_mod.camera_targets(cfg)
    assert len(specs) == 1 and specs[0]["name"] == "Solo"
    assert specs[0]["roll"] and specs[0]["track_cats"]   # legacy camera does both
    assert config_mod.camera_targets(Config()) == []     # nothing configured


# ---- loop: a fake detector keyed by camera source -------------------------
class FakeDet:
    def __init__(self, source, **kw):
        self.source = source
        self.kw = kw
        self.frame_size = (64, 48)
        self._smooth_desired = kw.get("smooth_feed", False)
        self.released = False
        self.release_count = 0
        self._cat_last = 0.0
        self.locator_classes = tuple(kw.get("locator_classes") or ("cat",))

    def read_and_detect(self, detect=True, force=False):
        CALLS[self.source] = CALLS.get(self.source, 0) + 1
        time.sleep(0.003)
        outcome = OUTCOMES.get(self.source, FrameOutcome(False, False))
        if isinstance(outcome, Exception):
            raise outcome
        # A no-motion outcome is only "seen" when the net actually runs (real
        # motion, or a forced still-cat scan) — mirror the real detector.
        if (outcome.motion or force) and any(
                c in (outcome.labels or ()) for c in self.locator_classes):
            self._cat_last = time.monotonic()
        return outcome

    def cat_last_seen(self):
        return self._cat_last

    def best_box(self, label):
        return (0.9, (1, 1, 9, 9))

    def annotated_jpeg(self):
        return b"\xff\xd8x\xff\xd9"

    def cat_present(self):
        o = OUTCOMES.get(self.source)
        return bool(isinstance(o, FrameOutcome) and "cat" in (o.labels or ()))

    def live_jpeg(self):
        return b"\xff\xd8x\xff\xd9"

    def live_version(self):
        return 1

    def release(self):
        self.released = True
        self.release_count += 1


OUTCOMES = {}     # source -> FrameOutcome | Exception
CALLS = {}        # source -> number of read_and_detect calls (round-robin assertions)


def _run_loop(cfg, monkeypatch, tmp_path, seconds=0.4):
    """Start the loop with FakeDet + isolated cats; return (loop, treats[list])."""
    monkeypatch.setattr(loopmod, "PersonDetector", FakeDet)
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)
    treats = []
    monkeypatch.setattr(loopmod.DetectionLoop, "_cast_for_treat",
                        lambda self, *a, **k: treats.append(1))
    dummy_caster = type("C", (), {"start_keepalive": lambda *a: None, "close": lambda *a: None})()
    monkeypatch.setattr(loopmod.DetectionLoop, "_caster_for", lambda self, cfg: dummy_caster)
    lp = loopmod.DetectionLoop()
    lp.cats = CatTracker(path=str(tmp_path / "cats.log"))
    lp.start()
    time.sleep(seconds)
    return lp, treats


def _cam(name, url, **kw):
    return {"name": name, "url": url, "confirm_frames": 1, **kw}


def _base_cfg(cameras, **kw):
    opts = dict(speaker_names=["Spk"], cooldown_seconds=3600, dice_sides=1, dc=1,
                pause_during_cooldown=False, cameras=cameras,
                active_cameras=[c["name"] for c in cameras])
    opts.update(kw)
    return Config(**opts)


def test_roll_camera_treats_track_only_camera_does_not(monkeypatch, tmp_path):
    global OUTCOMES
    # One roll camera (person) and one track-cats-only camera (also a person).
    OUTCOMES = {"rtsp://a/s": FrameOutcome(True, True),
                "rtsp://c/s": FrameOutcome(True, True)}
    cfg = _base_cfg([_cam("A", "rtsp://a/s", roll=True, track_cats=False),
                     _cam("C", "rtsp://c/s", roll=False, track_cats=True)])
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        # Exactly one treat — from A; C never rolls, and the shared cooldown caps it.
        assert sum(treats) == 1
        assert lp.status.rolls == 1
    finally:
        lp.stop()


def test_no_treats_when_no_camera_rolls(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://c/s": FrameOutcome(True, True)}    # person on a track-only cam
    cfg = _base_cfg([_cam("C", "rtsp://c/s", roll=False, track_cats=True)])
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert sum(treats) == 0 and lp.status.rolls == 0
    finally:
        lp.stop()


def test_two_roll_cameras_share_one_cooldown(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://a/s": FrameOutcome(True, True),
                "rtsp://b/s": FrameOutcome(True, True)}
    cfg = _base_cfg([_cam("A", "rtsp://a/s", roll=True),
                     _cam("B", "rtsp://b/s", roll=True)])
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert sum(treats) == 1    # two cameras race, one shared treat dispenser
    finally:
        lp.stop()


def test_cat_recorded_only_on_tracking_camera(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://t/s": FrameOutcome(True, False, labels=("cat",))}
    cfg = _base_cfg([_cam("Tracker", "rtsp://t/s", roll=False, track_cats=True)])
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        last = lp.cats.last()
        assert last is not None and last["camera"] == "Tracker"
        assert lp.cat_present() is True     # any cat-tracking camera sees a cat
    finally:
        lp.stop()


def test_locator_records_dog_when_enabled(monkeypatch, tmp_path):
    # A no-dog household opts into ["cat","dog"] → a dog is recorded as a sighting
    # with its actual label.
    global OUTCOMES
    OUTCOMES = {"rtsp://d/s": FrameOutcome(True, False, labels=("dog",))}
    cfg = _base_cfg([_cam("DogCam", "rtsp://d/s", roll=False, track_cats=True,
                          locator_classes=["cat", "dog"])])
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        last = lp.cats.last()
        assert last is not None and last["label"] == "dog"
        assert lp.cat_present() is True
    finally:
        lp.stop()


def test_dog_not_recorded_by_default(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://d/s": FrameOutcome(True, False, labels=("dog",))}
    cfg = _base_cfg([_cam("DogCam", "rtsp://d/s", roll=False, track_cats=True)])
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert lp.cats.last() is None      # default locator_classes = cat-only
    finally:
        lp.stop()


def test_cats_record_stores_label(tmp_path):
    ct = CatTracker(path=str(tmp_path / "c.log"))
    assert ct.record("Cam", (1, 2, 3, 4), (64, 48), 0.8, label="dog")["label"] == "dog"
    assert ct.record("Cam", (1, 2, 3, 4), (64, 48), 0.8)["label"] == "cat"   # default


def test_cat_not_recorded_when_tracking_disabled(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://n/s": FrameOutcome(True, False, labels=("cat",))}
    cfg = _base_cfg([_cam("NoTrack", "rtsp://n/s", roll=True, track_cats=False)])
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert lp.cats.last() is None and lp.cat_present() is False
    finally:
        lp.stop()


def test_one_camera_failing_does_not_stop_the_other(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://good/s": FrameOutcome(True, True),
                "rtsp://bad/s": CameraError("stream gone")}
    cfg = _base_cfg([_cam("Good", "rtsp://good/s", roll=True),
                     _cam("Bad", "rtsp://bad/s", roll=True)])
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.5)
    try:
        assert sum(treats) == 1           # the healthy camera still rolled
        status = {c["name"]: c for c in lp.cam_status()}
        assert status["Bad"]["last_error"]      # surfaced as failing
        assert status["Good"]["connected"]
    finally:
        lp.stop()


def test_cat_camera_keeps_detecting_during_a_roll_cooldown(monkeypatch, tmp_path):
    # After a roll camera rolls (closing the shared cooldown), a track_cats-only
    # camera must NOT be paused — it has to keep seeing cats. We assert it keeps
    # recording sightings while the (long) cooldown is in effect.
    global OUTCOMES
    OUTCOMES = {"rtsp://roll/s": FrameOutcome(True, True),
                "rtsp://cat/s": FrameOutcome(True, False, labels=("cat",))}
    cfg = _base_cfg(
        [_cam("Roller", "rtsp://roll/s", roll=True, track_cats=False),
         _cam("CatCam", "rtsp://cat/s", roll=False, track_cats=True)],
        pause_during_cooldown=True, cooldown_seconds=600,
    )
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.5)
    try:
        assert sum(treats) == 1                          # the roll happened (cooldown now open)
        # The cat camera kept tracking despite the active cooldown.
        assert lp.cats.last() is not None and lp.cats.last()["camera"] == "CatCam"
    finally:
        lp.stop()


def test_still_cat_scan_records_and_flags(monkeypatch, tmp_path):
    # A sleeping cat makes NO motion (motion=False) but is still a cat. The periodic
    # forced scan must catch it: record a sighting and flag it present (button flash).
    global OUTCOMES
    OUTCOMES = {"rtsp://nap/s": FrameOutcome(False, False, labels=("cat",))}
    cfg = _base_cfg([_cam("Nap", "rtsp://nap/s", roll=False, track_cats=True)],
                    cat_scan_interval=30)
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        last = lp.cats.last()
        assert last is not None and last["camera"] == "Nap"   # still cat recorded
        assert lp.cat_present() is True                        # flashes the button
        assert sum(treats) == 0                                # a still scan never rolls
    finally:
        lp.stop()


def test_still_cat_not_scanned_when_disabled(monkeypatch, tmp_path):
    # cat_scan_interval < 0 = off: a motionless cat is never looked for.
    global OUTCOMES
    OUTCOMES = {"rtsp://nap/s": FrameOutcome(False, False, labels=("cat",))}
    cfg = _base_cfg([_cam("Nap", "rtsp://nap/s", roll=False, track_cats=True)],
                    cat_scan_interval=-1)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert lp.cats.last() is None and lp.cat_present() is False
    finally:
        lp.stop()


def test_always_on_scan_dedupes_still_cat(monkeypatch, tmp_path):
    # cat_scan_interval == 0 = always-on: many scans of the same motionless cat must
    # record ONE sighting (rising edge), not one per scan.
    global OUTCOMES
    OUTCOMES = {"rtsp://nap/s": FrameOutcome(False, False, labels=("cat",))}
    cfg = _base_cfg([_cam("Nap", "rtsp://nap/s", roll=False, track_cats=True)],
                    cat_scan_interval=0)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.5)
    try:
        assert len(lp.cats.recent()) == 1
    finally:
        lp.stop()


def test_present_cameras_lists_each_room_with_a_cat(monkeypatch, tmp_path):
    # Two rooms each with a (still) cat → both appear in the rotation list.
    global OUTCOMES
    OUTCOMES = {"rtsp://a/s": FrameOutcome(False, False, labels=("cat",)),
                "rtsp://b/s": FrameOutcome(False, False, labels=("cat",))}
    cfg = _base_cfg([_cam("RoomA", "rtsp://a/s", roll=False, track_cats=True),
                     _cam("RoomB", "rtsp://b/s", roll=False, track_cats=True)],
                    cat_scan_interval=0)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert sorted(lp.cats_present_cameras()) == ["RoomA", "RoomB"]
    finally:
        lp.stop()


def test_still_person_on_scan_does_not_roll(monkeypatch, tmp_path):
    # A forced still-cat scan that happens to see a motionless person must NOT roll —
    # rolling stays gated on real motion.
    global OUTCOMES
    OUTCOMES = {"rtsp://p/s": FrameOutcome(False, True)}    # person, no motion
    cfg = _base_cfg([_cam("Both", "rtsp://p/s", roll=True, track_cats=True)],
                    cat_scan_interval=0)
    lp, treats = _run_loop(cfg, monkeypatch, tmp_path)
    try:
        assert sum(treats) == 0 and lp.status.rolls == 0
    finally:
        lp.stop()


def test_show_cat_boost_forces_detection_when_scanning_off(monkeypatch, tmp_path):
    # Periodic scanning OFF: a still cat is normally unseen. A "Show cat" boost must
    # force continuous detection so the live feed boxes the cat — and it gets recorded.
    global OUTCOMES
    OUTCOMES = {"rtsp://nap/s": FrameOutcome(False, False, labels=("cat",))}
    cfg = _base_cfg([_cam("Nap", "rtsp://nap/s", roll=False, track_cats=True)],
                    cat_scan_interval=-1)        # off (motion only)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.2)
    try:
        assert lp.cats.last() is None            # scanning off → still cat unseen
        assert lp.boost_detection("Ghost") is False   # not a watched camera
        assert lp.boost_detection("Nap") is True
        time.sleep(0.3)
        last = lp.cats.last()
        assert last is not None and last["camera"] == "Nap"   # boost found the still cat
        assert lp.cat_present() is True
    finally:
        lp.stop()


# ---- round-robin ----------------------------------------------------------
def _rr_cams():
    return [_cam("A", "rtsp://a/s", roll=False, track_cats=True),
            _cam("B", "rtsp://b/s", roll=False, track_cats=True),
            _cam("C", "rtsp://c/s", roll=False, track_cats=True)]


def _idle_outcomes():
    return {s: FrameOutcome(False, False) for s in ("rtsp://a/s", "rtsp://b/s", "rtsp://c/s")}


def test_round_robin_rotates_and_rests(monkeypatch, tmp_path):
    # 3 cameras, only 1 detecting at a time: every camera still gets its turn
    # (rotation reaches all), and at any settled moment some cameras are resting.
    global OUTCOMES, CALLS
    OUTCOMES, CALLS = _idle_outcomes(), {}
    cfg = _base_cfg(_rr_cams(), round_robin=True, round_robin_size=1,
                    round_robin_interval=0.4)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=1.6)
    try:
        assert all(CALLS.get(s, 0) > 0 for s in OUTCOMES), CALLS   # all got a turn
        st = {c["name"]: c for c in lp.cam_status()}
        assert sum(1 for c in st.values() if c["resting"]) >= 1    # someone is resting
        assert any(d.release_count > 0 for d in lp._detectors.values())   # rested → released
    finally:
        lp.stop()


def test_round_robin_off_keeps_all_active(monkeypatch, tmp_path):
    global OUTCOMES, CALLS
    OUTCOMES, CALLS = _idle_outcomes(), {}
    cfg = _base_cfg(_rr_cams(), round_robin=False)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.5)
    try:
        st = {c["name"]: c for c in lp.cam_status()}
        assert not any(c["resting"] for c in st.values())   # nobody rests
        assert all(CALLS.get(s, 0) > 0 for s in OUTCOMES)
    finally:
        lp.stop()


def test_always_watch_never_rests(monkeypatch, tmp_path):
    global OUTCOMES, CALLS
    OUTCOMES, CALLS = _idle_outcomes(), {}
    cams = _rr_cams()
    cams[0]["always_watch"] = True       # "A" is exempt from rotation
    cfg = _base_cfg(cams, round_robin=True, round_robin_size=1, round_robin_interval=0.4)
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=1.0)
    try:
        st = {c["name"]: c for c in lp.cam_status()}
        assert st["A"]["always_watch"] is True and st["A"]["resting"] is False
    finally:
        lp.stop()


def test_viewed_camera_stays_active(monkeypatch, tmp_path):
    # A camera the GUI is streaming must not rest, even when it's not its rotation turn.
    global OUTCOMES, CALLS
    OUTCOMES, CALLS = _idle_outcomes(), {}
    cfg = _base_cfg(_rr_cams(), round_robin=True, round_robin_size=1,
                    round_robin_interval=5.0)   # long interval → C would otherwise rest
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.4)
    try:
        lp.note_viewing("C")              # pin C (as the live stream would, each frame)
        time.sleep(0.6)
        st = {c["name"]: c for c in lp.cam_status()}
        assert st["C"]["resting"] is False
    finally:
        lp.stop()


def test_stop_releases_all_detectors(monkeypatch, tmp_path):
    global OUTCOMES
    OUTCOMES = {"rtsp://a/s": FrameOutcome(False, False),
                "rtsp://b/s": FrameOutcome(False, False)}
    cfg = _base_cfg([_cam("A", "rtsp://a/s"), _cam("B", "rtsp://b/s")])
    lp, _ = _run_loop(cfg, monkeypatch, tmp_path, seconds=0.2)
    dets = list(lp._detectors.values())
    assert len(dets) == 2
    lp.stop()
    assert all(d.released for d in dets)
    assert lp._detectors == {} and lp._threads == []


# ---- webapp: per-camera endpoints -----------------------------------------
def _client(tmp_path, monkeypatch):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    return create_app().test_client(), cfgfile


def test_saved_camera_round_trips_full_settings(tmp_path, monkeypatch):
    c, cfgfile = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={
        "name": "Kitchen", "url": "rtsp://1/s", "password": "sec",
        "roll": False, "track_cats": True, "model": "yolo11n",
        "scan_fps": 5, "roi": [10, 20, 30, 40]})
    cam = c.get("/api/cameras/saved").get_json()[0]
    assert cam["roll"] is False and cam["track_cats"] is True
    assert cam["model"] == "yolo11n" and cam["scan_fps"] == 5
    assert cam["roi"] == [10, 20, 30, 40]
    assert "password" not in cam and cam["has_password"] is True


def test_active_cameras_validates_and_persists(tmp_path, monkeypatch):
    c, cfgfile = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={"name": "Kitchen", "url": "rtsp://1/s"})
    r = c.post("/api/cameras/active", json={"names": ["Kitchen", "Ghost"]}).get_json()
    assert r["active_cameras"] == ["Kitchen"]            # Ghost dropped
    assert config_mod.load(cfgfile).active_cameras == ["Kitchen"]


def test_status_exposes_per_camera_list(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    assert c.get("/api/status").get_json()["cameras"] == []   # not running


def test_cats_endpoint_exposes_present_cameras(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    body = c.get("/api/cats").get_json()
    assert body["cameras"] == []        # nothing running → no cat-cams present
    assert "present" in body


def test_cats_boost_endpoint(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    # Nothing running → the camera isn't watched, so the boost is a no-op.
    assert c.post("/api/cats/boost", json={"camera": "Kitchen"}).get_json()["ok"] is False
    assert c.post("/api/cats/boost", json={}).get_json()["ok"] is False


def test_round_robin_config_round_trips(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    c.post("/api/config", json={"round_robin": True, "round_robin_size": 3,
                                "round_robin_interval": 20})
    cfg = c.get("/api/config").get_json()
    assert cfg["round_robin"] is True
    assert cfg["round_robin_size"] == 3 and cfg["round_robin_interval"] == 20


def test_camera_locator_and_always_watch_round_trip(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={
        "name": "K", "url": "rtsp://1/s", "cat_scan_tiling": "4x4",
        "cat_scan_imgsz": 960, "always_watch": True})
    cam = c.get("/api/cameras/saved").get_json()[0]
    assert cam["cat_scan_tiling"] == "4x4" and cam["cat_scan_imgsz"] == 960
    assert cam["always_watch"] is True


def test_cat_confidence_and_locator_classes_round_trip(tmp_path, monkeypatch):
    assert Config().cat_confidence == 0.5 and Config().locator_classes == ["cat"]
    c, _ = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://1/s",
                                       "cat_confidence": 0.35, "locator_classes": ["cat", "dog"]})
    cam = c.get("/api/cameras/saved").get_json()[0]
    assert cam["cat_confidence"] == 0.35 and cam["locator_classes"] == ["cat", "dog"]


def test_cat_scan_interval_default_and_coercion():
    assert Config().cat_scan_interval == 30.0
    # arrives as a string from the form → coerced to float (0 = always, -1 = off)
    assert config_mod._coerce("0", 30.0) == 0.0
    assert config_mod._coerce("-1", 30.0) == -1.0
