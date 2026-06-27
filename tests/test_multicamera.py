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

    def read_and_detect(self, detect=True):
        time.sleep(0.003)
        outcome = OUTCOMES.get(self.source, FrameOutcome(False, False))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

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


OUTCOMES = {}     # source -> FrameOutcome | Exception


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
        "roll": False, "track_cats": True, "model": "mobilenet_ssd",
        "scan_fps": 5, "roi": [10, 20, 30, 40]})
    cam = c.get("/api/cameras/saved").get_json()[0]
    assert cam["roll"] is False and cam["track_cats"] is True
    assert cam["model"] == "mobilenet_ssd" and cam["scan_fps"] == 5
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
