"""Semantic zones + sighting heat maps + the time-of-day prior (#68)."""

import time

import numpy as np

import d20app.config as config_mod
from d20app.cats import CatTracker, box_in_exit_zone, zone_for
from d20app.detector import PersonDetector
from d20app.heatmap import render_heatmap
from d20app.loop import DetectionLoop
from d20app.webapp import create_app

ZONES = [{"name": "the couch", "box": [100, 100, 200, 100], "exit": False},
         {"name": "kitchen door", "box": [500, 0, 100, 300], "exit": True}]


# ---- zone_for / exit zones ---------------------------------------------------
def test_zone_for_names_the_containing_zone():
    assert zone_for((150, 120, 190, 160), ZONES) == "the couch"
    assert zone_for((520, 50, 560, 90), ZONES) == "kitchen door"
    assert zone_for((0, 0, 20, 20), ZONES) == ""            # outside every zone
    assert zone_for((150, 120, 190, 160), []) == ""


def test_zone_for_shifts_roi_coords_back_to_frame_space():
    # Zones are drawn on the FULL frame; detection boxes live in ROI-crop coords.
    # A box at crop (10,20) under roi origin (140,100) is at frame (150,120).
    assert zone_for((10, 20, 50, 60), ZONES, roi=[140, 100, 300, 200]) == "the couch"
    assert zone_for((10, 20, 50, 60), ZONES, roi=None) == ""


def test_box_in_exit_zone_only_matches_exits():
    assert box_in_exit_zone((520, 50, 560, 90), ZONES) is True     # the doorway
    assert box_in_exit_zone((150, 120, 190, 160), ZONES) is False  # the couch
    assert box_in_exit_zone((520, 50, 560, 90), []) is False


def test_record_stores_zone_and_old_records_load(tmp_path):
    path = str(tmp_path / "cats.log")
    t = CatTracker(directory=path)
    t.record("Cam", (150, 120, 190, 160), (640, 360), 0.8, zone="the couch")
    t.record("Cam", (10, 10, 30, 30), (640, 360), 0.7)             # no zone
    again = CatTracker(directory=path)
    zones = [s.get("zone") for s in again.recent()]
    assert zones == [None, "the couch"]


# ---- the time-of-day prior ---------------------------------------------------
def _ts_at_hour(hour):
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, 30, 0,
                        lt.tm_wday, lt.tm_yday, -1))


def test_by_hour_and_likely_cameras(tmp_path):
    t = CatTracker(directory=str(tmp_path / "cats"))
    for _ in range(3):
        t.record("Kitchen", (1, 1, 9, 9), (100, 100), 0.5, ts=_ts_at_hour(15))
    t.record("Bedroom", (1, 1, 9, 9), (100, 100), 0.5, ts=_ts_at_hour(15))
    t.record("Bedroom", (1, 1, 9, 9), (100, 100), 0.5, ts=_ts_at_hour(3))
    assert t.by_hour()[15] == 4 and t.by_hour("Kitchen")[15] == 3
    ranked = t.likely_cameras(hour=15)
    assert ranked[0] == ("Kitchen", 3) and ("Bedroom", 1) in ranked
    assert t.likely_cameras(hour=3) == [("Bedroom", 1)]
    # ±1-hour window wraps midnight
    assert t.likely_cameras(hour=4) == [("Bedroom", 1)]
    assert t.likely_cameras(hour=9) == []


# ---- heat map rendering -------------------------------------------------------
def test_render_heatmap_tints_hot_spots_only():
    frame = np.full((360, 640, 3), 110, np.uint8)
    boxes = [(280, 100, 360, 180)] * 5 + [(50, 300, 90, 340)]
    img = render_heatmap(frame, boxes)
    assert img is not None
    assert not np.array_equal(img[140, 320], frame[140, 320])   # hot spot tinted
    assert np.array_equal(img[20, 600], frame[20, 600])         # cold corner untouched
    # no usable boxes → None (degenerate + out-of-frame are clipped away)
    assert render_heatmap(frame, []) is None
    assert render_heatmap(frame, [(0, 0, 1, 1), (700, 400, 800, 500)]) is None


# ---- endpoints ----------------------------------------------------------------
def _running_loop_with_fake_camera(cam="Room"):
    loop = DetectionLoop()

    class _Alive:
        def is_alive(self):
            return True

    loop._thread = _Alive()
    det = PersonDetector(source="unused")
    loop._detectors = {cam: det}
    return loop, det


def test_api_cats_exposes_prior(tmp_path):
    loop, _ = _running_loop_with_fake_camera()
    loop.cats = CatTracker(directory=str(tmp_path / "cats"))
    loop.cats.record("Kitchen", (1, 1, 9, 9), (100, 100), 0.5)
    body = create_app(loop).test_client().get("/api/cats").get_json()
    assert len(body["by_hour"]) == 24 and sum(body["by_hour"]) == 1
    assert body["likely"][0]["camera"] == "Kitchen"


def test_api_heatmap_serves_jpeg_and_degrades(tmp_path):
    loop, det = _running_loop_with_fake_camera()
    loop.cats = CatTracker(directory=str(tmp_path / "cats"))
    c = create_app(loop).test_client()
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    # no sightings yet → 404
    assert c.get("/api/cats/heatmap?camera=Room").status_code == 404
    loop.cats.record("Room", (280, 100, 360, 180), (640, 360), 0.8)
    r = c.get("/api/cats/heatmap?camera=Room")
    assert r.status_code == 200 and r.data[:2] == b"\xff\xd8"
    assert c.get("/api/cats/heatmap?camera=Nope").status_code == 404
    assert create_app().test_client().get(
        "/api/cats/heatmap?camera=Room").status_code == 409


def test_exit_zone_suppresses_probable(monkeypatch, tmp_path):
    # Trail ends interior by the frame-edge test, but inside a DOORWAY zone —
    # the cat may have left, so no "probable location" claim (#68).
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    config_mod.update({"vlm_escalation": True, "cameras": [
        {"name": "Room", "url": "x",
         "zones": [{"name": "door", "box": [250, 80, 150, 120], "exit": True}]}]})
    loop, det = _running_loop_with_fake_camera()
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    base = np.full((360, 640), 110, np.uint8)
    det._trail.update(base, False)
    mover = base.copy()
    mover[100:180, 280:360] = 220             # ends inside the doorway zone
    det._trail.update(mover, True)
    c = create_app(loop).test_client()
    body = c.post("/api/vlm/escalate",
                  json={"camera": "Room", "use_vlm": False}).get_json()
    assert body["found"] is False and body["probable"] is None


def test_camera_zones_round_trip_via_saved_cameras(monkeypatch, tmp_path):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/cameras/saved", json={
        "name": "Kitchen", "url": "rtsp://x", "zones": ZONES})
    cams = c.get("/api/cameras/saved").get_json()
    kitchen = next(x for x in cams if x["name"] == "Kitchen")
    assert kitchen["zones"] == ZONES
    # an old camera without zones coerces to []
    c.post("/api/cameras/saved", json={"name": "Garage", "url": "rtsp://y"})
    cams = c.get("/api/cameras/saved").get_json()
    assert next(x for x in cams if x["name"] == "Garage")["zones"] == []
