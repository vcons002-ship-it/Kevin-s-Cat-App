"""The cat trail (#67): null-frame silhouettes, recency colouring, endpoint
targeting, and the "probable location" tier. TrailTracker is exercised with
synthetic frames and controlled clocks; the live escalate path runs against a real
DetectionLoop carrying a real PersonDetector whose frames are injected — no camera,
net stubbed where irrelevant. What only the NAS shows: real-scene trail quality
(ghosting, lighting drift) — flagged in the PR checklist.
"""

import numpy as np
import pytest

import d20app.config as config_mod
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop
from d20app.trail import (EPISODE_GAP_SECS, NULL_REFRESH_SECS, TrailTracker)
from d20app.webapp import create_app

BASE = np.full((360, 640), 110, np.uint8)


def _blob(x1, y1, x2, y2, value=220):
    f = BASE.copy()
    f[y1:y2, x1:x2] = value
    return f


# ---- TrailTracker -----------------------------------------------------------
def test_first_frame_adopts_null_no_trail():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    assert t.endpoint(now=0.1) is None and not t.has_trail()
    assert t.render(np.zeros((360, 640, 3), np.uint8), now=0.1) is None


def test_moving_blob_paints_trail_and_endpoint():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    t.update(_blob(280, 100, 360, 180), True, now=3.0)
    ep = t.endpoint(now=3.5)
    assert ep and ep["interior"] and ep["age_s"] == 0.5
    x1, y1, x2, y2 = ep["box"]
    assert 260 <= x1 <= 290 and x2 >= 350        # the newest blob, frame coords
    assert t.has_trail()


def test_render_colours_old_blue_new_red():
    import cv2

    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    t.update(_blob(280, 100, 360, 180), True, now=3.0)
    img = t.render(cv2.cvtColor(BASE, cv2.COLOR_GRAY2BGR), now=3.5)
    old_px = img[140, 120].astype(int)            # first position → blue-dominant
    new_px = img[140, 320].astype(int)            # latest position → red-dominant
    assert old_px[0] > old_px[2] and new_px[2] > new_px[0]
    # the endpoint marker is drawn (pure red rectangle pixels exist near the box)
    assert (img[..., 2] == 255).any()


def test_edge_endpoint_not_interior():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(0, 0, 60, 60), True, now=1.0)   # touching the corner
    assert t.endpoint(now=1.1)["interior"] is False


def test_new_episode_resets_the_trail():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    # long stillness (null re-adopts along the way), then a fresh mover
    t.update(_blob(80, 100, 160, 180), False, now=2.0)
    t.update(_blob(80, 100, 160, 180), False, now=2.0 + NULL_REFRESH_SECS + 1)
    t.update(_blob(80, 100, 160, 180).copy(), False, now=20.0)
    later = 20.0 + EPISODE_GAP_SECS + 5
    f = _blob(80, 100, 160, 180)
    f[200:260, 400:470] = 30
    t.update(f, True, now=later)
    ep = t.endpoint(now=later + 0.1)
    assert ep["box"][0] >= 390                     # only the new mover remains


def test_stale_endpoint_expires():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    assert t.endpoint(now=2.0) is not None
    assert t.endpoint(now=1.0 + 601.0) is None     # past ENDPOINT_FRESH_SECS


def test_null_refresh_absorbs_settled_scene():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    parked = _blob(80, 100, 160, 180)
    t.update(parked, True, now=1.0)
    # scene goes still WITH the blob present; after the refresh window the null
    # re-adopts it, so an identical frame no longer paints anything new
    t.update(parked, False, now=2.0)
    t.update(parked, False, now=2.0 + NULL_REFRESH_SECS + 1)
    before = t.endpoint(now=10.0)["ts"]
    t.update(parked, True, now=10.0)               # "motion" but no diff vs new null
    assert t.endpoint(now=10.1)["ts"] == before    # nothing new painted


# ---- live wiring: detector + escalate probable tier + /api/trail -------------
def _running_loop_with_fake_camera(monkeypatch, cam="Room"):
    """A real DetectionLoop that LOOKS running and carries a real PersonDetector
    with an injected live frame — the pattern for exercising live-camera paths
    without a camera."""
    loop = DetectionLoop()

    class _Alive:
        def is_alive(self):
            return True

    loop._thread = _Alive()
    det = PersonDetector(source="unused")
    loop._detectors = {cam: det}
    return loop, det


def _cfg_isolated(monkeypatch, tmp_path, **values):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    if values:
        config_mod.update(values)


def test_escalate_probable_when_trail_ends_interior(monkeypatch, tmp_path):
    _cfg_isolated(monkeypatch, tmp_path, vlm_escalation=True)
    loop, det = _running_loop_with_fake_camera(monkeypatch)
    # a cat walked in and settled mid-frame; frames injected straight into the
    # detector's live buffer + trail (what read_and_detect would have done)
    frame = np.full((360, 640, 3), 110, np.uint8)
    det._publish_frame(frame)
    det._trail.update(BASE, False)
    det._trail.update(_blob(280, 100, 360, 180), True)
    c = create_app(loop).test_client()
    body = c.post("/api/vlm/escalate",
                  json={"camera": "Room", "use_vlm": False}).get_json()
    assert body["found"] is False                  # blank frame: YOLO finds nothing
    assert body["probable"] and "probable location" in body["probable"]["note"]
    px1 = body["probable"]["box"][0]
    assert 260 <= px1 <= 300                       # the trail endpoint
    assert body["trail"] and body["trail"].startswith("data:image/jpeg")
    # the endpoint fed the ladder as a hint too
    assert any(h[0] >= 200 for h in body["hints_used"])


def test_escalate_no_probable_when_endpoint_at_edge(monkeypatch, tmp_path):
    _cfg_isolated(monkeypatch, tmp_path, vlm_escalation=True)
    loop, det = _running_loop_with_fake_camera(monkeypatch)
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    det._trail.update(BASE, False)
    det._trail.update(_blob(0, 0, 60, 60), True)   # exited toward the corner
    c = create_app(loop).test_client()
    body = c.post("/api/vlm/escalate",
                  json={"camera": "Room", "use_vlm": False}).get_json()
    assert body["found"] is False and body["probable"] is None


def test_api_trail_serves_jpeg_and_degrades(monkeypatch, tmp_path):
    loop, det = _running_loop_with_fake_camera(monkeypatch)
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    c = create_app(loop).test_client()
    # no motion yet → 404 with a clear message
    r = c.get("/api/trail?camera=Room")
    assert r.status_code == 404 and "moved" in r.get_json()["error"]
    # after motion → a real JPEG
    det._trail.update(BASE, False)
    det._trail.update(_blob(280, 100, 360, 180), True)
    r2 = c.get("/api/trail?camera=Room")
    assert r2.status_code == 200 and r2.data[:2] == b"\xff\xd8"
    # unknown camera → 404; loop not running → 409
    assert c.get("/api/trail?camera=Nope").status_code == 404
    assert create_app().test_client().get("/api/trail?camera=Room").status_code == 409


def test_read_and_detect_feeds_the_trail():
    # the loop path really updates the trail (wiring, not just the class)
    det = PersonDetector(source="unused")
    frames = [np.full((360, 640, 3), 110, np.uint8)]

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: []
    det.read_and_detect(detect=False)              # first: adopts the null frame
    moving = np.full((360, 640, 3), 110, np.uint8)
    moving[100:180, 280:360] = 220
    frames.append(moving)
    det.read_and_detect(detect=True)
    assert det.trail_endpoint() is not None
    assert det.trail_endpoint()["box"][0] >= 250
