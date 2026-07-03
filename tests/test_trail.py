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


# ---- 0.39.0: flood guard, blob hygiene, route path, live overlay --------------
def test_global_lighting_change_resets_null_instead_of_flooding():
    # The first NAS screenshot: an exposure shift made the WHOLE frame differ from
    # the null and the room was flood-stamped green. Now: a global change re-adopts
    # the scene as the new null and stamps NOTHING.
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    flood = np.full_like(BASE, 200)              # everything got brighter
    t.update(flood, True, now=5.0)
    assert not t.has_trail() and t.endpoint(now=5.1) is None
    # the flooded scene became the new null: a real blob on it stamps normally
    blob = flood.copy()
    blob[100:180, 280:360] = 60
    t.update(blob, True, now=6.0)
    ep = t.endpoint(now=6.1)
    assert ep and 260 <= ep["box"][0] <= 290     # just the blob, not the room


def test_oversized_blob_is_not_an_animal():
    # A single blob covering ~30% of the frame (below the global-change guard but
    # far bigger than a cat) is filtered by the silhouette-size ceiling.
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    big = BASE.copy()
    big[:, :192] = 220                           # 30% of a 640-wide frame
    t.update(big, True, now=5.0)
    assert not t.has_trail()


def test_route_path_is_recorded_and_rendered_with_legend():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    t.update(_blob(280, 100, 360, 180), True, now=3.0)
    assert len(t._path) == 2                     # one centroid per stamped frame
    frame = np.full((360, 640, 3), 110, np.uint8)
    out = t.render(frame, now=3.5)
    assert out is not None and out.shape == frame.shape
    # the legend's black backing plate sits at the bottom-left
    assert out[352, 10].tolist() != frame[352, 10].tolist()


def test_live_jpeg_trail_overlay():
    det = PersonDetector(source="unused")
    frame = np.full((360, 640, 3), 110, np.uint8)
    det._publish_frame(frame)
    plain = det.live_jpeg()
    assert plain and det.live_jpeg(trail=True) == plain   # no trail yet → plain feed
    gray = frame[:, :, 0].copy()
    det._trail.update(gray, False)
    blob = gray.copy()
    blob[100:180, 280:360] = 220
    det._trail.update(blob, True)
    overlaid = det.live_jpeg(trail=True)
    assert overlaid and overlaid != det.live_jpeg()       # ribbon + legend drawn


# ---- 0.41.0: people don't leave trails ----------------------------------------
def test_exclude_boxes_blank_person_from_the_stamp():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    # person moves on the left, cat on the right — same frame
    f = _blob(80, 100, 160, 180)
    f[120:200, 400:480] = 200
    t.update(f, True, now=1.0, exclude=[(70, 90, 170, 190)])
    ep = t.endpoint(now=1.1)
    assert ep and ep["box"][0] >= 380              # only the cat stamped
    assert len(t._path) == 1


def test_excluded_person_only_frame_stamps_nothing():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0,
             exclude=[(70, 90, 170, 190)])
    assert not t.has_trail() and t.endpoint(now=1.1) is None


def test_erase_recent_scrubs_the_person_keeps_the_cat():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(400, 120, 480, 200), True, now=1.0)   # the cat, a while ago
    t.update(_blob(80, 100, 160, 180), True, now=8.0)    # the person, just now
    # the net names the mover: scrub its stamps (made < ERASE_WINDOW_SECS ago)
    t.erase_recent([(70, 90, 170, 190)], now=8.1)
    ep = t.endpoint(now=8.2)
    assert ep and ep["box"][0] >= 380              # endpoint falls back to the cat
    assert t.has_trail()                           # the cat's older trail survives
    assert len(t._path) == 1 and t._path[0][1] * t._scale >= 380
    # old stamps inside the box are NOT scrubbed (only recent ones)
    t2 = TrailTracker()
    t2.update(BASE, False, now=0.0)
    t2.update(_blob(80, 100, 160, 180), True, now=1.0)
    t2.erase_recent([(70, 90, 170, 190)], now=20.0)      # stamp is 19 s old
    assert t2.has_trail()


def test_erase_recent_person_only_episode_resets():
    t = TrailTracker()
    t.update(BASE, False, now=0.0)
    t.update(_blob(80, 100, 160, 180), True, now=1.0)
    t.erase_recent([(70, 90, 170, 190)], now=1.1)
    assert not t.has_trail() and t.endpoint(now=1.2) is None


def test_read_and_detect_person_never_trails():
    # Wiring: the net reports a person → the entry-frame stamp is erased and the
    # following frames are excluded, so human movement leaves NO trail.
    det = PersonDetector(source="unused")
    frames = [np.full((360, 640, 3), 110, np.uint8)]

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: [
        ("person", 0.9, (270, 90, 370, 190))]
    det.read_and_detect(detect=False)              # adopts the null frame
    for x in (280, 300, 320):                      # the person's arm keeps moving
        f = np.full((360, 640, 3), 110, np.uint8)
        f[100:180, x:x + 60] = 220
        frames.append(f)
        det.read_and_detect(detect=True)
    assert not det._trail.has_trail()
    assert det.trail_endpoint() is None


def test_read_and_detect_cat_still_trails_next_to_person():
    det = PersonDetector(source="unused")
    frames = [np.full((360, 640, 3), 110, np.uint8)]

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: [
        ("person", 0.9, (60, 90, 180, 190))]
    det.read_and_detect(detect=False)
    f = np.full((360, 640, 3), 110, np.uint8)
    f[100:180, 80:160] = 220                       # the person's arm
    f[120:200, 400:480] = 200                      # the cat, well clear of the box
    frames.append(f)
    det.read_and_detect(detect=True)               # person box known from this run
    frames.append(f.copy())
    det.read_and_detect(detect=False)              # trail keeps stamping the cat
    ep = det.trail_endpoint()
    assert ep is not None and ep["box"][0] >= 380  # the cat's spot, not the arm


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
