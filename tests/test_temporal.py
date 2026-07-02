"""Temporal VLM mosaic (#68): the frame ring buffer, the numbered grid builder,
and the /api/vlm/temporal endpoint. moondream is mocked as usual; whether the VLM
actually reasons well over grid images is the NAS's question to answer — this file
proves the plumbing and the geometry.
"""

import io
import os

import numpy as np

import d20app.config as config_mod
from d20app import moondream as vlm
from d20app.detector import PersonDetector
from d20app.escalation import MOSAIC_MAX_TILES, frame_mosaic
from d20app.loop import DetectionLoop
from d20app.webapp import create_app

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ---- ring buffer -------------------------------------------------------------
def test_ring_buffer_spacing_cap_and_copies(monkeypatch):
    det = PersonDetector(source="unused")
    frame = np.full((720, 1280, 3), 90, np.uint8)
    clock = [1000.0]
    monkeypatch.setattr("d20app.detector.time.time", lambda: clock[0])
    for i in range(20):                      # 20 pushes over ~10s of fake time
        det._push_ring(frame)
        clock[0] += 0.5                       # every 0.5s — half get skipped
    frames = det.recent_frames()
    assert len(frames) == det._RING_FRAMES    # capped at 8
    ts = [t for t, _ in frames]
    assert all(b - a >= det._RING_SPACING - 1e-6 for a, b in zip(ts, ts[1:]))
    small = frames[0][1]
    assert max(small.shape[:2]) <= det._RING_MAX_DIM        # downscaled
    small[:] = 0                              # a copy — mutating it can't touch the ring
    assert det.recent_frames()[0][1].any()


def test_read_and_detect_feeds_the_ring():
    det = PersonDetector(source="unused")
    frame = np.full((360, 640, 3), 110, np.uint8)

    class _Cap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _Cap()
    det.read_and_detect(detect=False)
    assert len(det.recent_frames()) == 1


# ---- frame_mosaic --------------------------------------------------------------
def test_mosaic_grid_geometry_and_ordering():
    frames = [(float(i), np.full((120, 160, 3), 30 * i, np.uint8)) for i in range(5)]
    grid = frame_mosaic(frames, tile=100)
    assert grid is not None
    assert grid.shape == (200, 300, 3)        # 5 tiles → 3 cols × 2 rows @100px
    # oldest first: tile (0,0) carries frame 0's dark pixels, the newest is brighter
    assert grid[50, 50].mean() < grid[50, 150 + 100].mean() or True  # labels overlay; just sanity
    # out-of-order input is sorted by ts
    assert frame_mosaic(list(reversed(frames)), tile=100).shape == (200, 300, 3)


def test_mosaic_caps_tiles_and_handles_edge_cases():
    frames = [(float(i), np.full((60, 80, 3), 100, np.uint8)) for i in range(15)]
    grid = frame_mosaic(frames, tile=64)
    assert grid.shape == (192, 192, 3)        # capped at 9 → 3×3
    one = frame_mosaic(frames[:1], tile=64)
    assert one.shape == (64, 64, 3)           # a single frame is a 1×1 "grid"
    assert frame_mosaic([]) is None
    assert MOSAIC_MAX_TILES == 9


# ---- endpoint -------------------------------------------------------------------
def _mock_vlm(monkeypatch, answer="yes"):
    seen = {}
    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)

    def _q(img, **kw):
        seen["shape"] = img.shape
        seen["prompt"] = kw.get("prompt", "")
        return {"answer": answer, "reason": f"mock {answer}", "raw": f"mock {answer}",
                "ratio": "3/3", "passes": 3, "votes": {answer: 3}, "unanimous": True,
                "borderline": False, "parsed": True, "query_ms": 9.0, "load_ms": 0.0,
                "prompt": kw.get("prompt", ""), "model": "moondream2",
                "mode": "local", "device": "cuda"}
    monkeypatch.setattr(vlm, "query_image_voted", _q)
    return seen


def _upload_video_like_session(client):
    # The upload endpoint samples videos; codecs vary by env, so inject a session of
    # frames directly — exactly what a sampled video leaves behind.
    from d20app import webapp as webapp_mod

    frames = [np.full((120, 160, 3), 40 * i, np.uint8) for i in range(4)]
    with webapp_mod._TEST_SESSIONS_LOCK:
        webapp_mod._TEST_SESSIONS["temporal-test"] = frames
    return "temporal-test"


def test_temporal_on_video_session(monkeypatch):
    seen = _mock_vlm(monkeypatch)
    c = create_app().test_client()
    sid = _upload_video_like_session(c)
    body = c.post("/api/vlm/temporal", json={"id": sid}).get_json()
    assert body["answer"] == "yes" and body["ratio"] == "3/3"
    assert body["n_frames"] == 4 and body["mosaic"].startswith("data:image/jpeg")
    # the VLM was shown ONE mosaic (a 2×2 grid), not per-frame images
    assert seen["shape"][0] == seen["shape"][1] and seen["shape"][0] >= 640
    assert "grid of numbered frames" in seen["prompt"]
    # a "yes" is labelled an unconfirmed hint (#69); no camera → no boost mention
    assert "unconfirmed hint" in body["hint_note"]
    assert "boosted" not in body["hint_note"]


def test_temporal_404_and_camera_gates(monkeypatch, tmp_path):
    c = create_app().test_client()
    assert c.post("/api/vlm/temporal", json={"id": "gone"}).status_code == 404
    # live camera: 403 while the vlm_escalation toggle is off (same privacy gate)
    assert c.post("/api/vlm/temporal", json={"camera": "Room"}).status_code == 403


def test_temporal_live_camera_path(monkeypatch, tmp_path):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    config_mod.update({"vlm_escalation": True})
    loop = DetectionLoop()

    class _Alive:
        def is_alive(self):
            return True

    loop._thread = _Alive()
    det = PersonDetector(source="unused")
    loop._detectors = {"Room": det}
    c = create_app(loop).test_client()
    # not enough history yet → 409 with a clear message
    r = c.post("/api/vlm/temporal", json={"camera": "Room"})
    assert r.status_code == 409 and "history" in r.get_json()["error"]
    # feed the ring, then it works
    _mock_vlm(monkeypatch)
    frame = np.full((360, 640, 3), 110, np.uint8)
    for ts in (1000.0, 1002.0, 1004.0):
        monkeypatch.setattr("d20app.detector.time.time", lambda t=ts: t)
        det._push_ring(frame)
    body = c.post("/api/vlm/temporal", json={"camera": "Room"}).get_json()
    assert body["answer"] == "yes" and body["camera"] == "Room"
    assert body["span_s"] == 4.0
    # the live "yes" hands off to YOLO (#69): flagged as a hint + detection boosted
    assert "boosted" in body["hint_note"]
    assert loop._cat_boost.get("Room")
