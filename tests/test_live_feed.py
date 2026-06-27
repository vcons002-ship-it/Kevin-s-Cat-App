"""Live detection feed: detector.live_jpeg() and the /api/stream MJPEG route."""

import time

import numpy as np

import d20app.config as config_mod
from d20app.detector import PersonDetector
from d20app.webapp import create_app


def _jpeg_ok(buf) -> bool:
    return isinstance(buf, bytes) and buf[:2] == b"\xff\xd8" and buf[-2:] == b"\xff\xd9"


def test_live_jpeg_none_until_a_frame_is_read():
    det = PersonDetector(source="unused")
    assert det.live_jpeg() is None


def test_live_jpeg_encodes_the_latest_frame_with_fresh_boxes():
    det = PersonDetector(source="unused", confidence=0.4)
    det._live_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    det._last_boxes = [("person", 0.95, (5, 5, 30, 40))]
    det._live_boxes_at = time.monotonic()       # fresh → boxes drawn
    assert _jpeg_ok(det.live_jpeg())


def test_live_jpeg_drops_stale_boxes_but_still_streams_the_frame():
    det = PersonDetector(source="unused", confidence=0.4)
    det._live_frame = np.zeros((48, 64, 3), dtype=np.uint8)
    det._last_boxes = [("person", 0.95, (5, 5, 30, 40))]
    det._live_boxes_at = time.monotonic() - (det._LIVE_BOX_TTL + 1)   # expired
    # A person who left shouldn't leave a box hanging, but the feed keeps going.
    fresh = det.live_jpeg()
    assert _jpeg_ok(fresh)


def test_stream_returns_409_when_not_running(tmp_path, monkeypatch):
    c = create_app().test_client()
    r = c.get("/api/stream")
    assert r.status_code == 409


def test_stream_serves_multipart_jpeg_when_running(monkeypatch):
    app = create_app()
    loop = app.config["loop"]
    monkeypatch.setattr(loop, "is_running", lambda: True)
    monkeypatch.setattr(loop, "live_jpeg", lambda name=None: b"\xff\xd8stub\xff\xd9")

    r = app.test_client().get("/api/stream")
    assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
    # Pull just the first part off the streaming generator, then stop.
    chunk = next(r.response)
    assert b"Content-Type: image/jpeg" in chunk and b"\xff\xd8stub\xff\xd9" in chunk
    r.close()
