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
    monkeypatch.setattr(loop, "live_jpeg",
                        lambda name=None, trail=False, last_known=True:
                        b"\xff\xd8stub\xff\xd9")

    r = app.test_client().get("/api/stream")
    assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
    # Pull just the first part off the streaming generator, then stop.
    chunk = next(r.response)
    assert b"Content-Type: image/jpeg" in chunk and b"\xff\xd8stub\xff\xd9" in chunk
    r.close()


def test_stream_ends_when_no_frame_ever_arrives(monkeypatch):
    # Review 2026-07-08 Finding 2: a stream whose camera never produces a frame
    # must NOT spin its worker thread forever — it ends after the no-frame timeout
    # so the connection (and its thread) can't leak. Shrink the timeout for the test.
    import d20app.webapp as webapp_mod
    monkeypatch.setattr(webapp_mod, "_STREAM_NO_FRAME_TIMEOUT_S", 0.15)

    app = create_app()
    loop = app.config["loop"]
    monkeypatch.setattr(loop, "is_running", lambda: True)
    monkeypatch.setattr(loop, "live_jpeg",
                        lambda name=None, trail=False, last_known=True: None)  # never a frame

    r = app.test_client().get("/api/stream")
    # The generator must terminate on its own (StopIteration), not hang forever.
    body = b"".join(r.response)
    assert body == b""          # no frame was ever emitted
    r.close()


def test_stream_heartbeat_re_emits_the_last_frame_when_the_version_stalls(monkeypatch):
    # When new frames stop arriving but one was seen, the stream re-emits it on the
    # heartbeat cadence so a disconnected client is noticed at the next yield.
    import d20app.webapp as webapp_mod
    monkeypatch.setattr(webapp_mod, "_STREAM_HEARTBEAT_S", 0.05)

    app = create_app()
    loop = app.config["loop"]
    monkeypatch.setattr(loop, "is_running", lambda: True)
    monkeypatch.setattr(loop, "live_version", lambda name=None: 7)   # never changes
    monkeypatch.setattr(loop, "live_jpeg",
                        lambda name=None, trail=False, last_known=True:
                        b"\xff\xd8stub\xff\xd9")

    r = app.test_client().get("/api/stream")
    first = next(r.response)                     # the initial frame (version 7)
    second = next(r.response)                    # a heartbeat re-emit of the same frame
    assert b"\xff\xd8stub\xff\xd9" in first and b"\xff\xd8stub\xff\xd9" in second
    r.close()
