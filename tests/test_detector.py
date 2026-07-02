"""Unit tests for the detection core (no camera, no model inference).

Motion pre-filter, local-vs-stream capture routing, detector tuning, and the
detect=False / forced-scan paths. (Person-vs-cat gating is exercised through the
YOLO box path in test_yolo.py / test_detection_accuracy.py.)
"""

import numpy as np

from d20app import detector


def test_parse_local_index():
    assert detector.parse_local_index("usb:0") == 0
    assert detector.parse_local_index("usb:2") == 2
    assert detector.parse_local_index("usb:x") is None
    assert detector.parse_local_index("rtsp://cam/stream") is None
    assert detector.parse_local_index("") is None


def test_open_capture_routes_local_index_vs_ffmpeg_stream(monkeypatch):
    import cv2

    calls = []

    class _Cap:
        def isOpened(self): return True
        def read(self): return True, None
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: calls.append(a) or _Cap())

    detector._open_capture("usb:1")
    assert calls[-1][0] == 1                      # opened by integer device index
    assert calls[-1][1] != cv2.CAP_FFMPEG         # platform backend, not FFmpeg

    detector._open_capture("rtsp://cam/stream")
    assert calls[-1] == ("rtsp://cam/stream", cv2.CAP_FFMPEG)


def test_motion_prefilter_first_frame_and_change():
    # First frame reports NO motion (nothing to compare yet); an identical frame
    # reports none; a large solid change reports motion.
    mp = detector.MotionPrefilter(min_area_frac=0.01)
    blank = np.zeros((200, 200), dtype=np.uint8)
    assert mp.update(blank) is False           # first frame: no baseline
    assert mp.update(blank.copy()) is False    # no change
    moved = blank.copy()
    moved[0:120, 0:120] = 255                   # 36% of pixels change
    assert mp.update(moved) is True


def test_motion_prefilter_ignores_sensor_noise():
    # Random per-pixel grain (like night-vision noise / compression) must NOT
    # register as motion after the median blur — this is the false-trigger fix.
    rng = np.random.default_rng(0)
    mp = detector.MotionPrefilter()
    base = np.full((300, 300), 120, dtype=np.uint8)
    mp.update(base)                                    # prime baseline
    noisy = np.clip(base.astype(int) + rng.integers(-30, 31, base.shape), 0, 255)
    assert mp.update(noisy.astype(np.uint8)) is False


def test_motion_prefilter_ignores_decode_artifact_lines():
    # A corrupt camera frame shows a long, thin band of bad pixels: lots of
    # changed pixels but only a few rows tall. It must NOT trigger motion.
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)                                    # prime baseline
    for thickness in (2, 4, 8):                        # thin to moderately thick
        artifact = base.copy()
        artifact[180:180 + thickness, :] = 255         # full-width bright band
        assert mp.update(artifact) is False, f"{thickness}px band triggered motion"
        mp._prev = None                                # reset baseline for next case
        mp.update(base)


def test_motion_prefilter_triggers_on_solid_blob():
    # A compact moving object (a person/cat-sized blob) must still trigger.
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)
    moved = base.copy()
    moved[120:300, 250:360] = 255                      # ~110x180 solid blob
    assert mp.update(moved) is True


# --- motion blob retention: the escalation ladder's "look here" hints (#66) ----

def test_motion_blobs_retained_with_location():
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    assert mp.update(base) is False and mp.last_blobs == []   # first frame: no hints
    moved = base.copy()
    moved[120:300, 250:360] = 255
    assert mp.update(moved) is True
    assert len(mp.last_blobs) == 1 and mp.last_blobs_ts > 0
    x1, y1, x2, y2 = mp.last_blobs[0]
    # the retained box covers the moving blob (median blur shifts edges a little)
    assert x1 <= 255 and x2 >= 355 and y1 <= 125 and y2 >= 295


def test_small_mover_is_a_hint_but_not_a_motion_verdict():
    # A blob below min_area_frac (the distant-cat case) must NOT trip motion,
    # but MUST be retained as an escalation hint — the new hint/verdict split.
    mp = detector.MotionPrefilter(min_area_frac=0.05, min_blob_px=10)
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)
    moved = base.copy()
    moved[100:140, 100:140] = 255                      # 40x40: solid but small
    assert mp.update(moved) is False                    # verdict unchanged
    assert len(mp.last_blobs) == 1                      # ...yet the hint survives


def test_static_frame_clears_blob_hints():
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)
    moved = base.copy()
    moved[120:300, 250:360] = 255
    mp.update(moved)
    assert mp.last_blobs
    mp.update(moved.copy())                             # no change now
    assert mp.last_blobs == []


# --- PersonDetector tuning + cooldown pause -----------------------------------

def test_detector_forwards_motion_params_and_label_floor():
    det = detector.PersonDetector(
        source="unused", motion_min_area_frac=0.01, motion_diff_threshold=40,
        motion_min_blob_px=20, label_floor=0.7)
    assert det.label_floor == 0.7
    assert det._motion.min_area_frac == 0.01
    assert det._motion.diff_threshold == 40
    assert det._motion.min_blob_px == 20


def test_read_and_detect_skips_net_when_paused():
    # detect=False must NOT run the neural net (the CPU saver). It still reads a
    # frame and returns a neutral, no-motion outcome.
    det = detector.PersonDetector(source="unused")
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    class _FakeCap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _FakeCap()
    det._detect_boxes = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the net ran while detection was paused"))

    out = det.read_and_detect(detect=False)
    assert out.motion is False and out.person is False
    assert det.frame_size == (60, 40)        # the frame was still read


def test_read_and_detect_force_runs_net_without_motion():
    # The periodic still-cat scan: force=True runs the net even when nothing moved
    # (motion stays False), surfaces the cat label, and stamps cat_last_seen so the
    # GUI flash can persist between scans.
    det = detector.PersonDetector(source="unused")
    frame = np.zeros((40, 60, 3), dtype=np.uint8)

    class _FakeCap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _FakeCap()
    # No motion (a static frame), but the forced scan still classifies it as a cat.
    # force=True routes through the locator path → _run_net (tiling off by default).
    det._run_net = lambda img, floor, size=None: [("cat", 0.9, (1, 1, 9, 9))]

    out = det.read_and_detect(detect=False, force=True)
    assert out.motion is False            # the pre-filter didn't fire...
    assert "cat" in out.labels            # ...but the net ran and saw the cat
    assert det.cat_last_seen() > 0.0      # stamped for the flash window
