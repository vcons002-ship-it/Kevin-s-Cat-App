"""Smooth live feed: the decoupled capture thread, version gating, and toggling.

A fake VideoCapture lets us drive the detector without real hardware. The grab
thread must become the sole camera reader when smooth is on, the loop must
reconcile toggles on its own thread, and nothing must regress in normal mode.
"""

import pathlib
import threading
import time

import numpy as np

from d20app import detector as det_mod
from d20app.detector import CameraError, PersonDetector
from d20app.webapp import create_app


class FakeCap:
    """A capture that hands out unique frames and counts reads (thread-safe)."""

    def __init__(self):
        self._open = True
        self._lock = threading.Lock()
        self.reads = 0

    def isOpened(self):
        return self._open

    def read(self):
        with self._lock:
            self.reads += 1
            n = self.reads
        # A distinct value per frame so motion/version actually change.
        return True, np.full((48, 64, 3), n % 256, dtype=np.uint8)

    def release(self):
        self._open = False


class DyingCap:
    """Hands out `good` real frames, then fails every read after that."""

    def __init__(self, good=2):
        self._open = True
        self._left = good
        self._lock = threading.Lock()

    def isOpened(self):
        return self._open

    def read(self):
        with self._lock:
            if self._left > 0:
                self._left -= 1
                return True, np.full((48, 64, 3), 7, dtype=np.uint8)
        return False, None

    def release(self):
        self._open = False


def _detector_with_fake_cap(monkeypatch, **kw):
    det = PersonDetector(source="usb:0", confidence=0.4, model="yolo11n", **kw)
    cap = FakeCap()
    # _ensure_cap returns our fake instead of opening a real device.
    monkeypatch.setattr(det, "_ensure_cap", lambda: cap)
    det._cap = cap
    return det, cap


def test_normal_mode_reads_on_the_loop_thread_no_grab_thread(monkeypatch):
    det, cap = _detector_with_fake_cap(monkeypatch)
    det.read_and_detect(detect=False)
    assert det._grab_thread is None and det.smooth_feed is False
    assert cap.reads == 1                       # exactly one read per call
    assert det.live_version() == 1 and det.live_jpeg() is not None


def test_smooth_mode_starts_grab_thread_and_streams_without_loop_reads(monkeypatch):
    det, cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True)
    try:
        det.read_and_detect(detect=False)       # loop thread reconciles → grabber starts
        assert det.smooth_feed is True and det._grab_thread is not None
        # The grab thread keeps reading and bumping the version on its own.
        v0 = det.live_version()
        deadline = time.time() + 2.0
        while det.live_version() <= v0 and time.time() < deadline:
            time.sleep(0.02)
        assert det.live_version() > v0          # advanced without another loop call
        assert cap.reads > 1
        assert det.live_jpeg() is not None
    finally:
        det.release()
    assert det._grab_thread is None             # release joins the grabber


def test_toggle_on_then_off_is_reconciled_on_the_loop_thread(monkeypatch):
    det, cap = _detector_with_fake_cap(monkeypatch)
    try:
        det.read_and_detect(detect=False)
        assert det.smooth_feed is False

        det._smooth_desired = True              # what loop.set_smooth() does
        det.read_and_detect(detect=False)       # reconcile → grabber on
        assert det.smooth_feed is True and det._grab_thread is not None

        det._smooth_desired = False
        det.read_and_detect(detect=False)       # reconcile → grabber stopped
        assert det.smooth_feed is False and det._grab_thread is None
        # Back on the synchronous path: a loop read still produces a frame.
        assert det.live_jpeg() is not None
    finally:
        det.release()


def test_smooth_mode_surfaces_a_grab_error_to_the_loop(monkeypatch):
    det = PersonDetector(source="usb:0", confidence=0.4,
                         model="yolo11n", smooth_feed=True)
    monkeypatch.setattr(det, "_ensure_cap",
                        lambda: (_ for _ in ()).throw(CameraError("camera gone")))
    try:
        # The first call starts the grabber; once it can't open the camera, a
        # subsequent read surfaces CameraError to the loop (which call exactly is
        # a race with the grab thread, so allow a few).
        raised = False
        deadline = time.time() + 2.0
        while not raised and time.time() < deadline:
            try:
                det.read_and_detect(detect=False)
            except CameraError:
                raised = True
            time.sleep(0.05)
        assert raised                            # the loop sees the camera failure
    finally:
        det.release()


def test_smooth_mode_surfaces_a_camera_that_dies_after_a_good_frame(monkeypatch):
    # Regression: the grab thread holds the last good frame, so the loop must still
    # learn the camera died (otherwise detection silently freezes on a stale frame).
    det = PersonDetector(source="usb:0", confidence=0.4,
                         model="yolo11n", smooth_feed=True)
    cap = DyingCap(good=2)
    monkeypatch.setattr(det, "_ensure_cap", lambda: cap)
    monkeypatch.setattr(det, "_GRAB_STALE_SECONDS", 0.3)   # don't wait the full 2s
    det._cap = cap
    try:
        det.read_and_detect(detect=False)        # starts grabber; it gets ~2 frames
        deadline = time.time() + 3.0
        raised = False
        while not raised and time.time() < deadline:
            try:
                det.read_and_detect(detect=False)
            except CameraError:
                raised = True
            time.sleep(0.05)
        assert raised                             # stale frame + grab error -> surfaced
    finally:
        det.release()


def test_smooth_watchdog_respawns_a_dead_grabber(monkeypatch):
    det, cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True)
    try:
        det.read_and_detect(detect=False)         # starts the grabber
        assert det._grab_thread is not None and det._grab_thread.is_alive()

        # Simulate the grabber dying without a clean reconcile (the wedged-then-
        # unwedged case): kill it but leave smooth_feed True.
        original = det._grab_thread
        det._grab_stop.set()
        original.join(timeout=2)
        assert not original.is_alive()

        det.read_and_detect(detect=False)         # watchdog must respawn it
        assert det._grab_thread is not None and det._grab_thread.is_alive()
        assert det._grab_thread is not original
    finally:
        det.release()


def test_live_version_endpoint_path_via_loop(monkeypatch):
    # The stream relies on loop.live_version(); it's 0 when nothing's running.
    app = create_app()
    assert app.config["loop"].live_version() == 0


def test_the_smooth_toggle_endpoint_is_gone(tmp_path, monkeypatch):
    # 0.59.0: continuous reading is no longer a user preference. A network stream
    # queues, so reading it slower than it arrives means acting on ever-older
    # frames — turning it "off" didn't save meaningful CPU, it just made the
    # detector work minutes in the past. The decision is now the source type.
    c = create_app().test_client()
    assert c.post("/api/live/smooth", json={"on": True}).status_code == 404


def test_a_network_camera_reads_continuously_and_usb_does_not():
    from d20app.detector import continuous_read_wanted

    assert continuous_read_wanted("rtsp://192.0.2.1/stream1") is True
    assert continuous_read_wanted("http://192.0.2.1/video.mjpg") is True
    assert continuous_read_wanted("usb:0") is False      # local capture doesn't queue
    assert continuous_read_wanted("usb:2") is False


def test_a_saved_camera_no_longer_carries_a_smooth_setting():
    import d20app.config as config_mod

    spec = config_mod.coerce_camera({"name": "K", "url": "rtsp://a"}, config_mod.Config())
    assert "smooth_feed" not in spec
    # …and a leftover key from an older config is ignored, not honoured.
    stale = config_mod.coerce_camera(
        {"name": "K", "url": "rtsp://a", "smooth_feed": False}, config_mod.Config())
    assert "smooth_feed" not in stale


# ---- the two things that ride on continuous reading ---------------------------
def test_the_feed_shows_what_the_net_sees(monkeypatch):
    # The grab thread used to publish the RAW frame while the net got the adjusted
    # one, so a camera relying on gamma to see a dark room looked untouched in the
    # GUI. Now that every network camera reads continuously, that would have hit
    # everyone — the published frame must carry the adjustments.
    det, cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True, brightness=40)
    det._smooth_desired = True
    det.read_and_detect(detect=False)                  # loop thread starts the grabber
    try:
        deadline = time.time() + 2
        while det.latest_frame() is None and time.time() < deadline:
            time.sleep(0.01)
        published = det.latest_frame()
        assert published is not None
        # FakeCap hands out flat frames; brightness must have moved the value up.
        raw_value = int(cap.reads % 256)
        assert int(published.mean()) > raw_value or int(published.mean()) >= 40
    finally:
        det.release()


def test_detection_never_analyses_the_same_frame_twice(monkeypatch):
    # With scan_fps at or above the camera's rate the newest frame hasn't changed
    # yet. Diffing a frame against itself gives zero motion, so the gate would
    # never fire and detection would silently stop. Skipping is the honest answer.
    det, _cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True)
    det.smooth_feed = True                     # pretend the grabber is running
    det._grab_thread = threading.current_thread()
    frame = np.full((48, 64, 3), 90, np.uint8)
    det._publish_frame(frame)

    runs = []
    det._detect_boxes = lambda img, floor: runs.append(1) or []
    det._motion.update = lambda gray, ts_ms=None: True    # would always detect

    first = det.read_and_detect(detect=True)
    second = det.read_and_detect(detect=True)             # same frame, no new version
    assert len(runs) == 1                                  # the repeat was skipped
    assert second.motion is False and second.person is False

    det._publish_frame(np.full((48, 64, 3), 200, np.uint8))   # a genuinely new frame
    det.read_and_detect(detect=True)
    assert len(runs) == 2
    assert first is not None


def test_the_measured_camera_rate_needs_real_samples(monkeypatch):
    # The scan_fps warning must never fire off one lucky frame — and CAP_PROP_FPS
    # is not trusted here because cameras misreport it.
    det, _cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True)
    assert det.camera_fps() is None            # nothing measured yet
    det._grab_t0 = time.monotonic() - 2.0
    det._grab_frames = 5
    assert det.camera_fps() is None            # too few samples to claim a rate
    det._grab_frames = 50
    assert det.camera_fps() is not None


def test_lag_is_measured_on_the_thread_that_actually_reads(monkeypatch):
    # The lag badge exists to show a feed falling behind. Once every network camera
    # moved to the grab thread, measuring in the synchronous path would have
    # reported "unknown" on precisely the cameras it was built to watch.
    det, _cap = _detector_with_fake_cap(monkeypatch, smooth_feed=True)
    src = pathlib.Path(det_mod.__file__).read_text(encoding="utf-8")
    grab = src[src.index("def _grab_loop"):src.index("def camera_fps")]
    assert "_note_lag" in grab and "_read_history" in grab
    # …and the smooth branch of read_and_detect no longer blanks the figure.
    read = src[src.index("if self.smooth_feed:"):src.index("cropped = self._crop")]
    assert "self._lag_s = None" not in read
