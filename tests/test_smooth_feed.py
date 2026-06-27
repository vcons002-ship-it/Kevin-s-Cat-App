"""Smooth live feed: the decoupled capture thread, version gating, and toggling.

A fake VideoCapture lets us drive the detector without real hardware. The grab
thread must become the sole camera reader when smooth is on, the loop must
reconcile toggles on its own thread, and nothing must regress in normal mode.
"""

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


def _detector_with_fake_cap(monkeypatch, **kw):
    det = PersonDetector(source="usb:0", confidence=0.4, model="mobilenet_ssd", **kw)
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
                         model="mobilenet_ssd", smooth_feed=True)
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


def test_live_version_endpoint_path_via_loop(monkeypatch):
    # The stream relies on loop.live_version(); it's 0 when nothing's running.
    app = create_app()
    assert app.config["loop"].live_version() == 0


def test_smooth_toggle_endpoint_persists_and_is_safe_when_stopped(tmp_path, monkeypatch):
    import d20app.config as config_mod
    cfgfile = str(tmp_path / "config.yaml")
    real_update, real_load = config_mod.update, config_mod.load
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))

    c = create_app().test_client()
    r = c.post("/api/live/smooth", json={"on": True})
    assert r.get_json()["smooth_live_feed"] is True
    assert real_load(cfgfile).smooth_live_feed is True      # persisted
