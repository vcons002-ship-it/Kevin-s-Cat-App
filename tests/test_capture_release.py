"""Camera capture is released on every reconnect (review 2026-07-08, Finding 1).

An RTSP/FFmpeg VideoCapture context holds native buffers that __del__ does not
reliably free, so dropping the handle without release() leaks it on every
reconnect — the ~30-minute OOM. These tests pin the release behaviour.
"""

import pytest

import d20app.detector as det_mod
from d20app.detector import CameraError, PersonDetector


class _Cap:
    """A fake capture: opened-or-not, always fails reads, tracks release()."""

    def __init__(self, is_open=True):
        self._open = is_open
        self.released = False

    def isOpened(self):
        return self._open

    def read(self):
        return (False, None)          # a failed read → the reconnect path

    def release(self):
        self.released = True


def test_failed_read_releases_the_capture_before_dropping_it():
    det = PersonDetector(source="unused")
    cap = _Cap(is_open=True)
    det._cap = cap                    # _ensure_cap returns it (already "open")
    det.read_and_detect(detect=True)  # one failed read
    assert cap.released is True       # released, not just abandoned to GC
    assert det._cap is None           # handle cleared → next call reconnects


def test_ensure_cap_releases_a_stale_closed_capture_before_reopening(monkeypatch):
    det = PersonDetector(source="unused")
    stale = _Cap(is_open=False)       # opened once, now closed
    det._cap = stale
    fresh = _Cap(is_open=True)
    monkeypatch.setattr(det_mod, "_open_capture", lambda source: fresh)
    got = det._ensure_cap()
    assert stale.released is True     # old context freed before the reopen
    assert got is fresh and det._cap is fresh


def test_release_cap_is_idempotent_and_safe_when_empty():
    det = PersonDetector(source="unused")
    det._release_cap()                # no capture yet — must not raise
    assert det._cap is None
    cap = _Cap()
    det._cap = cap
    det._release_cap()
    det._release_cap()                # double release — the fake sees one, no crash
    assert cap.released is True and det._cap is None


def test_grab_loop_open_failure_surfaces_error_not_crash(monkeypatch):
    # The grab thread's reconnect path also routes through _release_cap; a genuine
    # open failure must raise CameraError (caught by the grab loop), not leak.
    det = PersonDetector(source="unused")
    stale = _Cap(is_open=False)
    det._cap = stale

    def _boom(source):
        raise CameraError("cannot open")

    monkeypatch.setattr(det_mod, "_open_capture", _boom)
    with pytest.raises(CameraError):
        det._ensure_cap()
    assert stale.released is True     # freed even though the reopen then failed
    assert det._cap is None
