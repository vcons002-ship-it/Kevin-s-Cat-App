"""When the periodic still-cat scan actually runs.

The bug this file exists for: the cadence was reset by ANY net run, motion
included. So a person moving in a room pushed the next still-scan back on every
frame and it never fired there — and the room with someone in it is exactly where
a cat sleeps unnoticed. Live detection finding nothing proves nothing about a
still-scan either: the live net is untiled, the scan is tiled and may run a
heavier model. A weak look is not a look.
"""

from d20app.config import Config
from d20app.loop import _cat_scan_due


def _cfg(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_a_scan_is_due_once_the_interval_has_passed():
    cfg = _cfg(cat_scan_interval=30.0)
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=31.0) is True
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=29.0) is False


def test_motion_no_longer_defers_the_scan():
    # The regression guard. The cadence is measured from the last SCAN; whatever
    # else the net did in between is irrelevant.
    cfg = _cfg(cat_scan_interval=30.0)
    # A busy room: the net ran constantly, but no scan has run for 100 s.
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=100.0) is True


def test_a_camera_that_does_not_track_cats_never_scans():
    cfg = _cfg(cat_scan_interval=30.0)
    assert _cat_scan_due(cfg, False, last_scan=0.0, now=999.0) is False


def test_off_and_always_on():
    assert _cat_scan_due(_cfg(cat_scan_interval=-1), True, 0.0, 999.0) is False
    assert _cat_scan_due(_cfg(cat_scan_interval=0), True, 0.0, 0.1) is True


# ---- trigger modes ------------------------------------------------------------
def test_quiet_mode_skips_a_room_where_a_cat_was_just_found():
    cfg = _cfg(cat_scan_interval=30.0, cat_scan_trigger="quiet")
    # Interval has passed, but a cat was seen 5 s ago — no point hunting for one
    # we already have.
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=40.0, last_cat=35.0) is False
    # …and once she's been gone longer than the window, look again.
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=70.0, last_cat=35.0) is True


def test_quiet_mode_scans_a_camera_that_has_never_seen_a_cat():
    # last_cat == 0.0 means "never seen", which must not be read as a monotonic
    # timestamp at the dawn of time — that would suppress the scan forever on
    # exactly the cameras most worth checking.
    cfg = _cfg(cat_scan_interval=30.0, cat_scan_trigger="quiet")
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=31.0, last_cat=0.0) is True


def test_timer_mode_ignores_recent_cats():
    cfg = _cfg(cat_scan_interval=30.0, cat_scan_trigger="timer")
    assert _cat_scan_due(cfg, True, last_scan=0.0, now=40.0, last_cat=39.0) is True


def test_the_default_is_timer():
    assert Config().cat_scan_trigger == "timer"
    # An unknown value must behave like the timer, not silently stop scanning.
    cfg = _cfg(cat_scan_interval=30.0, cat_scan_trigger="nonsense")
    assert _cat_scan_due(cfg, True, 0.0, 40.0, last_cat=39.0) is True
