"""The lag readout: how far behind the live edge a camera's frames are.

Shown on the camera chips in Setup, where all cameras are visible at once — the
per-feed badge was tried and removed, since the chips already answer it and cover
cameras that aren't on a feed.

The measurement itself is the substance here. It was wrong twice on real
hardware: first reporting jumps of +14 s inside one second (impossible — lag can
only grow as fast as wall clock), then reporting a camera 93 s behind as healthy.
Both directions are covered below.
"""

import pathlib

import d20app

_BASE = pathlib.Path(d20app.__file__).parent
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (_BASE / "static" / "style.css").read_text(encoding="utf-8")


def test_the_per_feed_badge_is_gone():
    assert 'id="lag-0"' not in INDEX and 'id="lag-1"' not in INDEX
    assert "renderFeedLag" not in APP_JS
    assert "feed-lag" not in STYLE          # and its styling went with it


def test_the_camera_chip_still_reports_lag():
    chip = APP_JS[APP_JS.index("function renderCamChips"):]
    chip = chip[:chip.index("\n}\n")]
    assert "lag_s" in chip and "consume_rate" in chip
    # Both an absolute figure and the "losing ground" rate, so a camera whose
    # clock is too erratic for seconds still can't pass as healthy.
    assert "s behind" in chip and "s/min" in chip


# ---- the measurement itself ----------------------------------------------------
# Lag can only grow as fast as wall clock, so a reading that leaps is the stream
# CLOCK moving, not the backlog. Measured per STEP and accumulated, so one bad
# timestamp costs one step instead of poisoning every later reading (which is
# exactly what the first two builds did).
from d20app.detector import PersonDetector    # noqa: E402


def _det():
    return PersonDetector(source="unused")


def _step(det, frame_ms, wall_s):
    """Feed one frame `wall_s` of real time after the last."""
    if det._lag_prev_wall is not None:
        det._lag_prev_wall -= wall_s
    det._note_lag(frame_ms)


def test_no_lag_while_video_arrives_as_fast_as_it_is_produced():
    det = _det()
    _step(det, 10_000.0, 0)
    _step(det, 12_000.0, 2.0)                  # 2 s of video in 2 s of real time
    _step(det, 14_000.0, 2.0)
    assert det.feed_lag()["lag_s"] < 0.05
    assert det.feed_lag()["consume_rate"] is None or abs(
        det.feed_lag()["consume_rate"] - 1.0) < 0.05


def test_lag_accumulates_when_video_arrives_slower_than_real_time():
    det = _det()
    _step(det, 0.0, 0)
    _step(det, 2_000.0, 10.0)                  # 2 s of video took 10 s
    assert abs(det.feed_lag()["lag_s"] - 8.0) < 0.1
    _step(det, 4_000.0, 10.0)                  # and again
    assert abs(det.feed_lag()["lag_s"] - 16.0) < 0.2


def test_lag_recovers_when_the_stream_catches_up():
    # The old fixed-baseline version could only ever grow. Draining must show.
    det = _det()
    _step(det, 0.0, 0)
    _step(det, 1_000.0, 6.0)                   # fall 5 s behind
    assert abs(det.feed_lag()["lag_s"] - 5.0) < 0.1
    _step(det, 4_000.0, 1.0)                   # 3 s of video in 1 s: catching up
    assert abs(det.feed_lag()["lag_s"] - 3.0) < 0.1


def test_a_backward_clock_step_does_not_invent_lag():
    det = _det()
    _step(det, 10_000.0, 0)
    _step(det, 12_000.0, 2.0)
    _step(det, 8_000.0, 2.0)                   # RTP timestamp jumps BACKWARD
    assert det.feed_lag()["lag_s"] < 0.05      # not 6 s of invented lag


def test_a_forward_clock_leap_does_not_erase_real_lag():
    # The dangerous direction: a leap forward looks like video we consumed, which
    # would wipe out a genuine backlog and report a broken feed as healthy.
    det = _det()
    _step(det, 1_000.0, 0)
    _step(det, 2_000.0, 31.0)                  # genuinely ~30 s behind
    assert det.feed_lag()["lag_s"] > 29
    _step(det, 602_000.0, 1.0)                 # clock leaps ten minutes forward
    assert det.feed_lag()["lag_s"] > 29        # the backlog survives the lie


def test_a_clock_that_keeps_jumping_stops_reporting_seconds():
    # A clock that leaps ONCE is absorbed as a single bad step. One that keeps
    # tearing back and forth can't be integrated at all, and the honest output is
    # no seconds figure — the rate and queue signals carry on regardless.
    det = _det()
    _step(det, 1_000.0, 0)
    for i in range(10):
        _step(det, 90_000.0 + i, 1.0)          # leap forward…
        _step(det, 1_000.0 + i, 1.0)           # …and tear back
    out = det.feed_lag()
    assert out["clock_ok"] is False and out["lag_s"] is None
    assert out["consume_rate"] is not None      # still measurable


def test_the_consume_rate_reports_how_fast_ground_is_being_lost():
    # The number that says lag is GROWING — and the one that still works when the
    # absolute figure can't be trusted.
    det = _det()
    _step(det, 0.0, 0)
    for i in range(1, 20):
        _step(det, i * 400.0, 1.0)             # 0.4 s of video per real second
    assert abs(det.feed_lag()["consume_rate"] - 0.4) < 0.05


def test_the_queue_signal_survives_a_broken_clock():
    # Reads returning instantly mean frames were already waiting — true whatever
    # the timestamps say, which is the point.
    det = _det()
    for _ in range(12):
        det._read_history.append(1.5)
    det._note_lag(None)                        # no usable clock at all
    out = det.feed_lag()
    assert out["lag_s"] is None
    assert out["queued"] is True and out["read_median_ms"] == 1.5


def test_a_live_edge_read_is_not_mistaken_for_a_queue():
    det = _det()
    for _ in range(12):
        det._read_history.append(33.0)         # blocking ~1/30s = the live edge
    assert det.feed_lag()["queued"] is False


def test_too_few_samples_never_claims_a_queue():
    det = _det()
    for _ in range(4):
        det._read_history.append(0.5)
    assert det.feed_lag()["queued"] is False


def test_a_reconnect_clears_the_backlog_and_its_history():
    det = _det()
    _step(det, 0.0, 0)
    _step(det, 1_000.0, 20.0)
    det._read_history.append(1.0)
    assert det.feed_lag()["lag_s"] > 18
    det._release_cap()                          # the queue dies with the socket
    assert det.feed_lag()["lag_s"] is None and not det._read_history
