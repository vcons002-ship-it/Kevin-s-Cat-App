"""0.58.0: the feed-lag badge — a diagnostic for feeds falling minutes behind.

No JS runtime in the suite, so these are structural guards over the served markup
and script. The point they protect is placement: the first version put the number
only on the camera cards down in Setup, which is not where you are when you notice
a feed stuttering. It has to be on the feed itself.
"""

import pathlib

import d20app

_BASE = pathlib.Path(d20app.__file__).parent
INDEX = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (_BASE / "static" / "app.js").read_text(encoding="utf-8")
STYLE = (_BASE / "static" / "style.css").read_text(encoding="utf-8")


def test_both_feeds_carry_a_lag_badge():
    assert 'id="lag-0"' in INDEX and 'id="lag-1"' in INDEX


def test_each_badge_sits_inside_its_own_feed_stage():
    # Absolute positioning is relative to the stage; outside it the badge would
    # land somewhere arbitrary on the page.
    stage = INDEX.index('id="live-stage"')
    assert stage < INDEX.index('id="lag-0"') < INDEX.index('id="live-img"')
    stage2 = INDEX.index('class="feed2-stage"')
    assert stage2 < INDEX.index('id="lag-1"') < INDEX.index('id="live-img2"')


def test_the_badge_does_not_collide_with_the_feed_lock():
    # The lock pins to the top-right of the same stage; the badge must not sit on it.
    lag = STYLE[STYLE.index(".feed-lag {"):]
    lag = lag[:lag.index("}")]
    assert "left: 8px" in lag and "right" not in lag
    lock = STYLE[STYLE.index(".feed-lock {"):]
    assert "right: 8px" in lock[:lock.index("}")]


def test_the_badge_is_driven_by_the_status_poll():
    assert "renderFeedLag(body.cameras)" in APP_JS
    assert "function renderFeedLag" in APP_JS


def test_the_badge_reads_the_camera_each_feed_is_actually_showing():
    # Not the pickers' values: with Follow on, the router moves feeds and the
    # pickers trail it — reading them would label the lag with the wrong room.
    fn = APP_JS[APP_JS.index("function renderFeedLag"):]
    fn = fn[:fn.index("\n}")]
    assert "[liveCam, feed2Cam]" in fn
    assert "live-camera" not in fn


def _render_fn():
    fn = APP_JS[APP_JS.index("function renderFeedLag"):]
    return fn[:fn.index("\n}\n")]


def test_a_healthy_feed_still_shows_its_number():
    # A hidden badge is indistinguishable from a broken one, so 0.0s must render.
    fn = _render_fn()
    assert "toFixed(1)" in fn
    # The badge hides only when there is no feed or no camera — never because the
    # value is small, and never because it couldn't be measured.
    hide = fn[fn.index("if (!shown[i]"):fn.index("const lag =")]
    assert "lag" not in hide and ">" not in hide


def test_an_unmeasurable_lag_says_so_instead_of_printing_a_number():
    # The first build trusted the stream clock and reported +14s inside one second
    # — impossible for a real backlog. A number we don't believe is worse than
    # none, so an unusable clock must render as unknown, not as zero.
    fn = _render_fn()
    unknown = fn[fn.index("if (lag === null)"):fn.index("continue;\n    }")]
    # The untrusted absolute is never printed…
    assert "lag.toFixed" not in unknown
    # …but the rate is, because it's built from clamped per-step deltas and does
    # not depend on the absolute clock. Showing nothing at all is what let a real
    # minutes-long lag read as healthy on live cameras.
    assert "⏱ —" in unknown and "queued" in unknown and "s/min" in unknown


def test_the_backlog_signal_does_not_depend_on_the_stream_clock():
    # `queued` is the median blocking time of recent reads: a read that returns
    # instantly had a frame already waiting. It must stay usable precisely when
    # the seconds figure isn't.
    src = pathlib.Path(d20app.__file__).parent.joinpath("detector.py").read_text(encoding="utf-8")
    fn = src[src.index("def feed_lag"):]
    fn = fn[:fn.index("\n    def ")]
    assert "_read_history" in fn and "_lag_s" in fn
    assert "queued" in fn and "clock_ok" in fn


def test_card_chip_and_feed_badge_agree_on_when_it_is_bad():
    # Two readouts of one number that disagreed would be worse than one.
    assert APP_JS.count("lag >= 3") == 2


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
