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
    assert "toFixed" not in unknown
    assert "⏱ —" in unknown and "queued" in unknown


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
# Lag can only grow as fast as wall clock. Anything faster is the stream CLOCK
# moving, not the backlog — and the first build absorbed exactly that as lag.
from d20app.detector import PersonDetector    # noqa: E402


def _det():
    return PersonDetector(source="unused")


def test_a_backward_clock_step_rebaselines_instead_of_inflating_the_lag():
    det = _det()
    det._note_lag(10_000.0)
    det._lag_t0 -= 2.0
    det._note_lag(12_000.0)                    # keeping up
    assert det.feed_lag()["lag_s"] < 0.1
    det._note_lag(8_000.0)                     # RTP timestamp jumps BACKWARD
    assert det.feed_lag()["lag_s"] == 0.0      # not 4 seconds of invented lag


def test_a_forward_clock_leap_is_not_read_as_a_recovery():
    det = _det()
    det._note_lag(1_000.0)
    det._lag_t0 -= 30.0                        # genuinely 30 s behind
    det._note_lag(2_000.0)
    assert det.feed_lag()["lag_s"] > 25
    det._note_lag(600_000.0)                   # clock leaps ten minutes forward
    assert det.feed_lag()["lag_s"] == 0.0      # re-baselined, not negative/absurd


def test_a_clock_that_keeps_jumping_is_reported_as_unknown():
    # The honest outcome when the instrument is unreliable: no number at all.
    det = _det()
    det._note_lag(1_000.0)
    for i in range(4):                         # repeated discontinuities
        det._note_lag(50_000.0 + i)
        det._note_lag(1_000.0 + i)
    out = det.feed_lag()
    assert out["lag_s"] is None and out["clock_ok"] is False


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
