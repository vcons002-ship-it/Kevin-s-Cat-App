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


def test_a_healthy_feed_still_shows_its_number():
    # A hidden badge is indistinguishable from a broken one, so 0.0s must render.
    # Only an unknown lag (no stream clock) or a dark feed hides it.
    fn = APP_JS[APP_JS.index("function renderFeedLag"):]
    fn = fn[:fn.index("\n}")]
    assert "lag === null" in fn and "toFixed(1)" in fn
    # …and the hide condition is about absence, never about the value being small.
    hide = fn[fn.index("if (!shown[i]"):fn.index("el.classList.remove")]
    assert "lag === null" in hide and ">" not in hide


def test_card_chip_and_feed_badge_agree_on_when_it_is_bad():
    # Two readouts of one number that disagreed would be worse than one.
    assert APP_JS.count("lag >= 3") == 2
