"""Follow mode (#113): sticky, debounced camera→feed assignment.

The point of these tests is the anti-flicker contract — feeds adopt and HOLD a
camera rather than competing for "most recent", so two cats in two rooms produce
two stable feeds and a doorway transient never yanks a feed off what you're
watching. All timing is injected, so none of this is flaky.
"""

from d20app.feeds import LAST_SEEN, LIVE, FeedRouter
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def _cams(r):
    return [s["camera"] for s in r]


def test_single_feed_adopts_the_active_camera():
    fr = FeedRouter(hold_s=3.0, persist_s=1.0)
    assert _cams(fr.update(0.0, {"Kitchen": 0.0}, slots=1)) == [None]   # not persistent yet
    out = fr.update(1.0, {"Kitchen": 1.0}, slots=1)
    assert out == [{"camera": "Kitchen", "source": LIVE}]


def test_a_transient_elsewhere_does_not_steal_the_feed():
    # The doorway-pass case: a blip on another camera must not move the feed.
    fr = FeedRouter(hold_s=3.0, persist_s=1.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=1)
    fr.update(1.0, {"Kitchen": 1.0}, slots=1)
    out = fr.update(1.5, {"Kitchen": 1.5, "Hall": 1.5}, slots=1)
    assert _cams(out) == ["Kitchen"]                 # held, not reassigned


def test_held_camera_survives_a_quiet_gap_then_follows_the_cat():
    fr = FeedRouter(hold_s=3.0, persist_s=1.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=1)
    fr.update(1.0, {"Kitchen": 1.0}, slots=1)
    # Cat leaves the kitchen; within the hold window the feed stays put.
    assert _cams(fr.update(2.0, {}, slots=1)) == ["Kitchen"]
    assert _cams(fr.update(3.5, {}, slots=1)) == ["Kitchen"]
    # Cat turns up in the hall and stays; once the hold lapses the feed follows.
    fr.update(4.0, {"Hall": 4.0}, slots=1)
    out = fr.update(5.2, {"Hall": 5.2}, slots=1)
    assert _cams(out) == ["Hall"]


def test_new_detections_on_a_held_camera_refresh_the_hold():
    fr = FeedRouter(hold_s=3.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=1)
    for t in (2.0, 4.0, 6.0, 8.0):          # each past hold_s since the previous
        assert _cams(fr.update(t, {"Kitchen": t}, slots=1)) == ["Kitchen"]


def test_two_cats_two_rooms_are_stable_and_never_swap():
    # The headline anti-flicker case: each feed holds its own cat.
    fr = FeedRouter(hold_s=3.0, persist_s=0.0)
    seen = []
    for i in range(12):
        t = float(i)
        # Both rooms keep reporting; their "most recent" order alternates.
        active = ({"Kitchen": t, "Hall": t - 0.1} if i % 2
                  else {"Kitchen": t - 0.1, "Hall": t})
        seen.append(_cams(fr.update(t, active, slots=2)))
    assert all(row == seen[0] for row in seen)       # never swapped
    assert set(seen[0]) == {"Kitchen", "Hall"}


def test_second_feed_shows_the_previous_room_when_only_one_cat():
    fr = FeedRouter(hold_s=2.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=2)
    assert _cams(fr.update(1.0, {"Kitchen": 1.0}, slots=2)) == ["Kitchen", None]
    # The cat moves to the hall; after the kitchen's hold lapses the primary
    # follows and the kitchen becomes the secondary's "last seen".
    fr.update(2.0, {"Hall": 2.0}, slots=2)
    out = fr.update(4.5, {"Hall": 4.5}, slots=2)
    assert out[0] == {"camera": "Hall", "source": LIVE}
    assert out[1] == {"camera": "Kitchen", "source": LAST_SEEN}


def test_a_live_second_cat_outranks_a_stale_last_seen_view():
    fr = FeedRouter(hold_s=2.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=2)
    fr.update(1.0, {"Hall": 1.0}, slots=2)
    out = fr.update(4.0, {"Hall": 4.0}, slots=2)
    assert out[1]["source"] == LAST_SEEN          # secondary is showing the old room
    # A real cat appears in a third room — it takes the secondary immediately.
    out = fr.update(4.5, {"Hall": 4.5, "Study": 4.5}, slots=2)
    assert out[0] == {"camera": "Hall", "source": LIVE}
    assert out[1] == {"camera": "Study", "source": LIVE}


def test_a_camera_is_never_shown_on_both_feeds():
    fr = FeedRouter(hold_s=2.0, persist_s=0.0)
    for t in (0.0, 1.0, 2.0, 3.0):
        out = _cams(fr.update(t, {"Kitchen": t}, slots=2))
        assert out[0] == "Kitchen" and out[1] != "Kitchen"


def test_freed_feed_repicks_the_most_recent_unheld_camera():
    fr = FeedRouter(hold_s=1.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0, "Hall": 0.0}, slots=2)
    assert set(_cams(fr.update(0.1, {"Kitchen": 0.1, "Hall": 0.1}, slots=2))) == {
        "Kitchen", "Hall"}
    # Hall's cat leaves and a newer one shows in the study; hall's slot takes it.
    fr.update(2.0, {"Kitchen": 2.0, "Study": 2.0}, slots=2)
    out = _cams(fr.update(3.5, {"Kitchen": 3.5, "Study": 3.5}, slots=2))
    assert set(out) == {"Kitchen", "Study"}


def test_slot_count_changes_are_handled():
    fr = FeedRouter(hold_s=2.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0, "Hall": 0.0}, slots=2)
    two = _cams(fr.update(0.5, {"Kitchen": 0.5, "Hall": 0.5}, slots=2))
    assert len(two) == 2
    one = fr.update(1.0, {"Kitchen": 1.0, "Hall": 1.0}, slots=1)   # user turns the 2nd off
    assert len(one) == 1 and one[0]["camera"] == two[0]            # primary undisturbed
    assert len(fr.update(1.5, {"Kitchen": 1.5, "Hall": 1.5}, slots=2)) == 2


def test_no_cameras_means_no_assignment():
    fr = FeedRouter()
    assert fr.update(0.0, {}, slots=2) == [{"camera": None, "source": None}] * 2


def test_reset_clears_assignments():
    fr = FeedRouter(hold_s=2.0, persist_s=0.0)
    fr.update(0.0, {"Kitchen": 0.0}, slots=1)
    fr.reset()
    assert _cams(fr.update(0.1, {}, slots=1)) == [None]


# ---- loop + endpoint wiring ---------------------------------------------------
class _Det:
    def __init__(self, last):
        self._last = last

    def cat_last_seen(self):
        return self._last


def test_cat_camera_times_only_lists_cat_tracking_cameras_with_a_cat():
    import time as _time

    loop = DetectionLoop()
    now = _time.monotonic()
    loop._detectors = {"Kitchen": _Det(now), "Hall": _Det(now),
                       "Old": _Det(now - 3600.0), "Person": _Det(now)}
    loop._cam_status = {"Kitchen": {"track_cats": True}, "Hall": {"track_cats": True},
                        "Old": {"track_cats": True}, "Person": {"track_cats": False}}
    times = loop.cat_camera_times()
    assert set(times) == {"Kitchen", "Hall"}          # not stale, not non-tracking
    assert set(loop.cats_present_cameras()) == {"Kitchen", "Hall"}


def test_feeds_endpoint_returns_one_row_per_slot():
    loop = DetectionLoop()
    client = create_app(loop).test_client()
    body = client.get("/api/feeds").get_json()
    assert body["slots"] == [{"camera": None, "source": None}]
    body = client.get("/api/feeds?slots=2").get_json()
    assert len(body["slots"]) == 2
    # Bad / out-of-range input is clamped, never a 500.
    assert len(client.get("/api/feeds?slots=9").get_json()["slots"]) == 2
    assert len(client.get("/api/feeds?slots=junk").get_json()["slots"]) == 1


def test_feeds_endpoint_follows_the_camera_with_the_cat(monkeypatch):
    import time as _time

    import d20app.config as config_mod

    cfg = config_mod.Config()
    cfg.follow_persist_seconds = 0.0        # adopt immediately instead of after 1 s
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)

    loop = DetectionLoop()
    loop._detectors = {"Kitchen": _Det(_time.monotonic())}
    loop._cam_status = {"Kitchen": {"track_cats": True}}
    client = create_app(loop).test_client()
    assert client.get("/api/feeds").get_json()["slots"][0]["camera"] == "Kitchen"


def test_feeds_endpoint_honours_the_configured_knobs(monkeypatch):
    # The hold/persist knobs are meant to be tuned live, so the endpoint must
    # actually apply them to the router rather than using the defaults.
    import d20app.config as config_mod

    cfg = config_mod.Config()
    cfg.follow_hold_seconds, cfg.follow_persist_seconds = 12.0, 0.25
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)

    loop = DetectionLoop()
    create_app(loop).test_client().get("/api/feeds")
    assert loop._feeds.hold_s == 12.0 and loop._feeds.persist_s == 0.25
