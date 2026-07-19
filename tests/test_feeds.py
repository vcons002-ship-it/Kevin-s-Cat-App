"""Follow mode: camera→feed assignment by sighting recency, held.

The contract that matters is STABILITY: the pair only reshuffles when a cat turns
up outside it. The first design gated on "has a cat right now" with a hold timer,
which made a sleeping cat's feed flicker — she'd drop out of the window between
still-scans, the slot would free, the feed would hide, and the next scan brought it
back. These tests pin the recency behaviour so that can't come back.
"""

from d20app.feeds import LAST_SEEN, LIVE, FeedRouter
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def _cams(rows):
    return [r["camera"] for r in rows]


def test_primary_and_secondary_are_the_two_most_recent():
    fr = FeedRouter()
    out = fr.update({"Office": 30.0, "Hallway": 20.0, "Study": 10.0}, slots=2)
    assert _cams(out) == ["Office", "Hallway"]


def test_single_slot_shows_the_most_recent():
    fr = FeedRouter()
    assert _cams(fr.update({"Office": 30.0, "Hallway": 20.0}, slots=1)) == ["Office"]


def test_more_detections_inside_the_pair_change_nothing():
    # The reported bug in essence: a cat active in the primary must not disturb the
    # secondary, and the two must not swap places.
    fr = FeedRouter()
    fr.update({"Office": 10.0, "Hallway": 9.0}, slots=2)
    for t in (11.0, 12.0, 13.0, 40.0, 99.0):
        out = fr.update({"Office": t, "Hallway": 9.0}, slots=2)
        assert _cams(out) == ["Office", "Hallway"]
    # …even when the SECONDARY is the one being seen (it does not get promoted).
    for t in (100.0, 120.0, 300.0):
        out = fr.update({"Office": 99.0, "Hallway": t}, slots=2)
        assert _cams(out) == ["Office", "Hallway"]


def test_a_sleeping_cat_keeps_her_room_indefinitely():
    # Recency is not windowed: the room holds no matter how long ago it saw her, so
    # the feed can't blink out between still-scans.
    fr = FeedRouter()
    fr.update({"Office": 100.0, "Hallway": 99.0}, slots=2)
    for _ in range(50):
        out = fr.update({"Office": 100.0, "Hallway": 99.0}, slots=2)
        assert out == [{"camera": "Office", "source": LAST_SEEN, "locked": False},
                       {"camera": "Hallway", "source": LAST_SEEN, "locked": False}]


def test_a_third_room_resets_the_pair():
    fr = FeedRouter()
    fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2)
    # A cat appears in the kitchen: it takes primary, and the most recent of the
    # old pair drops to secondary.
    out = fr.update({"Office": 20.0, "Hallway": 10.0, "Kitchen": 30.0}, slots=2)
    assert _cams(out) == ["Kitchen", "Office"]
    # …and that new pair now holds.
    out = fr.update({"Office": 20.0, "Hallway": 10.0, "Kitchen": 31.0}, slots=2)
    assert _cams(out) == ["Kitchen", "Office"]


def test_a_still_scan_hit_outside_the_pair_is_just_a_reset():
    # A sweep that finds a cat elsewhere is an ordinary detection, so it reshuffles.
    fr = FeedRouter()
    fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2)
    out = fr.update({"Office": 20.0, "Hallway": 10.0, "Attic": 25.0}, slots=2)
    assert _cams(out) == ["Attic", "Office"]


def test_source_labels_presence_without_affecting_assignment():
    fr = FeedRouter()
    out = fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2, present={"Hallway"})
    assert out == [{"camera": "Office", "source": LAST_SEEN, "locked": False},
                   {"camera": "Hallway", "source": LIVE, "locked": False}]
    # Presence changing does NOT move anything.
    out = fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2, present=set())
    assert _cams(out) == ["Office", "Hallway"]


def test_second_feed_fills_when_turned_on_without_disturbing_the_primary():
    fr = FeedRouter()
    assert _cams(fr.update({"Office": 20.0, "Hallway": 10.0}, slots=1)) == ["Office"]
    out = fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2)
    assert _cams(out) == ["Office", "Hallway"]          # primary undisturbed


def test_never_the_same_camera_twice():
    fr = FeedRouter()
    out = fr.update({"Office": 20.0}, slots=2)
    assert out[0]["camera"] == "Office" and out[1]["camera"] is None


def test_cameras_that_go_away_are_dropped():
    fr = FeedRouter()
    fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2)
    out = fr.update({"Hallway": 10.0}, slots=2)         # Office no longer watched
    assert _cams(out) == ["Hallway", None]


def test_no_sightings_yet_means_no_assignment():
    fr = FeedRouter()
    assert fr.update({}, slots=2) == [
        {"camera": None, "source": None, "locked": False}] * 2


def test_reset_clears_the_pair():
    fr = FeedRouter()
    fr.update({"Office": 20.0, "Hallway": 10.0}, slots=2)
    fr.reset()
    assert _cams(fr.update({}, slots=2)) == [None, None]


# ---- safeguards for overlapping camera views ----------------------------------
def test_swap_confirm_count_ignores_a_transient_overlap():
    # A cat clipping the edge of another camera's view for one frame shouldn't take
    # the feed; three sustained detections should.
    fr = FeedRouter(swap_confirm_count=3)
    fr.update({"Office": 10.0}, slots=1)
    assert _cams(fr.update({"Office": 11.0, "Edge": 12.0}, slots=1)) == ["Office"]
    assert _cams(fr.update({"Office": 11.0, "Edge": 13.0}, slots=1)) == ["Office"]
    assert _cams(fr.update({"Office": 11.0, "Edge": 14.0}, slots=1)) == ["Edge"]


def test_confirm_count_zero_or_one_swaps_immediately():
    for n in (0, 1):
        fr = FeedRouter(swap_confirm_count=n)
        fr.update({"Office": 10.0}, slots=1)
        assert _cams(fr.update({"Office": 10.0, "Hall": 11.0}, slots=1)) == ["Hall"]


def test_reuse_cooldown_breaks_a_two_camera_ping_pong():
    # Both cameras genuinely keep detecting (overlapping views), so a confirm count
    # can't settle it — the cooldown has to.
    fr = FeedRouter(camera_reuse_cooldown_seconds=30.0)
    fr.update({"A": 1.0}, slots=1, now=0.0)
    assert _cams(fr.update({"A": 1.0, "B": 2.0}, slots=1, now=1.0)) == ["B"]
    # A keeps firing, but it just lost the slot — it's barred for 30 s.
    for t in (2.0, 5.0, 20.0, 29.0):
        assert _cams(fr.update({"A": t + 10, "B": 2.0}, slots=1, now=t)) == ["B"]
    # Once the cooldown lapses it can take the feed again.
    assert _cams(fr.update({"A": 99.0, "B": 2.0}, slots=1, now=40.0)) == ["A"]


def test_cooldown_off_by_default_allows_immediate_reuse():
    fr = FeedRouter()
    fr.update({"A": 1.0}, slots=1, now=0.0)
    assert _cams(fr.update({"A": 1.0, "B": 2.0}, slots=1, now=1.0)) == ["B"]
    assert _cams(fr.update({"A": 3.0, "B": 2.0}, slots=1, now=2.0)) == ["A"]


# ---- feed lock ----------------------------------------------------------------
def test_locked_feed_never_reassigns():
    fr = FeedRouter()
    fr.update({"Office": 10.0, "Hall": 9.0}, slots=2)
    # Pin the primary on the office, then let a cat tour two other rooms.
    out = fr.update({"Office": 10.0, "Hall": 9.0, "Kitchen": 20.0}, slots=2, locks={0})
    assert out[0]["camera"] == "Office" and out[0]["locked"] is True
    assert out[1]["camera"] == "Kitchen"
    out = fr.update({"Office": 10.0, "Hall": 9.0, "Kitchen": 20.0, "Study": 30.0},
                    slots=2, locks={0})
    assert out[0]["camera"] == "Office"        # still pinned
    assert out[1]["camera"] == "Study"         # only the unlocked feed moved


def test_locked_camera_is_out_of_the_pool_for_the_other_feed():
    # The sleeping-cat case: pin her room, and her ongoing detections must not drag
    # the other feed onto it, nor let it be shown twice.
    fr = FeedRouter()
    fr.update({"Office": 10.0}, slots=2, locks=set())
    out = fr.update({"Office": 999.0, "Hall": 5.0}, slots=2, locks={0})
    assert out[0]["camera"] == "Office"
    assert out[1]["camera"] == "Hall"          # not Office, despite Office being newest


def test_both_locked_means_nothing_moves():
    fr = FeedRouter()
    fr.update({"Office": 10.0, "Hall": 9.0}, slots=2)
    for t in (50.0, 100.0, 200.0):
        out = fr.update({"Office": 10.0, "Hall": 9.0, "Kitchen": t}, slots=2,
                        locks={0, 1})
        assert _cams(out) == ["Office", "Hall"]
        assert [r["locked"] for r in out] == [True, True]


def test_unlocking_returns_the_feed_to_normal_assignment():
    fr = FeedRouter()
    fr.update({"Office": 10.0, "Hall": 9.0}, slots=2)
    fr.update({"Office": 10.0, "Hall": 9.0, "Kitchen": 20.0}, slots=2, locks={0})
    out = fr.update({"Office": 10.0, "Hall": 9.0, "Kitchen": 20.0, "Study": 30.0},
                    slots=2, locks=set())
    assert out[0]["camera"] == "Study"         # free again → newest takes primary
    assert all(r["locked"] is False for r in out)


def test_locks_ignore_the_safeguards():
    # A locked feed doesn't reassign, so a confirm count / cooldown can't strand it.
    fr = FeedRouter(swap_confirm_count=5, camera_reuse_cooldown_seconds=999.0)
    fr.update({"Office": 10.0}, slots=1, now=0.0)
    for t in (1.0, 2.0, 3.0):
        out = fr.update({"Office": 10.0, "Hall": 50.0}, slots=1, locks={0}, now=t)
        assert out[0]["camera"] == "Office" and out[0]["locked"] is True


# ---- loop + endpoint wiring ---------------------------------------------------
class _Det:
    def __init__(self, last):
        self._last = last

    def cat_last_seen(self):
        return self._last


def test_last_seen_times_are_not_windowed():
    # cat_camera_times() is windowed (present *now*); cat_last_seen_times() is not —
    # that difference is what stops a sleeping cat's feed from flickering.
    import time as _time

    loop = DetectionLoop()
    now = _time.monotonic()
    loop._detectors = {"Fresh": _Det(now), "Stale": _Det(now - 3600.0),
                       "NotTracked": _Det(now)}
    loop._cam_status = {"Fresh": {"track_cats": True}, "Stale": {"track_cats": True},
                        "NotTracked": {"track_cats": False}}
    assert set(loop.cat_last_seen_times()) == {"Fresh", "Stale"}
    assert set(loop.cat_camera_times()) == {"Fresh"}      # windowed


def test_feeds_endpoint_returns_one_row_per_slot():
    loop = DetectionLoop()
    client = create_app(loop).test_client()
    assert client.get("/api/feeds").get_json()["slots"] == [
        {"camera": None, "source": None, "locked": False}]
    assert len(client.get("/api/feeds?slots=2").get_json()["slots"]) == 2
    # Bad / out-of-range input is clamped, never a 500.
    assert len(client.get("/api/feeds?slots=9").get_json()["slots"]) == 2
    assert len(client.get("/api/feeds?slots=junk").get_json()["slots"]) == 1


def test_feeds_endpoint_accepts_locks_and_knobs(monkeypatch):
    import time as _time

    import d20app.config as config_mod

    cfg = config_mod.Config()
    cfg.swap_confirm_count, cfg.camera_reuse_cooldown_seconds = 4, 45.0
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)

    loop = DetectionLoop()
    now = _time.monotonic()
    loop._detectors = {"Office": _Det(now), "Hallway": _Det(now - 600.0)}
    loop._cam_status = {"Office": {"track_cats": True}, "Hallway": {"track_cats": True}}
    client = create_app(loop).test_client()

    rows = client.get("/api/feeds?slots=2&locked=0").get_json()["slots"]
    assert rows[0]["locked"] is True and rows[1]["locked"] is False
    # the configured safeguards actually reach the router
    assert loop._feeds.swap_confirm_count == 4
    assert loop._feeds.camera_reuse_cooldown_seconds == 45.0
    # junk in `locked` is ignored rather than 500ing
    assert len(client.get("/api/feeds?slots=2&locked=junk").get_json()["slots"]) == 2
    assert client.get("/api/feeds?slots=2").get_json()["slots"][0]["locked"] is False


def test_feeds_endpoint_assigns_by_recency():
    import time as _time

    loop = DetectionLoop()
    now = _time.monotonic()
    loop._detectors = {"Office": _Det(now), "Hallway": _Det(now - 600.0)}
    loop._cam_status = {"Office": {"track_cats": True}, "Hallway": {"track_cats": True}}
    rows = create_app(loop).test_client().get("/api/feeds?slots=2").get_json()["slots"]
    # The hour-old room still holds the second feed — no window, no flicker.
    assert [r["camera"] for r in rows] == ["Office", "Hallway"]
