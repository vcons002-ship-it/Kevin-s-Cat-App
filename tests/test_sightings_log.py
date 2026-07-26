"""The sightings log: unbounded daily storage, and when a row is written.

The old rules were four different gates that couldn't be predicted from outside —
a shared 10 s throttle for motion (checked BEFORE anything looked for a cat, so a
curtain could consume the budget), a rising edge for still-scans, the fuser's own
cooldown for tracks. Kevin: "I can never tell when something goes in there."

The rule now: still-scan and find always log; a moving cat logs then holds THAT
room for 60 s. Nothing else suppresses anything.
"""

import json
import os
import time

from d20app.cats import CatTracker, day_key
from d20app.webapp import create_app


def _t(tmp_path):
    return CatTracker(directory=str(tmp_path / "cats"))


# ---- storage ------------------------------------------------------------------
def test_sightings_are_written_to_a_file_per_day(tmp_path):
    t = _t(tmp_path)
    t.record("Office", (1, 1, 9, 9), (100, 100), 0.9)
    files = os.listdir(str(tmp_path / "cats"))
    assert files == [day_key(time.time()) + ".jsonl"]


def test_history_is_unbounded(tmp_path):
    # The whole point: "did she stay in that room all afternoon?" cannot be
    # answered by a rolling window of the last N sightings.
    t = CatTracker(directory=str(tmp_path / "cats"), memory=10)
    for _ in range(50):
        t.record("Office", (1, 1, 9, 9), (100, 100), 0.9)
    assert len(t.recent()) == 10                       # memory window is bounded…
    assert len(t.day(day_key(time.time()))) == 50      # …the day's file is not


def test_startup_only_reads_enough_to_fill_the_window(tmp_path):
    # Constant-time startup however much history has piled up: a year of files
    # must cost the same as a week.
    d = str(tmp_path / "cats")
    os.makedirs(d)
    for day in ("2020-01-01", "2020-01-02", "2026-07-25"):
        with open(os.path.join(d, day + ".jsonl"), "w", encoding="utf-8") as fh:
            for i in range(5):
                fh.write(json.dumps({"ts": 1.0 + i, "camera": day}) + "\n")
    read = []
    orig = CatTracker._read

    def spy(path, limit=None):
        read.append(os.path.basename(path))
        return orig(path, limit)

    CatTracker._read = staticmethod(spy)
    try:
        t = CatTracker(directory=d, memory=5)
    finally:
        CatTracker._read = staticmethod(orig)
    assert read == ["2026-07-25.jsonl"]                # stopped once full
    assert len(t.recent()) == 5


def test_a_day_can_be_read_back_on_its_own(tmp_path):
    t = _t(tmp_path)
    t.record("Office", (1, 1, 9, 9), (100, 100), 0.9)
    key = day_key(time.time())
    assert len(t.day(key)) == 1
    assert t.day("2019-01-01") == []
    assert t.days() == [key]


def test_a_day_key_cannot_escape_the_directory(tmp_path):
    t = _t(tmp_path)
    for evil in ("../config", "..", "a/b", "x.yaml", ""):
        assert t.day(evil) == []


def test_a_legacy_single_file_log_is_split_into_days_once(tmp_path):
    legacy = str(tmp_path / "cats.log")
    with open(legacy, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": 1.0, "camera": "Old"}) + "\n")
        fh.write(json.dumps({"ts": time.time(), "camera": "New"}) + "\n")
    t = CatTracker(directory=str(tmp_path / "cats"), legacy_path=legacy)
    assert sorted(t.days()) == sorted({day_key(1.0), day_key(time.time())})
    # Renamed, not deleted — losing someone's history to a migration is unforgivable.
    assert not os.path.exists(legacy) and os.path.exists(legacy + ".migrated")


def test_a_torn_final_line_does_not_lose_the_rest(tmp_path):
    # A crash mid-write leaves half a line; the rest of the day must still load.
    d = str(tmp_path / "cats")
    os.makedirs(d)
    with open(os.path.join(d, "2026-07-25.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": 1.0, "camera": "A"}) + "\n")
        fh.write('{"ts": 2.0, "camera": "B"')          # torn
    assert len(CatTracker(directory=d).recent()) == 1


def test_clear_wipes_every_day_not_just_today(tmp_path):
    # "Clear log" has to mean it, or the full-log page would still show old days.
    d = str(tmp_path / "cats")
    os.makedirs(d)
    for day in ("2026-07-24", "2026-07-25"):
        with open(os.path.join(d, day + ".jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": 1.0, "camera": "A"}) + "\n")
    t = CatTracker(directory=d)
    t.clear()
    assert t.days() == [] and t.recent() == []


# ---- the full-log page --------------------------------------------------------
def test_the_sightings_page_lists_a_day(tmp_path, monkeypatch):
    app = create_app()
    app.config["loop"].cats = _t(tmp_path)
    app.config["loop"].cats.record("Office", (1, 1, 9, 9), (100, 100), 0.87,
                                   source="still-scan", zone="the couch")
    html = app.test_client().get("/sightings").get_data(as_text=True)
    assert "Office" in html and "still-scan" in html and "the couch" in html
    assert "0.87" in html


def test_the_sightings_page_survives_an_empty_or_unknown_day(tmp_path):
    app = create_app()
    app.config["loop"].cats = _t(tmp_path)
    r = app.test_client().get("/sightings?day=2019-01-01")
    assert r.status_code == 200
    assert "No sightings recorded this day" in r.get_data(as_text=True)


# ---- when a row is written ----------------------------------------------------
import d20app.config as config_mod                      # noqa: E402
import d20app.loop as loopmod                           # noqa: E402
from d20app.config import Config                        # noqa: E402
from d20app.detector import FrameOutcome                # noqa: E402
from tests.test_multicamera import FakeDet              # noqa: E402


def _loop_with(outcomes, tmp_path, monkeypatch, cfg_kw=None, seconds=0.45):
    import tests.test_multicamera as mc
    mc.OUTCOMES = outcomes
    cams = [{"name": n, "url": u, "confirm_frames": 1, "roll": False,
             "track_cats": True} for n, u in
            (("Office", "rtsp://office/s"), ("Kitchen", "rtsp://kitchen/s"))]
    cfg = Config(speaker_names=["S"], cooldown_seconds=3600, dice_sides=1, dc=1,
                 pause_during_cooldown=False, cameras=cams,
                 active_cameras=[c["name"] for c in cams],
                 **(cfg_kw or {}))
    monkeypatch.setattr(loopmod, "PersonDetector", FakeDet)
    monkeypatch.setattr(config_mod, "load", lambda path=None: cfg)
    monkeypatch.setattr(loopmod.DetectionLoop, "_cast_for_treat", lambda s, *a, **k: None)
    dummy = type("C", (), {"start_keepalive": lambda *a: None, "close": lambda *a: None})()
    monkeypatch.setattr(loopmod.DetectionLoop, "_caster_for", lambda s, c: dummy)
    lp = loopmod.DetectionLoop()
    lp.cats = _t(tmp_path)
    lp.start()
    time.sleep(seconds)
    return lp


def test_a_moving_cat_logs_once_then_holds_that_room(tmp_path, monkeypatch):
    lp = _loop_with({"rtsp://office/s": FrameOutcome(True, False, labels=("cat",))},
                    tmp_path, monkeypatch, {"cat_scan_interval": -1})
    try:
        rows = [r for r in lp.cats.recent() if r["camera"] == "Office"]
        # Many motion frames in half a second, one row: the room is held.
        assert len(rows) == 1 and rows[0]["source"] == "motion"
    finally:
        lp.stop()


def test_one_rooms_window_does_not_gag_another(tmp_path, monkeypatch):
    # The correction Kevin made: a cat tearing around the Kitchen has no bearing
    # on whether the Office may log.
    lp = _loop_with({"rtsp://office/s": FrameOutcome(True, False, labels=("cat",)),
                     "rtsp://kitchen/s": FrameOutcome(True, False, labels=("cat",))},
                    tmp_path, monkeypatch, {"cat_scan_interval": -1})
    try:
        cams = {r["camera"] for r in lp.cats.recent()}
        assert cams == {"Office", "Kitchen"}
    finally:
        lp.stop()


def test_a_non_cat_mover_cannot_spend_the_cat_budget(tmp_path, monkeypatch):
    # The bug that made this unpredictable: the throttle was checked BEFORE
    # anything looked for a cat and was shared with the "something moved" note, so
    # a curtain at t=11 meant a cat plainly detected at t=12 was never recorded.
    import tests.test_multicamera as mc
    outcomes = {"rtsp://office/s": FrameOutcome(True, False, labels=("chair",))}
    lp = _loop_with(outcomes, tmp_path, monkeypatch, {"cat_scan_interval": -1},
                    seconds=0.25)
    try:
        assert lp.cats.recent() == []                   # a chair is not a sighting
        mc.OUTCOMES = {"rtsp://office/s": FrameOutcome(True, False, labels=("cat",))}
        time.sleep(0.3)
        rows = lp.cats.recent()
        assert rows and rows[0]["camera"] == "Office"   # …and the cat still logs
    finally:
        lp.stop()


def test_something_moved_stays_out_of_the_sightings_log(tmp_path, monkeypatch):
    lp = _loop_with({"rtsp://office/s": FrameOutcome(True, False, labels=("chair",))},
                    tmp_path, monkeypatch, {"cat_scan_interval": -1})
    try:
        assert lp.cats.recent() == []
        notes = [e for e in lp.activity.entries()
                 if "moved" in (e.get("message") or "")]
        assert notes                                    # …but it IS in the activity log
    finally:
        lp.stop()
