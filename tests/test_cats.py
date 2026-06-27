"""Cat sighting tracking: region labels, the CatTracker store, and /api/cats."""

import time

from d20app.cats import CatTracker, describe_region
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def test_describe_region_names_the_third_a_box_centre_falls_in():
    fs = (300, 300)
    assert describe_region((10, 10, 40, 40), fs) == "top-left"
    assert describe_region((130, 130, 170, 170), fs) == "center"
    assert describe_region((260, 260, 290, 290), fs) == "bottom-right"
    assert describe_region((130, 10, 170, 40), fs) == "top"        # centre column, top row
    assert describe_region((10, 10, 40, 40), None) == ""            # unknown frame size


def test_tracker_records_persists_and_reloads(tmp_path):
    path = str(tmp_path / "cats.log")
    t = CatTracker(path=path)
    t.record("Kitchen", (10, 10, 40, 40), (300, 300), 0.81, image="a.jpg")
    t.record("Garage", (260, 260, 290, 290), (300, 300), 0.6)

    last = t.last()
    assert last["camera"] == "Garage" and last["region"] == "bottom-right"
    assert t.today_count() == 2
    assert [s["camera"] for s in t.recent()] == ["Garage", "Kitchen"]   # newest first

    # A fresh tracker on the same file sees the persisted history.
    again = CatTracker(path=path)
    assert again.today_count() == 2
    assert again.last()["camera"] == "Garage"


def test_count_since_and_clear(tmp_path):
    t = CatTracker(path=str(tmp_path / "cats.log"))
    t.record("Cam", (1, 1, 9, 9), (100, 100), 0.5, ts=time.time() - 99999)  # old
    t.record("Cam", (1, 1, 9, 9), (100, 100), 0.7)                         # now
    assert t.count_since(time.time() - 60) == 1
    t.clear()
    assert t.last() is None and t.today_count() == 0


def test_best_box_returns_strongest_label_box_or_none():
    det = PersonDetector(source="unused")
    det._last_boxes = [
        ("cat", 0.4, (0, 0, 5, 5)),
        ("cat", 0.9, (10, 10, 20, 20)),
        ("person", 0.95, (1, 1, 2, 2)),
    ]
    score, box = det.best_box("cat")
    assert score == 0.9 and box == (10, 10, 20, 20)
    assert det.best_box("dog") is None


def test_api_cats_reports_last_and_today(tmp_path):
    loop = DetectionLoop()
    loop.cats = CatTracker(path=str(tmp_path / "cats.log"))   # isolate from repo file
    app = create_app(loop)
    loop.cats.record("Kitchen", (10, 10, 40, 40), (300, 300), 0.77, image="x.jpg")

    body = app.test_client().get("/api/cats").get_json()
    assert body["today"] >= 1
    assert body["last"]["camera"] == "Kitchen" and body["last"]["region"] == "top-left"
    assert body["recent"][0]["image"] == "x.jpg"

    app.test_client().post("/api/cats/clear")
    assert app.test_client().get("/api/cats").get_json()["last"] is None
