"""#92: "Show me the cat" active scan — search on click, don't just jump."""

import numpy as np

import d20app.config as config_mod
import d20app.webapp as webapp
from d20app.cats import CatTracker
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop
from d20app.snapshots import SnapshotStore
from d20app.webapp import create_app


def _cfg(monkeypatch, tmp_path, **values):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    if values:
        config_mod.update(values)


class _Alive:
    def is_alive(self):
        return True


def _loop(tmp_path, cams=("Room", "Kitchen")):
    loop = DetectionLoop()
    loop._thread = _Alive()
    loop.cats = CatTracker(directory=str(tmp_path / "cats"))
    loop.snapshots = SnapshotStore(directory=str(tmp_path / "snaps"))
    # Register the cameras in config so the find endpoint's watched-set filter
    # (#103, camera_targets) recognises them as watched.
    config_mod.update({
        "cameras": [{"name": c, "url": f"rtsp://{c}"} for c in cams],
        "active_cameras": list(cams)})
    dets = {}
    for cam in cams:
        det = PersonDetector(source="unused")
        det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
        dets[cam] = det
    loop._detectors = dets
    loop._live_name = cams[0]
    return loop


def test_find_403_when_disabled(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path)                       # find_scan defaults False
    loop = _loop(tmp_path)
    r = create_app(loop).test_client().post("/api/cats/find", json={})
    assert r.status_code == 403 and "Find-my-cat" in r.get_json()["error"]


def test_find_409_when_not_running(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, find_scan=True)
    r = create_app().test_client().post("/api/cats/find", json={})
    assert r.status_code == 409


def test_find_sweeps_records_and_boosts(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, find_scan=True, find_model="yolo26m")
    loop = _loop(tmp_path)
    ran = []

    def fake_run(frame, settings, report=None):
        ran.append(settings["model"])
        # the cat is in the Kitchen (second call), not the Room
        if len(ran) == 2:
            return b"\xff\xd8jpeg\xff\xd9", [
                {"label": "cat", "score": 0.91, "box": [100, 100, 200, 200]}], 42.0
        return b"\xff\xd8jpeg\xff\xd9", [], 40.0

    monkeypatch.setattr(webapp, "_run_test_detection", fake_run)
    body = create_app(loop).test_client().post("/api/cats/find",
                                               json={}).get_json()
    assert set(ran) == {"yolo26m"}                    # the FIND model ran (#92)
    assert body["found"] == ["Kitchen"] and body["best"] == "Kitchen"
    sightings = loop.cats.recent()
    assert sightings and sightings[0]["source"] == "find"     # #93 source tag
    assert loop._cat_boost.get("Kitchen")             # feed jumps to a live box


def test_find_respects_camera_subset(monkeypatch, tmp_path):
    _cfg(monkeypatch, tmp_path, find_scan=True, find_cameras=["Room"])
    loop = _loop(tmp_path)
    monkeypatch.setattr(webapp, "_run_test_detection",
                        lambda frame, settings, report=None: (b"\xff\xd8j\xff\xd9", [], 5.0))
    body = create_app(loop).test_client().post("/api/cats/find",
                                               json={}).get_json()
    assert [r["camera"] for r in body["results"]] == ["Room"]
