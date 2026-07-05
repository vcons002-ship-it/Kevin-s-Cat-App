"""#94: the still-cat scan runs its own (heavier) model, independent of the
camera's live model, and its last run is glanceable.
"""

import time

import numpy as np

import d20app.config as config_mod
from d20app import yolo
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


def test_scan_net_loads_the_dedicated_model(monkeypatch):
    det = PersonDetector(source="unused", model="yolo11n",
                         cat_scan_model="yolo26m")
    loaded = []

    class _Runner:
        effective_accelerator = "cpu"
        fallback_reason = ""

        def infer(self, blob):
            return None

    monkeypatch.setattr(yolo, "load_net",
                        lambda v, a: loaded.append((v, a)) or _Runner())
    runner = det._locator_net()
    assert runner is not None
    assert loaded == [("yolo26m", "cpu")]
    det._locator_net()                       # cached — loaded once
    assert len(loaded) == 1


def test_scan_net_defaults_to_the_camera_model():
    det = PersonDetector(source="unused", model="yolo11n")     # no scan model
    assert det._locator_net() is None
    same = PersonDetector(source="unused", model="yolo26m",
                          cat_scan_model="yolo26m")            # same → no 2nd net
    assert same._locator_net() is None


def test_scan_net_failure_degrades_to_the_camera_net(monkeypatch):
    det = PersonDetector(source="unused", model="yolo11n",
                         cat_scan_model="yolo26x")

    def boom(v, a):
        raise FileNotFoundError("no yolo26x here")

    monkeypatch.setattr(yolo, "load_net", boom)
    assert det._locator_net() is None        # scans fall back to the camera net
    assert det._locator_net() is None        # and don't retry-crash every scan


def test_forced_scan_uses_the_scan_net(monkeypatch):
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = PersonDetector(source="unused", cat_scan_frames=1, model="yolo11n",
                         cat_scan_model="yolo26m")

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    scan_calls, live_calls = [], []

    class _Runner:
        def infer(self, blob):
            return None

    monkeypatch.setattr(yolo, "load_net", lambda v, a: _Runner())
    monkeypatch.setattr(yolo, "detect_boxes",
                        lambda net, img, floor, size=640:
                        scan_calls.append(size) or [])
    det._run_net = lambda img, floor, size=None: live_calls.append(1) or []
    det.read_and_detect(force=True)
    assert scan_calls and not live_calls     # the dedicated net took the scan


def test_cat_scan_model_round_trips(monkeypatch, tmp_path):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://a",
                                       "cat_scan_model": "yolo26x"})
    cams = c.get("/api/cameras/saved").get_json()
    assert cams[0]["cat_scan_model"] == "yolo26x"


def test_cam_status_carries_last_scan():
    loop = DetectionLoop()
    loop._cam_status = {"Room": {"connected": True, "roll": True,
                                 "track_cats": True, "always_watch": False,
                                 "last_error": "", "resting": False}}
    loop._scan_last = {"Room": {"ts": time.time() - 120.0, "found": True}}
    row = loop.cam_status()[0]
    assert 115 <= row["scan_ago_s"] <= 125 and row["scan_found"] is True
