"""#94: the still-cat scan runs its own (heavier) model, independent of the
camera's live model, and its last run is glanceable.
"""

import threading
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
    det.read_and_detect(force=True, scan=True)
    assert scan_calls and not live_calls     # the dedicated net took the scan


def test_boost_runs_the_live_pass_not_the_scan_pass(monkeypatch):
    # #111: a boost forces the net to RUN but must use the camera's own live
    # settings. It used to share the still-scan's flag, so for the boost window
    # the live feed ran the heavier scan net (showing things the camera's own
    # settings would never detect) and then reverted.
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

    det.read_and_detect(force=True, scan=False)      # a boost
    assert live_calls and not scan_calls             # camera's own net, not the scan net


def test_boost_keeps_the_cameras_own_cat_confidence():
    # #111: the scan's threshold must not apply to a boost — only to a real scan.
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = PersonDetector(source="unused", cat_confidence=0.5,
                         cat_scan_confidence=0.8, cat_scan_frames=1,
                         track_fusion=False)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: [("cat", 0.6, (10, 10, 90, 90))]

    out = det.read_and_detect(force=True, scan=False)   # boost: camera's 0.5 bar
    assert "cat" in out.labels
    out = det.read_and_detect(force=True, scan=True)    # scan: the 0.8 bar
    assert "cat" not in out.labels


def test_cat_scan_model_is_global(monkeypatch, tmp_path):
    # #101/#102: the still-scan model is a GLOBAL setting now, not per-camera.
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/config", json={"cat_scan_model": "yolo26x",
                                "cat_scan_tiling": "4x4",
                                "cat_scan_confidence": 0.4})
    cfg = config_mod.load()
    assert cfg.cat_scan_model == "yolo26x" and cfg.cat_scan_tiling == "4x4"
    assert cfg.cat_scan_confidence == 0.4
    # and it's not written onto a saved camera
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://a"})
    assert "cat_scan_model" not in c.get("/api/cameras/saved").get_json()[0]


def test_cam_status_carries_last_scan():
    loop = DetectionLoop()
    loop._cam_status = {"Room": {"connected": True, "roll": True,
                                 "track_cats": True, "always_watch": False,
                                 "last_error": "", "resting": False}}
    loop._scan_last = {"Room": {"ts": time.time() - 120.0, "found": True}}
    row = loop.cam_status()[0]
    assert 115 <= row["scan_ago_s"] <= 125 and row["scan_found"] is True


def test_last_scan_is_safe_under_concurrent_writes():
    # H2: last_scan() (web thread, the 1.2 s /api/cats poll) iterates _scan_last
    # while a worker thread inserts keys IN PLACE at `_scan_last[name] = {...}`.
    # An unguarded iteration then raises "dictionary changed size during
    # iteration" → a 500 on the poll. Drive both real paths concurrently and
    # assert nothing raises. (Verified to fail before the _scan_lock fix.)
    loop = DetectionLoop()
    # Pre-seed so each last_scan() iterates a non-trivial dict — this widens the
    # window in which a concurrent size change would trip an unguarded iteration,
    # making a regression fail reliably rather than flakily.
    now = time.time()
    for i in range(200):
        loop._scan_last[f"seed-{i}"] = {"ts": now, "found": False}

    errors = []
    stop = threading.Event()

    def writer():
        # Churn at BOUNDED size (add one key, drop an older one) so the dict size
        # keeps changing under the reader — the exact trigger — without growing
        # without limit (which would just make each snapshot slower and slower).
        i = 0
        try:
            while not stop.is_set():
                with loop._scan_lock:      # mirror the worker's guarded write site
                    loop._scan_last[f"cam-{i}"] = {"ts": time.time(),
                                                   "found": bool(i % 2)}
                    loop._scan_last.pop(f"cam-{i - 100}", None)
                i += 1
        except Exception as e:            # noqa: BLE001 — surface, don't swallow
            errors.append(("writer", repr(e)))

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(2000):
            try:
                loop.last_scan()          # web-thread read; must never raise
            except Exception as e:        # noqa: BLE001
                errors.append(("reader", repr(e)))
                break
    finally:
        stop.set()
        t.join(timeout=5)

    assert not errors, f"concurrent _scan_last access raised: {errors}"
    # Still returns a coherent newest-scan result afterward.
    result = loop.last_scan()
    assert result is not None and set(result) >= {"camera", "ago_s", "found"}
