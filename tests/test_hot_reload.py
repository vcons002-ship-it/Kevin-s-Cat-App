"""#100 config hot-reload + #101 live tiling / motion-off routing (detector side).
The loop's periodic re-read is exercised via reconfigure(); live-scene routing
is a loop-integration concern covered by the multicamera suite.
"""

import numpy as np

from d20app import dice
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop


def _det(**kw):
    return PersonDetector(source="unused", **kw)


class _Cfg:
    def __init__(self, cooldown_seconds):
        self.cooldown_seconds = cooldown_seconds


# ---- #102: the shared cooldown gate hot-reloads (no silent watch restart) ----
def test_shared_reload_updates_the_cooldown_gate():
    loop = DetectionLoop()
    loop._gate = dice.RollGate(300)          # built at start with the old value
    loop._apply_shared_reload(_Cfg(30))       # a saved change comes in mid-watch
    assert loop._gate.cooldown_s == 30.0      # applied live, no stop/start


def test_shared_reload_is_a_noop_before_watching():
    loop = DetectionLoop()
    assert loop._gate is None
    loop._apply_shared_reload(_Cfg(30))       # must not blow up when idle
    assert loop._gate is None


# ---- #100: reconfigure applies cheap knobs in place --------------------------
def test_reconfigure_updates_cheap_knobs():
    det = _det()
    changed = det.reconfigure({
        "confidence": 0.7, "cat_confidence": 0.4, "label_floor": 0.6,
        "cat_scan_tiling": "3x3", "cat_scan_tile_overlap": 0.35,
        "gamma": 1.4, "track_fusion": False,
    })
    assert changed
    assert det.confidence == 0.7 and det.cat_confidence == 0.4
    assert det.cat_scan_tiling == "3x3" and det.cat_scan_tile_overlap == 0.35
    assert det.gamma == 1.4 and det.track_fusion is False


def test_reconfigure_no_change_returns_false():
    det = _det(confidence=0.5)
    assert det.reconfigure({"confidence": 0.5}) is False


def test_reconfigure_model_change_drops_the_net():
    det = _det(model="yolo11n")
    det._yolo = object()                         # pretend a net is cached
    assert det.reconfigure({"model": "yolo26m"})
    assert det.model == "yolo26m" and det._yolo is None


def test_reconfigure_motion_gate_and_params():
    det = _det()
    assert det.motion_gate is True
    det.reconfigure({"motion_sensitivity": "off",
                     "motion_min_area_frac": 0.01, "motion_diff_threshold": 40})
    assert det.motion_gate is False
    assert det._motion.min_area_frac == 0.01 and det._motion.diff_threshold == 40


def test_reconfigure_roi():
    det = _det()
    det.reconfigure({"roi": [10, 20, 100, 80]})
    assert det.roi == [10, 20, 100, 80]
    det.reconfigure({"roi": None})
    assert det.roi is None


def test_reconfigure_scan_model_resets_loader():
    det = _det(model="yolo11n")
    det._locator_tried = True
    det.reconfigure({"cat_scan_model": "yolo26m"})
    assert det.cat_scan_model == "yolo26m" and det._locator_tried is False


# ---- #101: live tiling runs the live path tiled ------------------------------
def test_live_tiling_off_is_a_single_pass():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(cat_scan_frames=1)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(img.shape) or []
    det.read_and_detect(detect=False)               # baseline
    moving = np.full((360, 640, 3), 110, np.uint8); moving[100:180, 280:360] = 220
    frames.append(moving)
    det.read_and_detect(detect=True)
    assert len(calls) == 1                          # untiled: one forward pass


def test_live_tiling_on_tiles_the_live_path():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(cat_scan_frames=1, live_tiling="2x2", live_tile_overlap=0.2)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(img.shape) or []
    det.read_and_detect(detect=False)
    moving = np.full((360, 640, 3), 110, np.uint8); moving[100:180, 280:360] = 220
    frames.append(moving)
    det.read_and_detect(detect=True)
    assert len(calls) == 4                          # 2×2 grid: four passes


def test_reconfigure_live_tiling():
    det = _det()
    det.reconfigure({"live_tiling": "3x3", "live_tile_overlap": 0.35})
    assert det.live_tiling == "3x3" and det.live_tile_overlap == 0.35


# ---- #101: the still-scan can use its own cat threshold ----------------------
def test_forced_scan_uses_its_own_confidence():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    # camera would count a 0.5 cat; the scan demands 0.8
    det = _det(cat_confidence=0.5, cat_scan_confidence=0.8, cat_scan_frames=1)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: [("cat", 0.6, (10, 10, 90, 90))]
    out = det.read_and_detect(force=True, scan=True)   # a 0.6 cat on the scan…
    assert "cat" not in out.labels                     # …below the 0.8 scan bar
    det._run_net = lambda img, floor, size=None: [("cat", 0.85, (10, 10, 90, 90))]
    out = det.read_and_detect(force=True, scan=True)
    assert "cat" in out.labels                          # 0.85 clears it


def test_live_path_still_uses_camera_confidence():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(cat_confidence=0.5, cat_scan_confidence=0.8, cat_scan_frames=1)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: [("cat", 0.6, (10, 10, 90, 90))]
    det.read_and_detect(detect=False)                  # baseline
    moving = np.full((360, 640, 3), 110, np.uint8); moving[100:180, 280:360] = 220
    frames.append(moving)
    out = det.read_and_detect(detect=True)             # live path
    assert "cat" in out.labels                          # 0.6 clears the camera's 0.5
