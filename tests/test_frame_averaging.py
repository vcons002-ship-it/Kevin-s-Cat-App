"""Still-scan frame averaging (0.34.0): average a short burst of frames before the
locator net runs. Real signal recovery (N samples of the same still scene → noise
drops ~√N) — the opposite of the SR-style synthesized detail #69 rejected. Any
movement mid-burst aborts to the single sharp frame; the treat path never averages.
"""

import numpy as np

import d20app.config as config_mod
from d20app.detector import PersonDetector


class _Cap:
    """A fake capture serving a scripted list of frames (then repeating the last)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        self.reads += 1
        i = min(self.reads - 1, len(self.frames) - 1)
        f = self.frames[i]
        return (f is not None), f


def _noisy(base, seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(-15, 16, base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_averaging_cuts_noise_on_a_still_scene():
    # Big enough that the stillness probe genuinely downscales (noise averages
    # out in the probe, so per-frame sensor noise doesn't read as "movement").
    base = np.full((480, 640, 3), 100, np.uint8)
    frames = [_noisy(base, s) for s in range(4)]
    det = PersonDetector(source="unused", cat_scan_frames=4)
    out = det._read_still_average(_Cap(frames[1:]), frames[0])
    err_single = np.abs(frames[0].astype(int) - base.astype(int)).mean()
    err_avg = np.abs(out.astype(int) - base.astype(int)).mean()
    assert err_avg < err_single / 1.5          # ~√4 = 2× in theory; allow slack


def test_movement_mid_burst_aborts_to_the_single_frame():
    still = np.full((120, 160, 3), 100, np.uint8)
    mover = still.copy()
    mover[40:90, 60:120] = 250                 # a bright blob walked in
    det = PersonDetector(source="unused", cat_scan_frames=3)
    out = det._read_still_average(_Cap([mover]), still)
    assert out is still                        # untouched — no smeared ghost


def test_read_failure_or_size_change_aborts():
    still = np.full((120, 160, 3), 100, np.uint8)
    det = PersonDetector(source="unused", cat_scan_frames=3)
    assert det._read_still_average(_Cap([None]), still) is still
    other = np.full((60, 80, 3), 100, np.uint8)
    assert det._read_still_average(_Cap([other]), still) is still


def test_forced_scan_averages_but_the_treat_path_never_does():
    # Burst frames must be close enough (< _AVG_DIFF_MAX) to read as still, but
    # distinct enough that the average is measurably not the first frame.
    f90 = np.full((120, 160, 3), 90, np.uint8)
    f93 = np.full((120, 160, 3), 93, np.uint8)
    det = PersonDetector(source="unused", cat_scan_frames=3)
    seen = {}

    def _loc(frame, floor):
        seen["f"] = frame
        return []

    det._detect_locator = _loc
    det._detect_boxes = lambda frame, floor: []

    cap = _Cap([f90, f93, f93])
    det._ensure_cap = lambda: cap
    det.read_and_detect(detect=True, force=True)
    assert cap.reads == 3                      # the burst
    assert abs(float(seen["f"].mean()) - 92.0) <= 1.0   # mean(90,93,93) = 92

    cap2 = _Cap([f90])
    det._ensure_cap = lambda: cap2
    det.read_and_detect(detect=True, force=False)  # ordinary motion-path read
    assert cap2.reads == 1                     # one frame, no burst


def test_cat_scan_frames_one_disables_and_cap_clamps():
    det1 = PersonDetector(source="unused", cat_scan_frames=1)
    still = np.full((60, 80, 3), 100, np.uint8)
    cap = _Cap([still])
    assert det1._read_still_average(cap, still) is still and cap.reads == 0
    assert PersonDetector(source="unused", cat_scan_frames=99).cat_scan_frames == 8


def test_cat_scan_frames_is_a_global_setting(monkeypatch, tmp_path):
    # #101/#102: still-scan settings (incl. frames) are a GLOBAL group now, not
    # per-camera — every camera uses the one value.
    from d20app.webapp import create_app

    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/config", json={"cat_scan_frames": 5})
    assert config_mod.load().cat_scan_frames == 5
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://1/s"})
    assert "cat_scan_frames" not in c.get("/api/cameras/saved").get_json()[0]
