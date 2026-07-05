"""#96: motion "custom" is configurable and motion gating can be turned off
(continuous detection). The GUI half (revealed knob inputs) is covered by the
headless render check; here: the detector gate + config plumbing.
"""

import numpy as np

import d20app.config as config_mod
from d20app.detector import PersonDetector
from d20app.webapp import create_app


def _det(frames, **kw):
    det = PersonDetector(source="unused", cat_scan_frames=1, **kw)

    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    return det


def test_gate_on_skips_the_net_on_still_frames():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(frames)                                # default: gate on
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(1) or []
    det.read_and_detect(detect=True)                  # baseline
    det.read_and_detect(detect=True)                  # identical frame: no motion
    assert calls == []


def test_gate_off_runs_the_net_every_frame():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(frames, motion_gate=False)
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(1) or []
    det.read_and_detect(detect=True)                  # baseline: net still runs
    det.read_and_detect(detect=True)                  # still frame: net runs anyway
    assert len(calls) == 2
    # outcome.motion stays HONEST (False on a still frame) — rolls need a real
    # entrance, and the trail/null bookkeeping keep their true verdicts
    outcome = det.read_and_detect(detect=True)
    assert outcome.motion is False


def test_gate_off_respects_the_cooldown_pause():
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = _det(frames, motion_gate=False)
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(1) or []
    det.read_and_detect(detect=False)                 # paused: net must NOT run
    assert calls == []


def test_sensitivity_off_round_trips_through_the_camera_store(monkeypatch, tmp_path):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://a",
                                       "motion_sensitivity": "off"})
    cams = c.get("/api/cameras/saved").get_json()
    assert cams[0]["motion_sensitivity"] == "off"
