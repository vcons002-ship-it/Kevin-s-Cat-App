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


# ---- reference frame age: motion must not depend on how frames are buffered ----
def _walker(n, step, size=60, w=640, h=360):
    """n frames of a blob moving `step` px each — a cat crossing at a fixed speed."""
    import numpy as np

    out = []
    for i in range(n):
        f = np.full((h, w), 60, np.uint8)
        x = 20 + i * step
        f[180:180 + size, x:x + size] = 210
        out.append(f)
    return out


def test_motion_verdict_survives_closely_spaced_frames():
    # The bug: RTSP frames queue, so consecutive reads are ~33 ms apart in VIDEO
    # time even though the loop reads every 200 ms. The verdict is an AREA test, so
    # the same cat covers far less of it and slow movers were missed entirely.
    from d20app.detector import MotionPrefilter

    frames = _walker(40, step=3)          # a slow mover: 3 px per 33 ms frame
    kw = dict(min_area_frac=0.003, diff_threshold=25, min_blob_px=14)

    # Frames arriving 33 ms apart, compared against the previous one (old behaviour).
    old = MotionPrefilter(reference_ms=0, **kw)
    old_fires = sum(bool(old.update(g, ts_ms=i * 33)) for i, g in enumerate(frames))

    # Same frames, same settings — but compared against one ~200 ms old.
    new = MotionPrefilter(reference_ms=200, **kw)
    new_fires = sum(bool(new.update(g, ts_ms=i * 33)) for i, g in enumerate(frames))

    assert old_fires == 0, "precondition: this mover is invisible frame-to-frame"
    assert new_fires > 0, "a 200 ms reference must see the movement the gap hides"


def test_reference_age_makes_the_verdict_independent_of_frame_rate():
    # The point of the fix: the same motion should read the same whether frames
    # arrive every 33 ms (queued) or every 200 ms (drained), because the comparison
    # spans the same amount of VIDEO either way.
    from d20app.detector import MotionPrefilter

    kw = dict(min_area_frac=0.003, diff_threshold=25, min_blob_px=14,
              reference_ms=200)
    # Same physical speed (18 px per 200 ms), sampled at two different rates.
    dense = _walker(40, step=3)                       # 33 ms apart
    sparse = _walker(7, step=18)                      # 200 ms apart

    d = MotionPrefilter(**kw)
    dense_rate = sum(bool(d.update(g, ts_ms=i * 33)) for i, g in enumerate(dense)) / len(dense)
    s = MotionPrefilter(**kw)
    sparse_rate = sum(bool(s.update(g, ts_ms=i * 200)) for i, g in enumerate(sparse)) / len(sparse)

    assert dense_rate > 0 and sparse_rate > 0
    assert abs(dense_rate - sparse_rate) < 0.35, (dense_rate, sparse_rate)


def test_no_timestamps_keeps_the_previous_frame_behaviour():
    # Cameras that don't report a stream position (and the smooth path's own
    # callers) must keep working exactly as before.
    from d20app.detector import MotionPrefilter

    frames = _walker(10, step=30)
    mp = MotionPrefilter(reference_ms=200, min_area_frac=0.003,
                         diff_threshold=25, min_blob_px=14)
    assert sum(bool(mp.update(g)) for g in frames) > 0


def test_history_is_bounded():
    # The lookback keeps frames in memory; it must not grow with runtime.
    from d20app.detector import MotionPrefilter

    mp = MotionPrefilter(reference_ms=200, min_area_frac=0.003,
                         diff_threshold=25, min_blob_px=14)
    for i, g in enumerate(_walker(400, step=1)):
        mp.update(g, ts_ms=i * 33)
    assert len(mp._history) <= 12


# ---- post-motion hold (0.57.0) -------------------------------------------------
# Motion decides WHEN to look. On its own that means the net only ever sees frames
# that moved — the motion-blurred, mid-stride ones — and a cat pausing mid-walk runs
# no inference at all. The hold keeps looking for a moment after movement stops.
import time as _time  # noqa: F401 — used by the hold tests

import numpy as np

from d20app.detector import PersonDetector


def _hold_detector(hold, moved_flags):
    """A detector whose motion verdict is scripted; returns it plus a net-run log."""
    det = PersonDetector(source="unused", motion_hold_seconds=hold,
                         track_fusion=False)
    frame = np.full((360, 640, 3), 110, np.uint8)

    class _Cap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _Cap()
    seq = iter(moved_flags)
    det._motion.update = lambda gray, ts_ms=None: next(seq)
    runs = []
    det._detect_boxes = lambda img, floor: runs.append(1) or []
    return det, runs


def test_hold_keeps_the_net_running_after_motion_stops():
    # One motion frame, then four still ones: without the hold only the first would
    # reach the net. This is the whole point — the frames just after movement stops
    # are the sharpest and best-posed in the sequence.
    det, runs = _hold_detector(5.0, [True, False, False, False, False])
    outcomes = [det.read_and_detect(detect=True) for _ in range(5)]

    assert len(runs) == 5                       # every frame reached the net
    assert outcomes[0].motion and not outcomes[0].held
    assert all(not o.motion and o.held for o in outcomes[1:])   # motion stays honest


def test_hold_expires_and_the_gate_closes_again():
    det, runs = _hold_detector(0.05, [True, False, False])
    det.read_and_detect(detect=True)             # motion: opens the window
    det.read_and_detect(detect=True)             # inside it: still runs
    assert len(runs) == 2
    _time.sleep(0.2)                             # window lapses
    outcome = det.read_and_detect(detect=True)
    assert len(runs) == 2 and not outcome.held   # gate shut again — CPU handed back


def test_further_motion_refreshes_the_window():
    # An active cat is watched continuously: the setting only bounds how long a LULL
    # is tolerated, not how long a cat may be tracked.
    det, runs = _hold_detector(0.3, [True, False, True, False])
    det.read_and_detect(detect=True)
    _time.sleep(0.2)
    det.read_and_detect(detect=True)             # held, near the end of the window
    det.read_and_detect(detect=True)             # real motion: re-arms it
    _time.sleep(0.2)
    outcome = det.read_and_detect(detect=True)   # 0.4s after the FIRST motion
    assert outcome.held and len(runs) == 4


def test_hold_off_is_motion_frames_only():
    det, runs = _hold_detector(0.0, [True, False, False])
    outcomes = [det.read_and_detect(detect=True) for _ in range(3)]
    assert len(runs) == 1                        # pre-0.57 behaviour, exactly
    assert not any(o.held for o in outcomes)


def test_hold_never_overrides_a_detection_pause():
    # The hold gates like motion, NOT like a forced scan: a camera paused for the
    # treat cooldown (or resting under round-robin) must stay off. If this ever
    # regresses, a hold would quietly undo the pause's whole reason to exist.
    det, runs = _hold_detector(5.0, [True, False, False])
    det.read_and_detect(detect=True)             # motion: opens the window
    assert len(runs) == 1
    det.read_and_detect(detect=False)            # paused, mid-hold
    det.read_and_detect(detect=False)
    assert len(runs) == 1


def test_hold_is_hot_reloadable_and_closes_an_open_window():
    det, runs = _hold_detector(5.0, [True, False, False])
    det.read_and_detect(detect=True)
    det.reconfigure({"motion_hold_seconds": 0.0})
    outcome = det.read_and_detect(detect=True)
    assert det.motion_hold_seconds == 0.0
    assert len(runs) == 1 and not outcome.held   # the open window shut immediately
