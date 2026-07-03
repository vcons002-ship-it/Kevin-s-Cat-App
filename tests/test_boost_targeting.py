"""Targeted boost + last-known overlay (0.42.0): an unconfirmed lead's box aims
forced scans (full-res zoom crop + the heaviest model on disk, per #70), and the
live feed carries a grey, age-labelled "last known location" box so "where was
she last?" stays answered even when nothing is detected right now.

What only the NAS shows: the heavier boost model's real latency on the 3070/CPU
and whether the targeted zoom converts real VLM leads — flagged in the PR.
"""

import numpy as np

from d20app import yolo
from d20app.cats import CatTracker
from d20app.detector import PersonDetector
from d20app.loop import DetectionLoop
from d20app.webapp import create_app


# ---- boost_variant: strongest available model, never a downgrade --------------
def test_boost_variant_prefers_heavier_fp32_on_cpu():
    have = {yolo.model_path(v) for v in ("yolo26x", "yolo26m", "yolo11n")}
    assert yolo.boost_variant("yolo11n", "cpu", exists=lambda p: p in have) == "yolo26x"


def test_boost_variant_skips_missing_files():
    have = {yolo.model_path("yolo26m")}
    assert yolo.boost_variant("yolo11n", "cpu", exists=lambda p: p in have) == "yolo26m"
    assert yolo.boost_variant("yolo11n", "cpu", exists=lambda p: False) is None


def test_boost_variant_fp16_only_on_cuda():
    have = {yolo.model_path(v) for v in ("yolo26x_fp16", "yolo26x")}
    within = lambda p: p in have
    assert yolo.boost_variant("yolo11n", "onnx-cuda", exists=within) == "yolo26x_fp16"
    # FP16 pays only on CUDA; "auto" may resolve to CPU, so it gets FP32.
    assert yolo.boost_variant("yolo11n", "cpu", exists=within) == "yolo26x"
    assert yolo.boost_variant("yolo11n", "auto", exists=within) == "yolo26x"


def test_boost_variant_never_downgrades():
    always = lambda p: True
    assert yolo.boost_variant("yolo26x", "cpu", exists=always) is None
    assert yolo.boost_variant("yolo26x_fp16", "onnx-cuda", exists=always) is None
    assert yolo.boost_variant("yolo26m", "cpu", exists=always) == "yolo26x"


# ---- detector: the hint aims a zoom pass on forced scans ----------------------
def _det_reading(frame):
    det = PersonDetector(source="unused", cat_scan_frames=1)

    class _Cap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _Cap()
    return det


def test_boost_hint_expires():
    det = PersonDetector(source="unused")
    assert det.boost_hint() is None
    det.set_boost_hint((10, 10, 50, 50), 0.0)
    assert det.boost_hint() is None
    det.set_boost_hint((10, 10, 50, 50), 30.0)
    assert det.boost_hint() == (10, 10, 50, 50)


def test_forced_scan_zooms_the_hint_and_maps_back():
    det = _det_reading(np.full((360, 640, 3), 110, np.uint8))
    det._boost_net = lambda: None          # boost degrades to the camera's own net
    calls = []

    def run_net(img, floor, size=None):
        calls.append(img.shape)
        if img.shape[0] < 360:             # the zoom crop, not the full frame
            return [("cat", 0.8, (10, 10, 50, 50))]
        return []

    det._run_net = run_net
    det.set_boost_hint((300, 120, 380, 200), 30.0)
    outcome = det.read_and_detect(force=True)
    assert "cat" in outcome.labels
    assert any(shape[0] < 360 for shape in calls)   # the crop pass really ran
    box = next(b for lab, _s, b in det._last_boxes if lab == "cat")
    # crop coords mapped back into frame coords (the window sits right of origin)
    assert box[0] > 100 and box[2] <= 640


def test_forced_scan_without_hint_runs_no_extra_pass():
    det = _det_reading(np.full((360, 640, 3), 110, np.uint8))
    calls = []
    det._run_net = lambda img, floor, size=None: calls.append(img.shape) or []
    det.read_and_detect(force=True)
    assert len(calls) == 1                 # just the locator pass


def test_boost_net_failure_degrades_and_does_not_retry(monkeypatch):
    det = PersonDetector(source="unused")
    attempts = []
    monkeypatch.setattr(yolo, "boost_variant", lambda cur, acc: "yolo26x")

    def boom(variant, accelerator):
        attempts.append(variant)
        raise RuntimeError("file missing")

    monkeypatch.setattr(yolo, "load_net", boom)
    assert det._boost_net() is None
    assert det._boost_net() is None
    assert attempts == ["yolo26x"]         # tried once, then remembered


# ---- loop + feed: box plumbed through; last-known overlay ----------------------
class _Alive:
    def is_alive(self):
        return True


def _running_loop(tmp_path, cam="Room"):
    loop = DetectionLoop()
    loop._thread = _Alive()
    det = PersonDetector(source="unused")
    loop._detectors = {cam: det}
    loop._live_name = cam
    loop.cats = CatTracker(path=str(tmp_path / "cats.log"))   # isolate from repo file
    return loop, det


def test_boost_detection_passes_the_box_to_the_detector(tmp_path):
    loop, det = _running_loop(tmp_path)
    assert loop.boost_detection("Room", 10.0, box=(280, 100, 360, 180))
    assert det.boost_hint() == (280, 100, 360, 180)
    assert loop.boost_detection("Ghost", 10.0, box=(1, 1, 2, 2)) is False


def test_last_for_picks_the_camera(tmp_path):
    t = CatTracker(path=str(tmp_path / "c.log"))
    t.record("A", (1, 1, 5, 5), (100, 100), 0.5)
    t.record("B", (2, 2, 6, 6), (100, 100), 0.6)
    t.record("A", (3, 3, 7, 7), (100, 100), 0.7)
    assert t.last_for("A")["box"][0] == 3          # newest A, not the first
    assert t.last_for("B")["score"] == 0.6
    assert t.last_for("C") is None


def test_live_feed_draws_last_known_by_default(tmp_path):
    loop, det = _running_loop(tmp_path)
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    plain = loop.live_jpeg("Room", last_known=False)
    assert plain is not None
    assert loop.live_jpeg("Room") == plain          # nothing recorded yet → plain
    loop.cats.record("Room", (280, 100, 360, 180), (640, 360), 0.8)
    assert loop.live_jpeg("Room") != plain          # grey box + age label drawn
    assert loop.live_jpeg("Room", last_known=False) == plain   # opt-out honoured


def test_last_known_expires_after_ttl(tmp_path):
    loop, det = _running_loop(tmp_path)
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    plain = loop.live_jpeg("Room", last_known=False)
    loop.cats.record("Room", (280, 100, 360, 180), (640, 360), 0.8)
    loop.cats._sightings[-1]["ts"] -= 4000.0        # age it past _LAST_KNOWN_TTL
    assert loop.live_jpeg("Room") == plain


def test_live_feed_draws_the_lead_box_while_boosting(tmp_path):
    loop, det = _running_loop(tmp_path)
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))
    plain = loop.live_jpeg("Room")
    loop.boost_detection("Room", 10.0, box=(280, 100, 360, 180))
    assert loop.live_jpeg("Room") != plain          # orange "checking (lead)" box


def test_stream_last_known_param(monkeypatch):
    app = create_app()
    loop = app.config["loop"]
    seen = []
    monkeypatch.setattr(loop, "is_running", lambda: True)
    monkeypatch.setattr(loop, "live_jpeg",
                        lambda name=None, trail=False, last_known=True:
                        seen.append(last_known) or b"\xff\xd8s\xff\xd9")
    r = app.test_client().get("/api/stream?last_known=0")
    next(r.response)
    r.close()
    assert seen and seen[0] is False
    r2 = app.test_client().get("/api/stream")
    next(r2.response)
    r2.close()
    assert seen[-1] is True                         # default is ON
