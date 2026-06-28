"""Image adjustments + the "Test detection" tool (upload → tune → draw boxes).

Covers the pure adjustment maths, the stateless ``detect_image`` path, video
frame sampling, the new config fields (global + per-camera), and the
``/api/test/upload`` + ``/api/test/detect`` endpoints.
"""

import io
import os
import tempfile

import numpy as np
import pytest

import d20app.config as config_mod
from d20app import detector
from d20app.config import Config
from d20app.detector import (PersonDetector, apply_image_adjustments,
                             sample_video_frames)
from d20app.webapp import create_app

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_CAT = os.path.join(_FIXTURES, "cats", "cat01.jpg")


# ---- image adjustments (pure maths) ---------------------------------------
def test_adjustments_noop_returns_same_array():
    frame = np.full((8, 8, 3), 120, np.uint8)
    # All defaults → no work, same object back (cheap fast-path).
    assert apply_image_adjustments(frame) is frame


def test_brightness_raises_mean():
    frame = np.full((8, 8, 3), 100, np.uint8)
    out = apply_image_adjustments(frame, brightness=40)
    assert out.mean() > frame.mean() + 30


def test_gamma_above_one_brightens_midtones():
    frame = np.full((8, 8, 3), 100, np.uint8)
    assert apply_image_adjustments(frame, gamma=2.0).mean() > frame.mean()
    assert apply_image_adjustments(frame, gamma=0.5).mean() < frame.mean()


def test_saturation_zero_is_greyscale():
    frame = np.zeros((4, 4, 3), np.uint8)
    frame[..., 2] = 200        # pure-ish red
    out = apply_image_adjustments(frame, saturation=0.0)
    # With no saturation the three channels collapse to one grey value.
    assert int(out[..., 0].max()) == int(out[..., 1].max()) == int(out[..., 2].max())


# ---- detect_image (stateless single-frame detection) ----------------------
def test_detect_image_returns_jpeg_and_detections():
    import cv2

    frame = cv2.imread(_CAT)
    assert frame is not None
    det = PersonDetector(source="__test__", model="mobilenet_ssd",
                         detect_size=512, label_floor=0.4)
    annotated, dets = det.detect_image(frame)
    assert annotated and annotated[:2] == b"\xff\xd8"      # JPEG magic
    assert any(d["label"] == "cat" for d in dets)
    # Sorted strongest-first, and each carries a pixel box.
    assert dets == sorted(dets, key=lambda d: -d["score"])
    assert all(len(d["box"]) == 4 for d in dets)


def test_detect_image_respects_confidence_floor():
    import cv2

    frame = cv2.imread(_CAT)
    high = PersonDetector(source="x", model="mobilenet_ssd", detect_size=512,
                          confidence=0.99, label_floor=0.99)
    _, dets = high.detect_image(frame)
    assert dets == []      # nothing clears a 0.99 floor


# ---- video sampling --------------------------------------------------------
def test_sample_video_frames_evenly():
    import cv2

    path = tempfile.mktemp(suffix=".mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not vw.isOpened():
        pytest.skip("no mp4 writer codec available in this environment")
    try:
        for i in range(20):
            vw.write(np.full((48, 64, 3), (i * 12) % 255, np.uint8))
        vw.release()
        frames = sample_video_frames(path, 5)
        assert 1 <= len(frames) <= 5
        assert all(f.shape == (48, 64, 3) for f in frames)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def test_sample_video_frames_bad_path_returns_empty():
    assert sample_video_frames("/no/such/file.mp4", 4) == []


# ---- config: new fields, global + per-camera ------------------------------
def test_image_adjust_defaults_are_noops():
    c = Config()
    assert (c.gamma, c.brightness, c.contrast, c.saturation) == (1.0, 0, 1.0, 1.0)


def test_camera_inherits_and_overrides_adjustments():
    cfg = Config(gamma=1.8, contrast=1.2)
    inherited = config_mod.coerce_camera({"name": "A"}, cfg)
    assert inherited["gamma"] == 1.8 and inherited["contrast"] == 1.2
    overridden = config_mod.coerce_camera({"name": "B", "gamma": 0.7}, cfg)
    assert overridden["gamma"] == 0.7 and overridden["contrast"] == 1.2


# ---- endpoints -------------------------------------------------------------
def _client():
    return create_app().test_client()


def _upload(c, path, filename="cat01.jpg"):
    with open(path, "rb") as fh:
        data = fh.read()
    return c.post("/api/test/upload",
                  data={"file": (io.BytesIO(data), filename)},
                  content_type="multipart/form-data")


def test_test_upload_and_detect_roundtrip():
    c = _client()
    up = _upload(c, _CAT).get_json()
    assert up["count"] == 1 and up["kind"] == "image"
    assert up["width"] > 0 and up["thumbs"][0].startswith("data:image/jpeg;base64,")
    r = c.post("/api/test/detect", json={
        "id": up["id"], "frame_index": 0,
        "settings": {"model": "mobilenet_ssd", "detect_size": 512, "label_floor": 0.4},
    })
    body = r.get_json()
    assert r.status_code == 200
    assert body["annotated"].startswith("data:image/jpeg;base64,")
    assert any(d["label"] == "cat" for d in body["detections"])


def test_test_detect_expired_session_is_404():
    c = _client()
    r = c.post("/api/test/detect", json={"id": "nope", "settings": {}})
    assert r.status_code == 404


def test_test_upload_rejects_non_media():
    c = _client()
    r = c.post("/api/test/upload",
               data={"file": (io.BytesIO(b"not an image or video"), "notes.txt")},
               content_type="multipart/form-data")
    assert r.status_code == 400


def test_test_upload_requires_a_file():
    assert _client().post("/api/test/upload").status_code == 400


def test_test_detect_returns_inference_ms_and_respects_tiling():
    c = _client()
    up = _upload(c, _CAT).get_json()
    body = c.post("/api/test/detect", json={"id": up["id"], "settings": {
        "model": "mobilenet_ssd", "detect_size": 512, "label_floor": 0.4,
        "tiling": "2x2", "tile_overlap": 0.2}}).get_json()
    assert isinstance(body["inference_ms"], (int, float)) and body["inference_ms"] >= 0
    assert any(d["label"] == "cat" for d in body["detections"])


# ---- locator path: tiling + larger-input ----------------------------------
def test_merge_nms_dedupes_overlapping_same_class():
    from d20app import yolo

    boxes = [("cat", 0.9, (10, 10, 110, 110)), ("cat", 0.8, (14, 12, 112, 108)),
             ("person", 0.7, (300, 0, 360, 300))]
    merged = yolo.merge_nms(boxes, 0.3)
    assert sorted(lab for lab, _, _ in merged) == ["cat", "person"]   # 2 cats → 1
    cat = next(m for m in merged if m[0] == "cat")
    assert cat[1] == 0.9        # the stronger of the overlapping pair survives


def test_locator_tiling_runs_once_per_tile():
    det = PersonDetector(source="x", model="mobilenet_ssd",
                         cat_scan_tiling="2x2", cat_scan_tile_overlap=0.2)
    shapes = []
    det._run_net = lambda img, floor, size=None: (shapes.append(img.shape), [])[1]
    det._detect_locator(np.zeros((100, 120, 3), np.uint8), 0.3)
    assert len(shapes) == 4        # a 2x2 grid → four single passes


def test_locator_tiling_finds_cat_via_detect_image():
    import cv2

    frame = cv2.imread(_CAT)
    det = PersonDetector(source="x", model="mobilenet_ssd", label_floor=0.4,
                         cat_scan_tiling="2x2")
    _, dets = det.detect_image(frame)
    assert any(d["label"] == "cat" for d in dets)


def test_locator_imgsz_missing_yolo_variant_falls_back():
    import cv2

    frame = cv2.imread(_CAT)
    # yolo11m_1280 isn't committed (only the 960 export is) — selecting it must
    # degrade to the native size, not crash.
    det = PersonDetector(source="x", model="yolo11m", cat_scan_imgsz=1280,
                         cat_scan_tiling="off", label_floor=0.4)
    assert det._locator_net() is None
    annotated, _ = det.detect_image(frame)
    assert annotated is not None
