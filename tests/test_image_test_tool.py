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
    # Cat now has its own threshold, so raise it too (a real cat scores ~0.96).
    high = PersonDetector(source="x", model="mobilenet_ssd", detect_size=512,
                          confidence=0.99, label_floor=0.99, cat_confidence=0.99)
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


# ---- benchmark (#21) -------------------------------------------------------
def _benchmark(c, **kw):
    up = _upload(c, _CAT).get_json()
    payload = {"id": up["id"], "models": ["mobilenet_ssd"], "tilings": ["off", "2x2"]}
    payload.update(kw)
    return c.post("/api/test/benchmark", json=payload)


def test_benchmark_sweeps_and_builds_reports():
    c = _client()
    body = _benchmark(c).get_json()
    runs = body["runs"]
    assert len(runs) == 2                       # 1 model × 2 tilings
    assert {r["tiling"] for r in runs} == {"off", "2x2"}
    r = runs[0]
    assert "combined_score" in r and "inference_ms" in r and r["thumb"].startswith("data:image")
    assert any(x["cat_score"] > 0.5 for x in runs)   # the cat is found
    # runs are sorted best-first
    assert [x["combined_score"] for x in runs] == sorted(
        (x["combined_score"] for x in runs), reverse=True)
    # self-contained HTML — inline images, no external asset refs
    html = c.get(body["html_url"]).get_data(as_text=True)
    assert "data:image/jpeg;base64," in html and "<table" in html
    assert "src=\"http" not in html and "src='http" not in html


def test_benchmark_xlsx_downloads_when_openpyxl_present():
    c = _client()
    body = _benchmark(c).get_json()
    assert body["xlsx_url"] and body["xlsx_error"] is None
    xl = c.get(body["xlsx_url"])
    assert xl.status_code == 200 and xl.data[:2] == b"PK"   # xlsx is a zip


def test_benchmark_xlsx_degrades_without_openpyxl(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openpyxl", None)   # make `import openpyxl` fail
    c = _client()
    body = _benchmark(c).get_json()
    assert body["xlsx_url"] is None
    assert "openpyxl" in (body["xlsx_error"] or "")
    # HTML report is still produced.
    assert c.get(body["html_url"]).status_code == 200


def test_benchmark_caps_matrix_size():
    c = _client()
    up = _upload(c, _CAT).get_json()
    big = {"id": up["id"], "models": ["mobilenet_ssd"] * 7, "tilings": ["off", "2x2", "3x3", "4x4"]}
    assert c.post("/api/test/benchmark", json=big).status_code == 400


def test_benchmark_html_embeds_full_res_original_and_zoomable_thumbs():
    # #25: the report carries the unannotated original frame (download to re-run)
    # and each run thumbnail is click-to-enlarge.
    c = _client()
    body = _benchmark(c, name="kitchen-cam.jpg").get_json()
    html = c.get(body["html_url"]).get_data(as_text=True)
    assert "Original frame" in html and "download='kitchen-cam.jpg'" in html
    # thumbnails are click-to-enlarge via the in-page lightbox, not a data: nav
    assert "onclick='zoom(this.src)'" in html and "<img" in html


def test_benchmark_html_uses_lightbox_not_data_url_navigation():
    # #30: clicking a thumbnail must enlarge in-page; browsers block top-level
    # navigation to a data: URL (lands on about:blank). So no <a href='data:...'
    # target=_blank> wrappers — only the download link keeps its href.
    c = _client()
    html = c.get(_benchmark(c).get_json()["html_url"]).get_data(as_text=True)
    assert "id='lb'" in html and "function zoom(" in html
    assert "onclick='zoom(this.src)'" in html         # thumbnails enlarge in-page
    assert "target='_blank'" not in html              # no blocked data: navigation
    assert "download='" in html                        # download link (download=) kept


def test_benchmark_report_strips_at_size_from_model_label():
    # #31: SSD size variants shouldn't read "mobilenet_ssd@512" next to size 512 —
    # the model column shows the base name, the size column carries the number.
    c = _client()
    body = _benchmark(c, models=["mobilenet_ssd@512"], tilings=["off"]).get_json()
    assert body["runs"][0]["model"] == "mobilenet_ssd@512"   # key unchanged (re-runs)
    html = c.get(body["html_url"]).get_data(as_text=True)
    assert "mobilenet_ssd@512" not in html              # not in the visible label
    assert "<td>mobilenet_ssd</td>" in html             # base name in the model column


def test_benchmark_reports_have_human_readable_filenames():
    # #27: Content-Disposition uses a slug from the uploaded image name, not the uuid.
    c = _client()
    body = _benchmark(c, name="Living Room Cam.jpg").get_json()
    cd = c.get(body["html_url"]).headers.get("Content-Disposition", "")
    assert "benchmark-living-room-cam-" in cd and cd.endswith('.html"')
    xl = c.get(body["xlsx_url"])
    assert "benchmark-living-room-cam-" in xl.headers.get("Content-Disposition", "")


def test_benchmark_model_list_includes_ssd_size_variants():
    # #28: the default sweep list must not drift from the dropdown — it carries the
    # MobileNet size variants alongside any present YOLO models.
    c = _client()
    models = c.get("/api/test/benchmark/models").get_json()["models"]
    assert {"mobilenet_ssd", "mobilenet_ssd@512", "mobilenet_ssd@768"} <= set(models)


# ---- batch benchmark + cross-image summary (#32) ---------------------------
def _upload_blank(c, filename="empty-room.jpg"):
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))
    return c.post("/api/test/upload",
                  data={"file": (io.BytesIO(buf.tobytes()), filename)},
                  content_type="multipart/form-data").get_json()


def test_benchmark_batch_aggregates_and_summarizes():
    c = _client()
    a = _upload(c, _CAT, "kitchen.jpg").get_json()
    b = _upload(c, _CAT, "hallway.jpg").get_json()
    res = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": a["id"], "name": "kitchen.jpg", "has_cat": True},
                  {"id": b["id"], "name": "hallway.jpg", "has_cat": True}],
        "models": ["mobilenet_ssd"], "tilings": ["off", "2x2"],
    })
    body = res.get_json()
    assert res.status_code == 200
    assert len(body["images"]) == 2 and body["meta"]["n_images"] == 2
    # one aggregate per config (1 model × 2 tilings), with cross-image fields
    assert len(body["configs"]) == 2
    cfg = body["configs"][0]
    for k in ("found", "total", "rate", "avg_score", "avg_ms", "misses", "fp"):
        assert k in cfg
    assert cfg["total"] == 2                      # both images are cat-present
    # the cat is the same fixture twice, so a working config finds it in both
    assert any(c2["found"] == 2 for c2 in body["configs"])
    # summary report is self-contained-ish: lightbox + grid, links to per-image reports
    html = c.get(body["summary_url"]).get_data(as_text=True)
    assert "cross-image summary" in html and "config × image" in html
    assert "function zoom(" in html and "target='_blank'" not in html.split("<script")[0]
    # per-image reports are reachable
    assert c.get(body["images"][0]["report_url"]).status_code == 200


def test_benchmark_batch_nocat_control_reports_false_positives():
    c = _client()
    cat = _upload(c, _CAT, "with-cat.jpg").get_json()
    empty = _upload_blank(c)
    body = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": cat["id"], "name": "with-cat.jpg", "has_cat": True},
                  {"id": empty["id"], "name": "empty-room.jpg", "has_cat": False}],
        "models": ["mobilenet_ssd"], "tilings": ["off"],
    }).get_json()
    assert body["meta"]["n_cat"] == 1 and body["meta"]["n_nocat"] == 1
    cfg = body["configs"][0]
    assert cfg["total"] == 1 and cfg["fp_total"] == 1     # 1 cat image, 1 control
    assert cfg["fp"] == 0                                  # blank frame → no false fire
    html = c.get(body["summary_url"]).get_data(as_text=True)
    assert "false-positive check" in html


def test_benchmark_batch_rejects_empty_and_oversized():
    c = _client()
    assert c.post("/api/test/benchmark/batch", json={"items": []}).status_code == 400
    big = {"items": [{"id": "x", "name": str(i)} for i in range(20)],
           "models": ["mobilenet_ssd"], "tilings": ["off"]}
    assert c.post("/api/test/benchmark/batch", json=big).status_code == 400


def test_benchmark_batch_404_when_all_uploads_expired():
    c = _client()
    r = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": "gone", "name": "x.jpg"}],
        "models": ["mobilenet_ssd"], "tilings": ["off"]})
    assert r.status_code == 404


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


def test_cat_confidence_gates_independently_of_label_floor():
    import cv2

    frame = cv2.imread(_CAT)
    # A real cat scores ~0.96. Cat is gated by cat_confidence, not label_floor.
    seen = PersonDetector(source="x", model="mobilenet_ssd", detect_size=512,
                          cat_confidence=0.5, label_floor=0.9)
    assert any(d["label"] == "cat" for d in seen.detect_image(frame)[1])
    hidden = PersonDetector(source="x", model="mobilenet_ssd", detect_size=512,
                            cat_confidence=0.97, label_floor=0.2)
    assert not any(d["label"] == "cat" for d in hidden.detect_image(frame)[1])


def test_locator_classes_count_a_dog_as_the_cat():
    import numpy as np

    det = PersonDetector(source="x", model="mobilenet_ssd",
                         cat_confidence=0.4, locator_classes=("cat", "dog"))
    det._run_net = lambda img, floor, size=None: [("dog", 0.8, (5, 5, 50, 50))]
    assert det._is_locator_hit("dog", 0.8) is True
    _, dets = det.detect_image(np.zeros((80, 80, 3), np.uint8))
    assert [d["label"] for d in dets] == ["dog"]
    # Default cat-only doesn't count a dog.
    det2 = PersonDetector(source="x", model="mobilenet_ssd", cat_confidence=0.4)
    assert det2._is_locator_hit("dog", 0.9) is False


def test_mobilenet_ssd_size_suffix_parsed():
    assert PersonDetector(source="x", model="mobilenet_ssd@512")._ssd_size() == 512
    assert PersonDetector(source="x", model="mobilenet_ssd", detect_size=300)._ssd_size() == 300


def test_test_detect_honours_cat_confidence_and_dog():
    c = _client()
    up = _upload(c, _CAT).get_json()
    # A high cat_confidence drops the cat from the tester's detection list.
    body = c.post("/api/test/detect", json={"id": up["id"], "settings": {
        "model": "mobilenet_ssd", "detect_size": 512,
        "cat_confidence": 0.99, "label_floor": 0.2}}).get_json()
    assert not any(d["label"] == "cat" for d in body["detections"])


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
