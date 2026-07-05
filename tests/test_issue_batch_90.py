"""Issues #90/#91/#98: precision resolved from the accelerator (not the model
picker), effective-accelerator surfacing, quick-toggle merge saves, and
cat-not-dog log wording.
"""

import os

import d20app.config as config_mod
from d20app import yolo
from d20app.loop import DetectionLoop, _shown_label
from d20app.webapp import _model_options, create_app


# ---- #90: resolve_variant — precision is the accelerator's business ------------
def _have(*variants):
    paths = {yolo.model_path(v) for v in variants}
    return lambda p: p in paths


def test_resolve_cpu_is_always_fp32():
    assert yolo.resolve_variant("yolo26x", "cpu",
                                exists=_have("yolo26x", "yolo26x_fp16")) == "yolo26x"
    assert yolo.resolve_variant("yolo26x", "opencl",
                                exists=_have("yolo26x_fp16")) == "yolo26x"


def test_resolve_cuda_prefers_fp16_when_present():
    ex = _have("yolo26x", "yolo26x_fp16")
    assert yolo.resolve_variant("yolo26x", "onnx-cuda", exists=ex) == "yolo26x_fp16"
    assert yolo.resolve_variant("yolo26x", "auto", exists=ex) == "yolo26x_fp16"
    assert yolo.resolve_variant("yolo26x", "tensorrt", exists=ex) == "yolo26x_fp16"
    assert yolo.resolve_variant("yolo26x", "openvino-gpu", exists=ex) == "yolo26x_fp16"
    # no fp16 file → the base still works on CUDA (just FP32)
    assert yolo.resolve_variant("yolo26x", "onnx-cuda",
                                exists=_have("yolo26x")) == "yolo26x"


def test_resolve_normalizes_legacy_fp16_configs():
    # an old config naming yolo26x_fp16 can never force FP16 onto cv2.dnn
    assert yolo.resolve_variant("yolo26x_fp16", "cpu",
                                exists=_have("yolo26x", "yolo26x_fp16")) == "yolo26x"
    assert yolo.resolve_variant("yolo26x_fp16", "onnx-cuda",
                                exists=_have("yolo26x", "yolo26x_fp16")) == "yolo26x_fp16"


def test_fp16_variants_hidden_from_pickers():
    values = {m["value"] for m in _model_options()}
    assert not any(v.endswith("_fp16") for v in values)
    assert "yolo11n" in values and "yolo26m" in values


# ---- #90: what actually ran is stamped on the runner ----------------------------
def test_cpu_runner_annotated():
    net = yolo.load_net("yolo11n", "cpu")
    assert net.effective_accelerator == "cpu"
    assert net.fallback_reason == ""


def test_tensorrt_fallback_reason_is_carried():
    # CI has no driver: tensorrt → auto → cpu, and the whole story is on the runner
    net = yolo.load_net("yolo11n", "tensorrt")
    assert net.effective_accelerator == "cpu"
    assert "tensorrt" in net.fallback_reason and "CUDA" in net.fallback_reason


def test_cam_status_reports_effective_accelerator(tmp_path):
    loop = DetectionLoop()

    class _Det:
        def effective_accelerator(self):
            return "cpu", "requested tensorrt — unavailable: no driver"

    loop._detectors = {"Room": _Det()}
    loop._cam_status = {"Room": {"connected": True, "roll": True,
                                 "track_cats": True, "always_watch": False,
                                 "last_error": "", "resting": False}}
    row = loop.cam_status()[0]
    assert row["ran_on"] == "cpu" and "tensorrt" in row["fallback"]


# ---- #91: a partial save merges — other fields and cameras untouched -------------
def _cfg_isolated(monkeypatch, tmp_path):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))


def test_quick_toggle_partial_save_merges(monkeypatch, tmp_path):
    _cfg_isolated(monkeypatch, tmp_path)
    c = create_app().test_client()
    c.post("/api/cameras/saved", json={
        "name": "Kitchen", "url": "rtsp://a", "roll": True, "track_cats": True,
        "person_confidence": 0.7})
    c.post("/api/cameras/saved", json={"name": "Porch", "url": "rtsp://b",
                                       "roll": True})
    # the #91 quick-toggle payload: name + url + ONE field
    c.post("/api/cameras/saved", json={"name": "Kitchen", "url": "rtsp://a",
                                       "roll": False})
    cams = {cam["name"]: cam for cam in
            c.get("/api/cameras/saved").get_json()}
    assert cams["Kitchen"]["roll"] is False              # the toggle stuck
    assert cams["Kitchen"]["track_cats"] is True         # nothing else changed
    assert cams["Kitchen"]["person_confidence"] == 0.7
    assert cams["Porch"]["roll"] is True                 # other camera untouched


# ---- #98: counted-as-cat is CALLED cat -------------------------------------------
def test_shown_label_says_cat_for_locator_hits():
    assert _shown_label("dog") == "cat"
    assert _shown_label("cat") == "cat"
