"""#71: the benchmark-settled model lineup, the golden-export guard, and the
"auto" accelerator (CUDA when it genuinely binds, else CPU — loudly)."""

import numpy as np
import pytest

import d20app.config as config_mod
from d20app import yolo
from d20app.webapp import _model_options


class _FakeRunner:
    def __init__(self, out):
        self._out = out

    def infer(self, blob):
        return self._out


def test_end2end_export_is_refused_loudly():
    frame = np.zeros((240, 320, 3), np.uint8)
    bad = _FakeRunner(np.zeros((1, 300, 6), np.float32))   # NMS-baked head
    with pytest.raises(RuntimeError) as exc:
        yolo.detect_boxes(bad, frame, floor=0.5, size=320)
    msg = str(exc.value)
    assert "end2end" in msg and "(1, 300, 6)" in msg and "golden" in msg


def test_raw_head_still_decodes():
    frame = np.zeros((240, 320, 3), np.uint8)
    good = _FakeRunner(np.zeros((1, 84, 2100), np.float32))
    assert yolo.detect_boxes(good, frame, floor=0.5, size=320) == []


def test_auto_accelerator_falls_back_to_cpu(monkeypatch):
    def boom(path):
        raise RuntimeError("no CUDA anywhere")
    monkeypatch.setattr(yolo, "_OnnxRuntimeRunner", boom)
    runner = yolo.load_net("yolo11n", "auto")
    assert isinstance(runner, yolo._CvDnnRunner)           # CPU runner, not a crash


def test_selectable_lineup_excludes_dropped_models():
    values = [m["value"] for m in _model_options()]
    assert "yolo11n" in values and "yolo26m" in values     # bundled keepers
    assert "yolo11m" not in values                         # dropped (#71)
    assert "yolo11m_960" not in values
    # export-only keepers appear only once their file exists
    import os
    if not os.path.exists(yolo.model_path("yolo26x")):
        assert "yolo26x" not in values
    # ...and a config naming one gets a loud, actionable error (#79/#80)
    assert "yolo11m" in yolo.DROPPED_MODELS
    assert yolo.MODELS["yolo11n"]["size"] == 640    # the benchmarked floor (#80)


def test_benchmark_settled_defaults():
    cfg = config_mod.Config()
    assert cfg.accelerator == "auto"                       # CUDA-if-available (#71)
    assert cfg.cat_scan_tiling == "3x3"                    # 91%/0% pick (#70)
    assert cfg.cat_scan_tile_overlap == 0.35               # 3x3's best overlap (#70)
    assert "auto" in yolo.ACCELERATORS
