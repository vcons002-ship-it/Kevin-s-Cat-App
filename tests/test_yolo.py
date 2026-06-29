"""YOLO11n backend: it loads, detects people, decodes boxes, and falls back."""

import glob
import os

import cv2
import pytest

from d20app import yolo
from d20app.detector import PersonDetector

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PEOPLE = sorted(glob.glob(os.path.join(FIXTURES, "people", "*.jpg")))


def test_model_file_present_and_loads():
    assert os.path.exists(yolo.ONNX_PATH), "bundled yolo11n.onnx is missing"
    assert yolo.load_net() is not None


def test_detect_boxes_format_and_finds_a_person():
    net = yolo.load_net()
    img = cv2.imread(PEOPLE[0])
    boxes = yolo.detect_boxes(net, img, floor=0.25)
    # Box format matches the SSD path: (label, score, (x1,y1,x2,y2)) in frame px.
    for label, score, box in boxes:
        assert isinstance(label, str) and 0.0 <= score <= 1.0 and len(box) == 4
    h, w = img.shape[:2]
    persons = [b for b in boxes if b[0] == "person"]
    assert persons, "YOLO found no person in a clear pedestrian photo"
    (x1, y1, x2, y2) = persons[0][2]
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h     # box maps back inside the frame


def test_persondetector_yolo_detects_people():
    det = PersonDetector(source="unused", confidence=0.4, model="yolo11n")
    hits = sum(det.detect_in_frame(cv2.imread(p)) for p in PEOPLE)
    assert hits >= len(PEOPLE) * 0.8        # strong recall on clear photos


# The bundled variants ship their ONNX; the larger-input locator exports
# (yolo11m_960/_1280) are optional and produced on demand by scripts/export_yolo.py.
_BUNDLED_VARIANTS = ("yolo11n", "yolo11m", "yolo26m")
_CATS = sorted(glob.glob(os.path.join(FIXTURES, "cats", "*.jpg")))


def test_variant_registry_files_present_and_sized():
    # Every variant has a fixed input size; the bundled ones ship their ONNX.
    for variant, spec in yolo.MODELS.items():
        assert yolo.input_size(variant) == spec["size"]
        if variant in _BUNDLED_VARIANTS:
            assert os.path.exists(yolo.model_path(variant)), f"{variant} onnx missing"
    assert yolo.ONNX_PATH == yolo.model_path("yolo11n")
    assert yolo.INPUT_SIZE == yolo.input_size("yolo11n")


def test_load_net_rejects_unknown_variant():
    with pytest.raises(ValueError):
        yolo.load_net("yolo11xl")


def test_yolo11m_loads_and_finds_a_person():
    net = yolo.load_net("yolo11m")
    img = cv2.imread(PEOPLE[0])
    boxes = yolo.detect_boxes(net, img, floor=0.25, size=yolo.input_size("yolo11m"))
    assert any(b[0] == "person" for b in boxes), "yolo11m found no person in a clear photo"


def test_persondetector_yolo11m_detects_a_person():
    det = PersonDetector(source="unused", confidence=0.4, model="yolo11m")
    assert det.detect_in_frame(cv2.imread(PEOPLE[0])) is True
    assert det.model == "yolo11m" and det._yolo_size == 640


def test_yolo26m_raw_head_decodes_and_finds_a_cat():
    # #45: YOLO26 ships its raw (1,84,N) head (exported with end2end=False), so the
    # existing cv2.dnn decode works unchanged — the NMS-free e2e export does not.
    net = yolo.load_net("yolo26m")
    out = net.infer(__import__("numpy").zeros((1, 3, 640, 640), "float32"))
    assert out.shape[1] == 84, "yolo26m must be the raw head (1,84,N), not e2e (1,300,6)"
    boxes = yolo.detect_boxes(net, cv2.imread(_CATS[0]), floor=0.25, size=640)
    for label, score, box in boxes:           # same box format as the other variants
        assert isinstance(label, str) and 0.0 <= score <= 1.0 and len(box) == 4
    assert any(b[0] == "cat" for b in boxes), "yolo26m found no cat in a clear photo"


def test_unknown_accelerator_rejected():
    with pytest.raises(ValueError):
        yolo.load_net("yolo11n", accelerator="cuda")


def test_opencl_accelerator_loads_and_detects():
    # OpenCV silently runs on CPU when no OpenCL device is present, so this must
    # construct and still find a person regardless of the host's GPU.
    runner = yolo.load_net("yolo11n", accelerator="opencl")
    boxes = yolo.detect_boxes(runner, cv2.imread(PEOPLE[0]), floor=0.25)
    assert any(b[0] == "person" for b in boxes)


def test_openvino_runner_matches_cpu_when_installed():
    # If the optional OpenVINO runtime is present we can verify the integration
    # end-to-end on its CPU device (no GPU needed): it must find the same person.
    ov = pytest.importorskip("openvino")
    if "CPU" not in ov.Core().available_devices:        # pragma: no cover
        pytest.skip("no OpenVINO CPU device")
    runner = yolo._OpenVinoRunner(yolo.model_path("yolo11n"), "CPU")
    boxes = yolo.detect_boxes(runner, cv2.imread(PEOPLE[0]), floor=0.25)
    assert any(b[0] == "person" for b in boxes), "OpenVINO found no person"


def test_detector_falls_back_to_cpu_when_accelerator_fails(monkeypatch):
    # A dead GPU backend must not lose us the model: retry the same YOLO on CPU.
    real_load = yolo.load_net

    def fake_load(variant, accelerator="cpu"):
        if accelerator != "cpu":
            raise RuntimeError("no Intel GPU here")
        return real_load(variant, "cpu")

    monkeypatch.setattr(yolo, "load_net", fake_load)
    det = PersonDetector(source="unused", confidence=0.4,
                         model="yolo11n", accelerator="openvino-gpu")
    assert det.detect_in_frame(cv2.imread(PEOPLE[0])) is True
    assert det.model == "yolo11n" and det.accelerator == "cpu"   # kept model, fell to CPU


def test_falls_back_to_mobilenet_when_yolo_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("no onnx here")
    monkeypatch.setattr(yolo, "load_net", boom)
    det = PersonDetector(source="unused", confidence=0.4, model="yolo11n")
    # First detection triggers the load, which fails and silently downgrades.
    assert det.detect_in_frame(cv2.imread(PEOPLE[0])) in (True, False)
    assert det.model == "mobilenet_ssd"
