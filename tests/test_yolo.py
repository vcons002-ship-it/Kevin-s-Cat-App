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


def test_falls_back_to_mobilenet_when_yolo_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("no onnx here")
    monkeypatch.setattr(yolo, "load_net", boom)
    det = PersonDetector(source="unused", confidence=0.4, model="yolo11n")
    # First detection triggers the load, which fails and silently downgrades.
    assert det.detect_in_frame(cv2.imread(PEOPLE[0])) in (True, False)
    assert det.model == "mobilenet_ssd"
