"""Regression test: the bundled model must detect people and ignore cats.

This is the guard that would have caught the shipped-broken model (a training
snapshot whose weights didn't match the prototxt, so detection scored 0 on
everything). It runs the real PersonDetector over a few bundled real photos.

Fixtures: tests/fixtures/people/ (PennFudanPed pedestrians, many rear-view),
tests/fixtures/people_hard/ (people in hats/helmets/headgear), and
tests/fixtures/cats/ (ImageNet domestic cats). All downscaled.
"""

import glob
import os

import cv2
import pytest

from d20app.detector import PersonDetector

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PEOPLE = sorted(glob.glob(os.path.join(FIXTURES, "people", "*.jpg")))
PEOPLE_HARD = sorted(glob.glob(os.path.join(FIXTURES, "people_hard", "*.jpg")))
CATS = sorted(glob.glob(os.path.join(FIXTURES, "cats", "*.jpg")))


def _detector():
    # Mirror the app defaults (detect_size 300, confidence 0.4).
    return PersonDetector(source="unused", confidence=0.4)


def test_fixtures_present():
    assert PEOPLE, "no people fixtures found"
    assert CATS, "no cat fixtures found"


def test_people_are_detected():
    """At least 65% of people images must trigger (we measure ~99% on 170)."""
    det = _detector()
    hits = sum(det.detect_in_frame(cv2.imread(p)) for p in PEOPLE)
    rate = hits / len(PEOPLE)
    assert rate >= 0.65, f"person recall too low: {hits}/{len(PEOPLE)}"


def test_cats_do_not_trigger():
    """No cat image may be detected as a person (the whole point of the app)."""
    det = _detector()
    for p in CATS:
        assert not det.detect_in_frame(cv2.imread(p)), f"cat triggered person: {p}"


def test_hard_pose_people_are_detected():
    """People in head accessories (hats/helmets/headgear) must still trigger.

    Guards against detection regressing on harder real-world cases — the model
    scores these 0.88–1.00. Back-turned people are covered by the PennFudan
    `people/` fixtures (street pedestrians, many walking away from the camera).
    """
    det = _detector()
    assert PEOPLE_HARD, "no hard-case fixtures found"
    misses = [p for p in PEOPLE_HARD if not det.detect_in_frame(cv2.imread(p))]
    assert not misses, f"hard-pose people missed: {[os.path.basename(p) for p in misses]}"


def test_distant_cats_are_identified_at_high_detail():
    """At the selectable 512px detail, a cat ~1/4 of the frame is detected.

    512 is no longer the default (reverted to 300 to protect person recall), but
    it stays available for users who want distant-cat detection — so we assert
    the capability explicitly at detect_size=512.
    """
    import numpy as np

    det = PersonDetector(source="unused", confidence=0.4, detect_size=512)
    hits = 0
    for p in CATS:
        cat = cv2.imread(p)
        bg = np.full((720, 1280, 3), 110, np.uint8)
        ch = 150                          # ~1/4 of the 720px-tall frame
        cw = int(cat.shape[1] * ch / cat.shape[0])
        bg[285:285 + ch, 560:560 + cw] = cv2.resize(cat, (cw, ch))
        boxes = det._detect_boxes(bg, floor=0.3)
        if any(label == "cat" for label, _, _ in boxes):
            hits += 1
    assert hits >= 3, f"distant cats not detected at 512px: {hits}/{len(CATS)}"


@pytest.mark.parametrize("path", PEOPLE)
def test_each_person_reports_motion_person_outcome(path):
    """detect_in_frame returns a real boolean True on these clear photos."""
    det = _detector()
    assert det.detect_in_frame(cv2.imread(path)) is True


def test_boxes_and_annotated_snapshot():
    """A person image yields a person box and an annotated JPEG we can decode."""
    import numpy as np

    det = _detector()
    img = cv2.imread(PEOPLE[0])
    boxes = det._detect_boxes(img, floor=0.3)
    assert any(label == "person" for label, _, _ in boxes)

    # Prime the detector's "last frame" state and render the annotated JPEG.
    det._last_frame = img
    det._last_boxes = boxes
    jpeg = det.annotated_jpeg()
    assert jpeg and jpeg[:2] == b"\xff\xd8"          # JPEG magic
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[0] > 0
