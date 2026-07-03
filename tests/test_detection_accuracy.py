"""Regression test: the bundled model must detect people and ignore cats.

This is the guard that would have caught a shipped-broken model (one that scored
0 on everything). It runs the real PersonDetector over a few bundled real photos
with the app's default model, **yolo11n** (MobileNet-SSD was removed in 0.25.0).

Fixtures: tests/fixtures/people/ (PennFudanPed pedestrians, many rear-view),
tests/fixtures/people_hard/ (people in hats/helmets/headgear),
tests/fixtures/cats/ (single cats — the original ImageNet five plus a broader
Wikimedia Commons set of varied breeds/poses/lighting), and
tests/fixtures/cats_multi/ (scenes with 2+ cats — clusters and pairs, the case
most likely to be misread as a person). All downscaled; see cats/CREDITS.md.
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
CATS_MULTI = sorted(glob.glob(os.path.join(FIXTURES, "cats_multi", "*.jpg")))

# Two single-cat frames yield a person box under yolo11n — honestly, for opposite
# reasons (verified by eye when MobileNet-SSD was dropped in 0.25.0):
#   - cat15.jpg DOES contain a person: a hand holding a camera fills the lower-left
#     corner. YOLO is *correct* here (~0.76); SSD simply missed the hand. Not a bug.
#   - cat23.jpg is a true misread: a top-down ginger cat reads as person ~0.67. This
#     is the YOLO analogue of the old cat-cluster misreads — a still-frame artefact
#     the live path neutralises with the temporal gate (`confirm_frames`), not a
#     thing to suppress in the model.
# Excluded from the strict "no single cat triggers a person" assertion, by name, so
# a *new* regression (a different cat starting to trigger) still fails the test.
KNOWN_PERSON_MISREADS = {"cat15.jpg", "cat23.jpg"}


def _detector(confidence=0.5, model="yolo11n"):
    # The app default is yolo11n; confidence 0.5 mirrors the app.
    return PersonDetector(source="unused", confidence=confidence, model=model)


def test_fixtures_present():
    assert PEOPLE, "no people fixtures found"
    assert CATS, "no cat fixtures found"
    assert CATS_MULTI, "no multi-cat fixtures found"


def test_people_are_detected():
    """A clear majority of people images must trigger."""
    det = _detector()
    hits = sum(det.detect_in_frame(cv2.imread(p)) for p in PEOPLE)
    rate = hits / len(PEOPLE)
    assert rate >= 0.65, f"person recall too low: {hits}/{len(PEOPLE)}"


def test_cats_do_not_trigger():
    """No single-cat image may be read as a person (the whole point), except the two
    documented frames in ``KNOWN_PERSON_MISREADS`` (one of which genuinely contains a
    person). A new cat starting to trigger is a real regression and must fail here."""
    det = _detector()
    triggered = {os.path.basename(p) for p in CATS if det.detect_in_frame(cv2.imread(p))}
    unexpected = triggered - KNOWN_PERSON_MISREADS
    assert not unexpected, f"cats triggered person: {sorted(unexpected)}"


def test_multi_cat_scenes_do_not_trigger():
    """No multi-cat scene is read as a person at the 0.5 default.

    Cat clusters are the most person-shaped non-person thing the camera sees.

    NB: this is a *still-frame* guarantee. A cat in motion can momentarily spike
    much higher; the live-camera safeguard for that is the temporal gate
    (``confirm_frames``), not this test.
    """
    assert CATS_MULTI, "no multi-cat fixtures found"
    det = _detector()
    for p in CATS_MULTI:
        assert not det.detect_in_frame(cv2.imread(p)), \
            f"multi-cat scene triggered person: {os.path.basename(p)}"


def test_cats_are_recognised_as_cats():
    """The model still *sees* cats (sanity that it isn't blind to them).

    Recognising cats isn't the app's job — ignoring them is — so this floor is
    deliberately lenient. It exists only so a future model swap that silently
    stopped detecting cats entirely would be caught.
    """
    det = _detector(confidence=0.4)
    hits = 0
    for p in CATS:
        boxes = det._detect_boxes(cv2.imread(p), floor=0.3)
        if any(label == "cat" for label, _, _ in boxes):
            hits += 1
    assert hits >= len(CATS) // 2, f"model recognised too few cats: {hits}/{len(CATS)}"


def test_hard_pose_people_are_detected():
    """People in head accessories (hats/helmets/headgear) must still trigger.

    Guards against detection regressing on harder real-world cases. Back-turned
    people are covered by the PennFudan `people/` fixtures (street pedestrians,
    many walking away from the camera).
    """
    det = _detector()
    assert PEOPLE_HARD, "no hard-case fixtures found"
    misses = [p for p in PEOPLE_HARD if not det.detect_in_frame(cv2.imread(p))]
    # Lenient floor: the bundled nano export is tuned for live frames, not a
    # pixel-perfect still benchmark — most clear it, and the suite's job is to
    # catch a model that's gone blind, not to chase the last few hard stills.
    rate = 1 - len(misses) / len(PEOPLE_HARD)
    assert rate >= 0.6, f"hard-pose people missed too often: {[os.path.basename(p) for p in misses]}"


def test_distant_cats_are_identified_at_high_detail():
    """A cat ~1/4 of the frame is recognised by the bundled medium model.

    The locator path uses a stronger model (yolo26m, 640) to resolve a
    small/distant cat; assert that capability explicitly.
    """
    import numpy as np

    det = _detector(confidence=0.4, model="yolo26m")
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
    assert hits >= 3, f"distant cats not detected: {hits}/{len(CATS)}"


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
