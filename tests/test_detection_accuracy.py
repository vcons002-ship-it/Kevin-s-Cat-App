"""Regression test: the bundled model must detect people and ignore cats.

This is the guard that would have caught the shipped-broken model (a training
snapshot whose weights didn't match the prototxt, so detection scored 0 on
everything). It runs the real PersonDetector over a few bundled real photos.

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


def _detector():
    # Mirror the app defaults (detect_size 300, confidence 0.4).
    return PersonDetector(source="unused", confidence=0.4)


def test_fixtures_present():
    assert PEOPLE, "no people fixtures found"
    assert CATS, "no cat fixtures found"
    assert CATS_MULTI, "no multi-cat fixtures found"


def test_people_are_detected():
    """At least 65% of people images must trigger (we measure ~99% on 170)."""
    det = _detector()
    hits = sum(det.detect_in_frame(cv2.imread(p)) for p in PEOPLE)
    rate = hits / len(PEOPLE)
    assert rate >= 0.65, f"person recall too low: {hits}/{len(PEOPLE)}"


def test_cats_do_not_trigger():
    """No single-cat image may be detected as a person (the whole point)."""
    det = _detector()
    for p in CATS:
        assert not det.detect_in_frame(cv2.imread(p)), f"cat triggered person: {p}"


# Dense cat scenes the detector tolerates as occasional false "person" triggers.
# A low-confidence person box sitting on top of a pile of cats is exactly what a
# person *carrying* a cat also looks like, so we deliberately do NOT suppress it:
# a missed real person breaks the whole feature, while a stray treat-roll on a
# clump of cats is harmless. This set is pinned so the misread rate can't grow
# unnoticed as the model or fixtures change. (multi03: several cats around one
# bowl, seen top-down; multi07: two entangled cats — both only at one detail size.)
KNOWN_CLUSTER_MISREADS = {"multi03.jpg", "multi07.jpg"}


def test_multi_cat_scenes_rarely_trigger():
    """Most multi-cat scenes are ignored; only the pinned cluster cases may fire.

    Checked at both the 300px default and the selectable 512px detail. The point
    is to catch *new* false triggers creeping in, not to force zero — see
    ``KNOWN_CLUSTER_MISREADS`` for why a couple are accepted.
    """
    assert CATS_MULTI, "no multi-cat fixtures found"
    triggered = set()
    for size in (300, 512):
        det = PersonDetector(source="unused", confidence=0.4, detect_size=size)
        for p in CATS_MULTI:
            if det.detect_in_frame(cv2.imread(p)):
                triggered.add(os.path.basename(p))
    new = triggered - KNOWN_CLUSTER_MISREADS
    assert not new, f"new multi-cat scenes misread as a person: {sorted(new)}"


def test_cats_are_recognised_as_cats():
    """The model still *sees* cats (sanity that it isn't blind to them).

    Recognising cats isn't the app's job — ignoring them is — so this floor is
    deliberately lenient. It exists only so a future model swap that silently
    stopped detecting cats entirely would be caught. Measured ~66% at 512px on
    the single-cat set; we require at least half.
    """
    det = PersonDetector(source="unused", confidence=0.4, detect_size=512)
    hits = 0
    for p in CATS:
        boxes = det._detect_boxes(cv2.imread(p), floor=0.3)
        if any(label == "cat" for label, _, _ in boxes):
            hits += 1
    assert hits >= len(CATS) // 2, f"model recognised too few cats: {hits}/{len(CATS)}"


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
