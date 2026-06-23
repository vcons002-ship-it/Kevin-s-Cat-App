"""Regression test: the bundled model must detect people and ignore cats.

This is the guard that would have caught the shipped-broken model (a training
snapshot whose weights didn't match the prototxt, so detection scored 0 on
everything). It runs the real PersonDetector over a few bundled real photos.

Fixtures: tests/fixtures/people/ (PennFudanPed pedestrians, downscaled) and
tests/fixtures/cats/ (ImageNet domestic cats, downscaled).
"""

import glob
import os

import cv2
import pytest

from d20app.detector import PersonDetector

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PEOPLE = sorted(glob.glob(os.path.join(FIXTURES, "people", "*.jpg")))
CATS = sorted(glob.glob(os.path.join(FIXTURES, "cats", "*.jpg")))


def _detector():
    return PersonDetector(source="unused", confidence=0.5)


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


@pytest.mark.parametrize("path", PEOPLE)
def test_each_person_reports_motion_person_outcome(path):
    """detect_in_frame returns a real boolean True on these clear photos."""
    det = _detector()
    assert det.detect_in_frame(cv2.imread(path)) is True
