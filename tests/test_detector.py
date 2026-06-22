"""Unit tests for the detection-parsing core (no camera, no model inference).

We exercise ``person_in_detections`` directly with synthetic MobileNet-SSD
output so we can prove the key requirement: trigger on a person, ignore a cat.
"""

import numpy as np

from d20app import detector


def _detections(rows):
    """Build a (1, 1, N, 7) array of [image_id, class_id, score, x1,y1,x2,y2]."""
    arr = np.zeros((1, 1, len(rows), 7), dtype=np.float32)
    for i, (class_id, score) in enumerate(rows):
        arr[0, 0, i] = [0, class_id, score, 0.1, 0.1, 0.9, 0.9]
    return arr


PERSON = detector.CLASSES.index("person")   # 15
CAT = detector.CLASSES.index("cat")         # 8


def test_triggers_on_confident_person():
    det = _detections([(PERSON, 0.92)])
    assert detector.person_in_detections(det, confidence=0.5) is True


def test_ignores_cat_even_when_very_confident():
    det = _detections([(CAT, 0.99)])
    assert detector.person_in_detections(det, confidence=0.5) is False


def test_ignores_low_confidence_person():
    det = _detections([(PERSON, 0.30)])
    assert detector.person_in_detections(det, confidence=0.5) is False


def test_person_among_cats_still_triggers():
    det = _detections([(CAT, 0.97), (CAT, 0.88), (PERSON, 0.71)])
    assert detector.person_in_detections(det, confidence=0.5) is True


def test_empty_detections_do_not_trigger():
    det = _detections([])
    assert detector.person_in_detections(det, confidence=0.5) is False


def test_confidence_threshold_boundary():
    det = _detections([(PERSON, 0.5)])
    assert detector.person_in_detections(det, confidence=0.5) is True   # >= is inclusive


def test_motion_prefilter_first_frame_and_change():
    # First frame always reports motion; an identical frame reports none; a
    # very different frame reports motion. (Uses real numpy arrays, no OpenCV
    # dependency beyond absdiff/threshold which cv2 provides at runtime.)
    cv2 = __import__("cv2")
    mp = detector.MotionPrefilter(min_area_frac=0.01)
    blank = np.zeros((100, 100), dtype=np.uint8)
    assert mp.update(blank) is True            # first frame
    assert mp.update(blank.copy()) is False    # no change
    moved = blank.copy()
    moved[0:60, 0:60] = 255                     # 36% of pixels change
    assert mp.update(moved) is True
