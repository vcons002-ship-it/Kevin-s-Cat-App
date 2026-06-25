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
    # First frame reports NO motion (nothing to compare yet); an identical frame
    # reports none; a large solid change reports motion.
    mp = detector.MotionPrefilter(min_area_frac=0.01)
    blank = np.zeros((200, 200), dtype=np.uint8)
    assert mp.update(blank) is False           # first frame: no baseline
    assert mp.update(blank.copy()) is False    # no change
    moved = blank.copy()
    moved[0:120, 0:120] = 255                   # 36% of pixels change
    assert mp.update(moved) is True


def test_motion_prefilter_ignores_sensor_noise():
    # Random per-pixel grain (like night-vision noise / compression) must NOT
    # register as motion after the median blur — this is the false-trigger fix.
    rng = np.random.default_rng(0)
    mp = detector.MotionPrefilter()
    base = np.full((300, 300), 120, dtype=np.uint8)
    mp.update(base)                                    # prime baseline
    noisy = np.clip(base.astype(int) + rng.integers(-30, 31, base.shape), 0, 255)
    assert mp.update(noisy.astype(np.uint8)) is False


def test_motion_prefilter_ignores_decode_artifact_lines():
    # A corrupt camera frame shows a long, thin band of bad pixels: lots of
    # changed pixels but only a few rows tall. It must NOT trigger motion.
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)                                    # prime baseline
    for thickness in (2, 4, 8):                        # thin to moderately thick
        artifact = base.copy()
        artifact[180:180 + thickness, :] = 255         # full-width bright band
        assert mp.update(artifact) is False, f"{thickness}px band triggered motion"
        mp._prev = None                                # reset baseline for next case
        mp.update(base)


def test_motion_prefilter_triggers_on_solid_blob():
    # A compact moving object (a person/cat-sized blob) must still trigger.
    mp = detector.MotionPrefilter()
    base = np.full((360, 640), 110, dtype=np.uint8)
    mp.update(base)
    moved = base.copy()
    moved[120:300, 250:360] = 255                      # ~110x180 solid blob
    assert mp.update(moved) is True
