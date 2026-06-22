"""Person detection: cheap motion pre-filter + COCO MobileNet-SSD via cv2.dnn.

The goal is to trigger **only when a person enters frame and to ignore the
cats**. We do this in two stages to keep CPU low on a GPU-less NAS:

1. A frame-difference *motion pre-filter* — skip the neural net entirely while
   the scene is static.
2. When motion is seen, run MobileNet-SSD (COCO). COCO has ``person`` and
   ``cat`` as separate classes, so we report a trigger only for a ``person``
   box above the confidence threshold and ignore ``cat`` (and everything else).

The detection-parsing core (:func:`person_in_detections`) is a pure function
over a raw network output array, so it is unit-testable without a camera, a
model, or OpenCV's inference.
"""

from __future__ import annotations

import os
import time

# The 21 classes of the standard MobileNet-SSD (VOC-style) model shipped in
# d20app/models/. Index 15 is "person"; index 8 is "cat".
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]
PERSON_CLASS_ID = CLASSES.index("person")

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROTOTXT = os.path.join(_MODELS_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(_MODELS_DIR, "mobilenet_ssd.caffemodel")


def person_in_detections(detections, confidence: float) -> bool:
    """Return True if any ``person`` box clears ``confidence``.

    ``detections`` is the raw MobileNet-SSD output shaped ``(1, 1, N, 7)`` where
    each row is ``[image_id, class_id, score, x1, y1, x2, y2]``. This is pure
    array logic — no OpenCV calls — so it can be tested with a plain nested list
    or a numpy array.
    """
    for i in range(detections.shape[2]):
        score = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if class_id == PERSON_CLASS_ID and score >= confidence:
            return True
    return False


class MotionPrefilter:
    """Detect motion by comparing consecutive grayscale frames.

    Returns True when the fraction of changed pixels exceeds ``min_area_frac``.
    The first frame always reports "motion" so detection can run immediately.
    """

    def __init__(self, min_area_frac: float = 0.003, diff_threshold: int = 25) -> None:
        self.min_area_frac = min_area_frac
        self.diff_threshold = diff_threshold
        self._prev = None

    def update(self, gray) -> bool:
        import cv2  # local import: keep module importable without OpenCV

        if self._prev is None:
            self._prev = gray
            return True
        delta = cv2.absdiff(self._prev, gray)
        self._prev = gray
        _, thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
        changed = int((thresh > 0).sum())
        total = thresh.shape[0] * thresh.shape[1]
        return total > 0 and (changed / total) >= self.min_area_frac


class PersonDetector:
    """Open a camera stream and report when a person (not a cat) is present.

    Combines :class:`MotionPrefilter` with the MobileNet-SSD network. Handles
    stream reconnection with backoff so a flaky camera doesn't kill the loop.
    """

    def __init__(self, source: str, confidence: float = 0.5, roi=None) -> None:
        self.source = source
        self.confidence = confidence
        self.roi = roi          # optional [x, y, w, h]
        self._net = None
        self._cap = None
        self._motion = MotionPrefilter()

    # -- model / stream lifecycle -------------------------------------------
    def _ensure_net(self):
        import cv2

        if self._net is None:
            if not (os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL)):
                raise FileNotFoundError(
                    "MobileNet-SSD model files are missing from d20app/models/. "
                    "See d20app/models/README.md for how to fetch them."
                )
            self._net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        return self._net

    def _ensure_cap(self):
        import cv2

        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.source)
        return self._cap

    def _crop(self, frame):
        if not self.roi:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    # -- inference -----------------------------------------------------------
    def detect_in_frame(self, frame) -> bool:
        """Run the net on a single BGR frame and return True if a person shows."""
        import cv2

        frame = self._crop(frame)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,        # 1/127.5
            size=(300, 300),
            mean=127.5,
        )
        net = self._ensure_net()
        net.setInput(blob)
        detections = net.forward()
        return person_in_detections(detections, self.confidence)

    def read_and_detect(self) -> bool:
        """Grab one frame, apply the motion pre-filter, then detect.

        Returns True only when a person is detected. Returns False on read
        failure (caller may back off and retry).
        """
        import cv2

        cap = self._ensure_cap()
        ok, frame = cap.read()
        if not ok or frame is None:
            self._cap = None        # force reconnect next call
            return False
        gray = cv2.cvtColor(self._crop(frame), cv2.COLOR_BGR2GRAY)
        if not self._motion.update(gray):
            return False
        return self.detect_in_frame(frame)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
