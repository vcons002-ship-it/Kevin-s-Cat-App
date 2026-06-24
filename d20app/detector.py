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
from dataclasses import dataclass, field

# Make OpenCV's FFmpeg backend behave like VLC/ffplay for RTSP cameras:
#   * rtsp_transport;tcp — use TCP (retransmitted, in-order packets) instead of
#     the lossy UDP default, which eliminates most "error while decoding MB…"
#     and "missing reference picture" decoder spam.
#   * timeout;5000000 — fail a dead/unreachable camera after 5s (microseconds)
#     instead of blocking the loop on the OS default of a minute or more.
#   * a quiet log level — stop libavcodec printing cosmetic decode errors and
#     repeated "401 Unauthorized" lines straight to the console; the app reports
#     real failures in its own Activity log instead.
# All use setdefault() so an advanced user can override them from the shell,
# e.g. OPENCV_FFMPEG_LOGLEVEL=24 to see warnings again while debugging.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000"
)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")   # 8 = AV_LOG_FATAL

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

# Confidence floor for *naming* a non-person mover (e.g. "cat") in the log.
# Lower than the person threshold so distant/uncertain cats still get identified.
_LABEL_FLOOR = 0.3


class CameraError(RuntimeError):
    """The camera stream could not be opened or read (bad URL, auth, network)."""


@dataclass
class FrameOutcome:
    """What one processed frame contained."""

    motion: bool             # did the cheap motion pre-filter trigger?
    person: bool             # was a person detected above the threshold?
    labels: tuple = ()       # other classes seen (e.g. ("cat",)), best score first


_cv2_quieted = False


def _quiet_cv2_logs(cv2) -> None:
    """Hush OpenCV's own WARN chatter (e.g. the videoio backend warning) once.

    This is separate from OPENCV_FFMPEG_LOGLEVEL, which only governs FFmpeg.
    """
    global _cv2_quieted
    if _cv2_quieted:
        return
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    _cv2_quieted = True


def grab_frame_jpeg(source: str, skip: int = 4):
    """Open ``source``, grab one frame, and return it as JPEG bytes (or None).

    Used by the GUI's region-of-interest picker to show a still from the camera.
    A few frames are skipped so the returned image isn't the codec's first
    (often grey/partial) frame.
    """
    import cv2

    _quiet_cv2_logs(cv2)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for _ in range(skip + 1):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def mask_credentials(url: str) -> str:
    """Hide the password in an ``rtsp://user:pass@host`` URL for safe logging."""
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)        # host never contains '@'
    if ":" in creds:
        user, _ = creds.split(":", 1)
        creds = f"{user}:***"
    return f"{scheme}://{creds}@{host}"


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
    Frames are Gaussian-blurred first so sensor noise, compression grain and a
    ticking timestamp overlay don't register as "motion" (a common cause of
    false triggers on a scene where nothing actually moved). The very first
    frame reports **no** motion — there's nothing to compare it against yet, so
    a static scene never triggers detection until something really moves.
    """

    def __init__(self, min_area_frac: float = 0.003, diff_threshold: int = 25) -> None:
        self.min_area_frac = min_area_frac
        self.diff_threshold = diff_threshold
        self._prev = None

    def update(self, gray) -> bool:
        import cv2  # local import: keep module importable without OpenCV

        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._prev is None:
            self._prev = blurred
            return False
        delta = cv2.absdiff(self._prev, blurred)
        self._prev = blurred
        _, thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)
        changed = int((thresh > 0).sum())
        total = thresh.shape[0] * thresh.shape[1]
        return total > 0 and (changed / total) >= self.min_area_frac


class PersonDetector:
    """Open a camera stream and report when a person (not a cat) is present.

    Combines :class:`MotionPrefilter` with the MobileNet-SSD network. Handles
    stream reconnection with backoff so a flaky camera doesn't kill the loop.
    """

    def __init__(self, source: str, confidence: float = 0.5, roi=None,
                 detect_size: int = 512) -> None:
        self.source = source
        self.confidence = confidence
        self.roi = roi          # optional [x, y, w, h]
        # Net input resolution. 512 (vs the classic 300) recovers small/distant
        # subjects — e.g. a cat across the room — at a modest CPU cost.
        self.detect_size = int(detect_size) if detect_size else 512
        self._net = None
        self._cap = None
        self._read_fails = 0
        self.frame_size = None  # (w, h) of the last good frame; None until one reads
        self._last_frame = None    # last frame the net analysed (for snapshots)
        self._last_boxes = []      # detections on that frame: [(label, score, box)]
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

        _quiet_cv2_logs(cv2)
        if self._cap is None or not self._cap.isOpened():
            # Force the FFmpeg backend explicitly: some OpenCV builds otherwise
            # pick a backend that mishandles RTSP authentication, so a stream
            # that works in VLC fails here with "401 Unauthorized". FFmpeg does
            # the same Basic/Digest auth VLC does.
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                raise CameraError(
                    f"could not open the camera stream "
                    f"{mask_credentials(self.source)} — check the URL, and the "
                    "username/password if the camera needs a login"
                )
            self._cap = cap
        return self._cap

    def _crop(self, frame):
        if not self.roi:
            return frame
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    # -- inference -----------------------------------------------------------
    def _detect_boxes(self, frame, floor: float) -> list:
        """Return ``[(label, score, (x1, y1, x2, y2))]`` for one BGR frame.

        Coordinates are pixels within the (ROI-cropped) frame the net analysed.
        """
        import cv2

        cropped = self._crop(frame)
        h, w = cropped.shape[:2]
        s = self.detect_size
        blob = cv2.dnn.blobFromImage(
            cv2.resize(cropped, (s, s)),
            scalefactor=0.007843,        # 1/127.5
            size=(s, s),
            mean=127.5,
        )
        net = self._ensure_net()
        net.setInput(blob)
        det = net.forward()
        boxes = []
        for i in range(det.shape[2]):
            score = float(det[0, 0, i, 2])
            cid = int(det[0, 0, i, 1])
            if score >= floor and 0 <= cid < len(CLASSES):
                x1 = int(det[0, 0, i, 3] * w)
                y1 = int(det[0, 0, i, 4] * h)
                x2 = int(det[0, 0, i, 5] * w)
                y2 = int(det[0, 0, i, 6] * h)
                boxes.append((CLASSES[cid], score, (x1, y1, x2, y2)))
        return boxes

    @staticmethod
    def _best(boxes, label: str) -> float:
        return max((s for lab, s, _ in boxes if lab == label), default=0.0)

    def detect_in_frame(self, frame) -> bool:
        """Return True if a person is present in ``frame`` above the threshold."""
        boxes = self._detect_boxes(frame, floor=min(0.3, self.confidence))
        return self._best(boxes, "person") >= self.confidence

    # Colours (BGR) for drawing boxes: person = green, cat = orange, other = grey.
    _BOX_COLORS = {"person": (80, 220, 80), "cat": (40, 170, 240)}

    def annotated_jpeg(self) -> bytes | None:
        """JPEG of the last analysed frame with labelled detection boxes drawn."""
        import cv2

        if self._last_frame is None:
            return None
        img = self._last_frame.copy()
        for label, score, (x1, y1, x2, y2) in self._last_boxes:
            color = self._BOX_COLORS.get(label, (160, 160, 160))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {score:.2f}"
            ty = max(y1 - 6, 12)
            cv2.putText(img, tag, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def read_and_detect(self) -> FrameOutcome:
        """Grab one frame, apply the motion pre-filter, then classify it.

        Returns a :class:`FrameOutcome`. ``motion`` is False when nothing moved
        (or a frame couldn't be read); when motion is seen, ``person`` says
        whether a person cleared the threshold and ``labels`` lists the other
        things seen (e.g. ``("cat",)``) so the caller can report *what* moved.
        Raises :class:`CameraError` when the stream is really gone.
        """
        import cv2

        cap = self._ensure_cap()        # raises CameraError if it can't open
        ok, frame = cap.read()
        if not ok or frame is None:
            self._read_fails += 1
            self._cap = None            # force a reconnect next call
            # Tolerate a brief hiccup, but a run of empty reads means the
            # stream is really gone — surface it so the loop can back off.
            if self._read_fails >= 3:
                raise CameraError(
                    f"lost the camera stream {mask_credentials(self.source)} "
                    "(no frames received)"
                )
            return FrameOutcome(motion=False, person=False)
        self._read_fails = 0
        self.frame_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(self._crop(frame), cv2.COLOR_BGR2GRAY)
        if not self._motion.update(gray):
            return FrameOutcome(motion=False, person=False)

        boxes = self._detect_boxes(frame, floor=min(0.3, self.confidence))
        self._last_frame = self._crop(frame)   # what the net saw (box coords match)
        self._last_boxes = boxes
        person = self._best(boxes, "person") >= self.confidence
        # Identify non-person movers (cats!) at a lower bar than a person needs,
        # so a smaller/less-certain cat is still named rather than "something".
        labels = tuple(
            label
            for label, score, _ in sorted(boxes, key=lambda b: -b[1])
            if label != "person" and score >= _LABEL_FLOOR
        )
        return FrameOutcome(motion=True, person=person, labels=labels)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
