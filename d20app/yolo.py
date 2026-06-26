"""YOLO11 object detection via OpenCV's ``cv2.dnn`` (ONNX) — no PyTorch at runtime.

A drop-in alternative backend for :class:`d20app.detector.PersonDetector`: it
produces the same ``[(label, score, (x1, y1, x2, y2))]`` boxes the MobileNet-SSD
path does, so the rest of the pipeline (person trigger, ``cat`` labelling,
annotated snapshots) is unchanged.

Why offer it: the bundled MobileNet-SSD is fast but weak in low light / odd poses
(it scored 0.00 on a real dim night frame). YOLO11n scored ~0.87 on the same
frame at ~1.4x the CPU — a much better night/occlusion detector for a modest cost.

Two variants are available (see :data:`MODELS`):

- ``yolo11n`` — nano at 320x320 (~11 MB, ~28 ms/frame). The default; it already
  handles the night case well.
- ``yolo11m`` — medium at 640x640 (~77 MB, ~500 ms/frame on CPU). Bigger and
  slower; on our own night/day frames it did **not** beat nano on the night case
  that motivated it, so it's offered as an option, not the default. Worth trying
  if you have CPU headroom and want the extra capacity on hard scenes.

Each model is exported from its Ultralytics ``*.pt`` to a fixed-size ONNX (see
d20app/models/README.md). Raw output is ``(1, 84, N)``: 4 box coords (cx, cy, w,
h in letterboxed pixels) + 80 COCO class scores per anchor.
"""

from __future__ import annotations

import os

# COCO-80 class names in model order. Index 0 is "person", 15 is "cat" — the two
# the app cares about; the rest are named in the activity log like any mover.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Known YOLO variants → their exported ONNX file and the fixed input size that
# file was exported at (must match, or the net errors / decodes garbage).
MODELS = {
    "yolo11n": {"file": "yolo11n.onnx", "size": 320},
    "yolo11m": {"file": "yolo11m.onnx", "size": 640},
}
DEFAULT_VARIANT = "yolo11n"

# Back-compat aliases for the single-model era (some tests/callers import these).
ONNX_PATH = os.path.join(_MODELS_DIR, MODELS[DEFAULT_VARIANT]["file"])
INPUT_SIZE = MODELS[DEFAULT_VARIANT]["size"]
_NMS_IOU = 0.45


def model_path(variant: str = DEFAULT_VARIANT) -> str:
    """Absolute path to a variant's ONNX file (no existence check)."""
    return os.path.join(_MODELS_DIR, MODELS[variant]["file"])


def input_size(variant: str = DEFAULT_VARIANT) -> int:
    """The fixed square input size the given variant was exported at."""
    return MODELS[variant]["size"]


def load_net(variant: str = DEFAULT_VARIANT):
    """Load a YOLO11 variant's ONNX with cv2.dnn (raises if missing/unknown)."""
    import cv2

    if variant not in MODELS:
        raise ValueError(f"unknown YOLO variant {variant!r}; known: {sorted(MODELS)}")
    path = model_path(variant)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YOLO model {os.path.relpath(path)} is missing. "
            "See d20app/models/README.md to export it, or use the MobileNet-SSD model."
        )
    return cv2.dnn.readNetFromONNX(path)


def _letterbox(frame, size: int):
    """Resize ``frame`` into a ``size``x``size`` square, padding to keep aspect.

    Returns ``(canvas, ratio, pad_x, pad_y)`` so boxes can be mapped back to the
    original frame's pixel coordinates.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def detect_boxes(net, frame, floor: float, size: int = INPUT_SIZE) -> list:
    """Run YOLO11n on one BGR ``frame``; return ``[(label, score, (x1,y1,x2,y2))]``.

    ``floor`` is the confidence below which detections are dropped. Coordinates
    are pixels within ``frame``.
    """
    import cv2
    import numpy as np

    lb, r, pad_x, pad_y = _letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()                       # (1, 84, N)
    out = np.squeeze(out, 0).T                 # (N, 84): [cx, cy, w, h, 80 class scores]
    class_scores = out[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)
    keep = confs >= floor
    out, class_ids, confs = out[keep], class_ids[keep], confs[keep]
    if len(out) == 0:
        return []

    rects, scores, ids = [], [], []
    for row, cid, cf in zip(out, class_ids, confs):
        cx, cy, bw, bh = row[:4]
        x1 = (cx - bw / 2 - pad_x) / r
        y1 = (cy - bh / 2 - pad_y) / r
        rects.append([int(x1), int(y1), int(bw / r), int(bh / r)])
        scores.append(float(cf))
        ids.append(int(cid))

    boxes = []
    idxs = cv2.dnn.NMSBoxes(rects, scores, float(floor), _NMS_IOU)
    for i in np.array(idxs).reshape(-1):
        x, y, w, h = rects[i]
        cid = ids[i]
        label = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
        boxes.append((label, scores[i], (x, y, x + w, y + h)))
    return boxes
