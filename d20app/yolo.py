"""YOLO11n object detection via OpenCV's ``cv2.dnn`` (ONNX) — no PyTorch at runtime.

A drop-in alternative backend for :class:`d20app.detector.PersonDetector`: it
produces the same ``[(label, score, (x1, y1, x2, y2))]`` boxes the MobileNet-SSD
path does, so the rest of the pipeline (person trigger, ``cat`` labelling,
annotated snapshots) is unchanged.

Why offer it: the bundled MobileNet-SSD is fast but weak in low light / odd poses
(it scored 0.00 on a real dim night frame). YOLO11n scored ~0.87 on the same
frame at ~1.4x the CPU — a much better night/occlusion detector for a modest cost.

The model is exported from Ultralytics ``yolo11n.pt`` to a fixed 320x320 ONNX
(see d20app/models/README.md). Its raw output is ``(1, 84, N)``: 4 box coords
(cx, cy, w, h in letterboxed pixels) + 80 COCO class scores per anchor.
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
ONNX_PATH = os.path.join(_MODELS_DIR, "yolo11n.onnx")
INPUT_SIZE = 320          # must match the exported model's fixed input
_NMS_IOU = 0.45


def load_net():
    """Load the YOLO11n ONNX with cv2.dnn (raises if the file is missing)."""
    import cv2

    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            "YOLO model d20app/models/yolo11n.onnx is missing. "
            "See d20app/models/README.md to export it, or use the MobileNet-SSD model."
        )
    return cv2.dnn.readNetFromONNX(ONNX_PATH)


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
