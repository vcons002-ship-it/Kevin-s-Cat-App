# Person-detection models

Two detectors are **bundled in the repo** so there's nothing to download — the
app works straight after `setup.sh`. Both run through OpenCV's `cv2.dnn` on the
CPU — no PyTorch/TensorFlow at runtime, no GPU, no cloud.

The active one is chosen by `detector_model` in `config.yaml` / the GUI:

- **`yolo11n`** (default) — `yolo11n.onnx` (~10 MB), Ultralytics YOLO11-nano,
  COCO-80. Much better in low light / odd poses (scored ~0.87 on a real dim night
  frame where MobileNet scored 0.00) for ~1.4× the CPU (~28 ms vs ~20 ms at the
  bundled sizes). Class names live in `d20app/yolo.py` (`COCO_CLASSES`); `person`
  is index 0, `cat` is 15.
- **`mobilenet_ssd`** — the lightest option, and the automatic fallback if the
  YOLO model can't be loaded.

## MobileNet-SSD (COCO/VOC 21-class)

- `deploy.prototxt` — network definition (21 classes).
- `mobilenet_ssd.caffemodel` — trained weights (~23 MB).

The model classifies `person` and `cat` as **separate** classes, which is how
the app triggers on people while ignoring the cats. The class list lives in
`d20app/detector.py` (`CLASSES`); `person` is index 15, `cat` is index 8.

It runs through OpenCV's built-in `cv2.dnn` module — no PyTorch, no TensorFlow,
no GPU, and no separate AI service.

Measured accuracy (MobileNetSSD_deploy weights): **99.4%** of 170 PennFudanPed
pedestrian images detected at confidence 0.5. Across a 45-image cat set (35
single cats + 10 multi-cat scenes, tested at 300px and 512px) no single cat is
read as a person; a couple of dense cat *clusters* do produce a weak `person`
box. We accept those rather than suppress them, because a weak person box over a
cat is also what a person *carrying* a cat looks like — see the multi-cat test's
`KNOWN_CLUSTER_MISREADS` in `tests/test_detection_accuracy.py`.

## Re-fetching the weights

If the `.caffemodel` is ever missing (e.g. a shallow clone that skipped large
files), download the **deploy** weights **and** the **matching** deploy prototxt
from the same source so the layer names line up:

```
curl -L -o mobilenet_ssd.caffemodel \
  https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel
curl -L -o deploy.prototxt \
  https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.prototxt
```

> ⚠️ Use the **deploy** caffemodel, not a training snapshot like
> `mobilenet_iter_73000.caffemodel`. A training snapshot's BatchNorm layers
> don't match a deploy prototxt, so `cv2.dnn` loads it without error but every
> detection scores 0 — the model silently detects nothing. The regression test
> `tests/test_detection_accuracy.py` guards against shipping such a mismatch.

## Re-exporting `yolo11n.onnx`

The bundled ONNX is a fixed **320×320** export of Ultralytics `yolo11n.pt`.
Ultralytics + PyTorch are needed **only to export** (one-off, offline) — they are
*not* runtime dependencies; the app runs the ONNX via `cv2.dnn` alone.

```
pip install ultralytics            # pulls torch; do this in a throwaway venv
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=320, opset=12, simplify=True)"
mv yolo11n.onnx d20app/models/yolo11n.onnx
```

The input size is fixed at export time and must stay **320** unless you also
change `INPUT_SIZE` in `d20app/yolo.py` (OpenCV's importer needs a static shape).
