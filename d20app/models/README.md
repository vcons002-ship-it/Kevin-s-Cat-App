# Person-detection model (MobileNet-SSD, COCO/VOC 21-class)

These files are **bundled in the repo** so there's nothing to download — the app
works straight after `setup.sh`.

- `deploy.prototxt` — network definition (21 classes).
- `mobilenet_ssd.caffemodel` — trained weights (~23 MB).

The model classifies `person` and `cat` as **separate** classes, which is how
the app triggers on people while ignoring the cats. The class list lives in
`d20app/detector.py` (`CLASSES`); `person` is index 15, `cat` is index 8.

It runs through OpenCV's built-in `cv2.dnn` module — no PyTorch, no TensorFlow,
no GPU, and no separate AI service.

Measured accuracy (MobileNetSSD_deploy weights): **99.4%** of 170 PennFudanPed
pedestrian images detected at confidence 0.5, with **0** false-positive people
across a set of cat images.

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
