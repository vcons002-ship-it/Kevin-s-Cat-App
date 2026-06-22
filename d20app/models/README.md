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

## Re-fetching the weights

If the `.caffemodel` is ever missing (e.g. a shallow clone that skipped large
files), download the original VOC-trained weights and the matching prototxt:

```
curl -L -o mobilenet_ssd.caffemodel \
  https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel
curl -L -o deploy.prototxt \
  https://raw.githubusercontent.com/PINTO0309/MobileNet-SSD-RealSense/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt
```
