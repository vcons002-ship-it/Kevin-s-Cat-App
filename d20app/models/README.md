# Person-detection models

The detector is **YOLO** (Ultralytics, COCO-80), run through OpenCV's `cv2.dnn` —
no PyTorch/TensorFlow at runtime, no cloud. `yolo11n`, `yolo11m` and `yolo26m` are
**bundled in the repo** so there's nothing to download — the app works straight
after `setup.sh`.

> **MobileNet-SSD was removed in 0.25.0** (#57). It lost every benchmark to YOLO
> (notably it scored 0.00 on the dim night frame YOLO11n cleared at ~0.87), and
> keeping a second backend only to be a fallback meant the app could silently run
> a worse detector. There is **no silent fallback** any more: if the selected
> model can't load, the detector raises a clear, actionable error naming the
> missing ONNX (see `d20app/detector.py` `_ensure_net`). A GPU `accelerator` that
> can't start still retries the *same* model on CPU first.

The active model is chosen by `detector_model` in `config.yaml` / the GUI:

- **`yolo11n`** (default) — `yolo11n.onnx` (~10 MB), Ultralytics YOLO11-nano,
  COCO-80, exported at **320×320**. Strong in low light / odd poses (~0.87 on a
  real dim night frame) at modest CPU (~28 ms at the bundled size). Class names
  live in `d20app/yolo.py` (`COCO_CLASSES`); `person` is index 0, `cat` is 15.
- **`yolo11m`** — `yolo11m.onnx` (~77 MB), YOLO11-**medium**, exported at
  **640×640**. More capacity for users with CPU headroom. Be honest about the
  trade-off: on our own night/day frames it ran ~146 ms @320 / ~500 ms @640
  (≈5–18× nano) and **did not beat nano on the night case** that motivated it —
  nano @320 scored ~0.865 vs medium @640 ~0.914, but nano already clears the bar.
  Try it on genuinely hard scenes; don't assume it's strictly better.
- **`yolo11m_960` / `yolo11m_1280`** — higher-resolution locator exports (see
  below). **`yolo26m`** (bundled) / **`yolo26x`** (export-only) — the YOLO26 line.

The variant → file/size mapping lives in `d20app/yolo.py` (`MODELS`). Pick
resolution via the **model name** (e.g. `yolo11m_960`); there's no separate "net
input size" control — a YOLO ONNX is a fixed-shape export.

### Running YOLO on a GPU / Intel iGPU (`accelerator`)

By default the YOLO model runs on the CPU. The `accelerator` setting (Detection
card in the GUI, or `accelerator:` in `config.yaml`) can offload it:

- `cpu` (default) — OpenCV `cv2.dnn` on the CPU.
- `opencl` — the same net with OpenCV's `OPENCL_FP16` target, so the conv layers
  run on an OpenCL device such as an Intel iGPU. **No extra Python install**, but
  the host needs an OpenCL runtime (e.g. `intel-opencl-icd`). OpenCV silently
  falls back to CPU if there's no usable OpenCL device, so it's safe to try.
- `openvino-gpu` / `openvino-auto` — run the ONNX through Intel's **OpenVINO**
  runtime (optional `openvino` package; `setup.sh`/`setup.ps1` offer it) on the
  `GPU` device, or `AUTO` (GPU with a built-in CPU fallback). Typically 2–4× CPU
  on Intel hardware and the thing that makes `yolo11m` practical.

**Caveats, honestly:** OpenVINO's GPU plugin is **Intel-only** — it does nothing
on AMD/ARM and needs the host Intel GPU compute drivers installed. Whatever you
pick, the app retries the same model on CPU if the accelerator can't start, so a
missing driver won't break detection. The OpenVINO path is verified
end-to-end on the CPU device in the test suite; the real **iGPU** speed-ups are
from Intel's published figures and should be confirmed on your own hardware.

To check what your box actually does, run the diagnostic — it lists the detected
devices, resolves your `accelerator` to a real GPU vs a silent CPU fallback, and
times CPU vs your backend:

```
./venv/bin/python check_accelerator.py
```

Worth knowing: even with **no GPU**, OpenVINO's CPU runtime measured ~3× faster
than `cv2.dnn` on a dev box (yolo11m ~465 ms → ~150 ms/frame), so `openvino-auto`
can be a free win and is what makes `yolo11m` practical on CPU-only hardware.

## Re-exporting the YOLO ONNX files

Each bundled ONNX is a fixed-size export of the matching Ultralytics `*.pt`.
Ultralytics + PyTorch are needed **only to export** (one-off, offline) — they are
*not* runtime dependencies; the app runs the ONNX via `cv2.dnn` alone.

```
pip install ultralytics            # pulls torch; do this in a throwaway venv
# nano @ 320 (default):
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=320, opset=12, simplify=True)"
mv yolo11n.onnx d20app/models/yolo11n.onnx
# medium @ 640:
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
mv yolo11m.onnx d20app/models/yolo11m.onnx
```

The input size is fixed at export time; if you re-export at a different size,
update that variant's `size` in the `MODELS` table in `d20app/yolo.py` to match
(OpenCV's importer needs a static shape).

### High-resolution locator variants (optional)

The still-cat "locator" scan can run at a larger input to resolve a small/distant
cat (Option A). Those variants — `yolo11m_960`, `yolo11m_1280` — are already
registered in `MODELS` but **not committed**; produce them with the helper:

```
pip install ultralytics                 # export-only, offline; use a throwaway venv
python scripts/export_yolo.py --model yolo11m --imgsz 960
python scripts/export_yolo.py --model yolo11m --imgsz 1280
```

Until the file is present, selecting that locator size **falls back** to the native
size + tiling (no crash) — so tiling (Option B), which needs no export, is the
default.

### YOLO26 variants

`yolo26m` (bundled) and `yolo26x` (export-only — its ~213 MB ONNX is over GitHub's
100 MB limit) are the YOLO26 models. **They must be exported with the raw detection
head:** YOLO26 is NMS-free end-to-end and its default export emits a `(1, 300, 6)`
tensor that OpenCV's `cv2.dnn` mis-decodes (near-zero scores). The export helper sets
`end2end=False`, which produces the familiar `(1, 84, N)` head the app decodes
unchanged:

```
python scripts/export_yolo.py --model yolo26m --imgsz 640 --out yolo26m
python scripts/export_yolo.py --model yolo26x --imgsz 640 --out yolo26x
```

(If you export by hand instead of the helper, set
`m.model.model[-1].end2end = False` before `m.export(...)`, or the `cv2.dnn` path
will silently return garbage.)
