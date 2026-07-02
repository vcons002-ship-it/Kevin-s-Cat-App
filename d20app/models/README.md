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

The active model is chosen by `detector_model` in `config.yaml` / the GUI. The
selectable lineup is **benchmark-settled** (#70/#71 — full 199-cat + 43-null set,
golden exports, verified on the deployment 3070):

- **`yolo26x`** — the **workhorse**: 91% recall / 0% FP at 3×3/0.20 tiling;
  167 ms warm as FP16 on the 3070. Export-only (~213 MB, over GitHub's limit).
- **`yolo26m`** (bundled, ~79 MB) — the **lightweight**: 82%/0% at 2×2/0.20,
  64 ms FP16.
- **`yolo11n`** (bundled, ~10 MB, 320×320, the default) — the **floor**: 75%/0%
  at 3×3, tiny and CPU-friendly. Class names live in `d20app/yolo.py`
  (`COCO_CLASSES`); `person` is index 0, `cat` is 15.
- **FP16 variants** (`yolo26x_fp16` / `yolo26m_fp16`, export with `--half`) —
  identical accuracy, up to **2.2×** faster on CUDA (#70 §4). Verified with
  onnxruntime-CUDA; `cv2.dnn`'s FP16 handling is unverified — pair them with the
  `auto`/`onnx-cuda` accelerators.
- **Dropped by #71** (still loadable from old configs, not selectable for new):
  `yolo11m` and its `_960`/`_1280` locator exports — golden-exported 26m beats
  11m, and 11x/26n lost their tiers too (the earlier "drop 26m/26n" call was an
  artifact of the end2end export bug, now guarded against).

The variant → file/size mapping lives in `d20app/yolo.py` (`MODELS`). Pick
resolution via the **model name**; there's no separate "net input size" control —
a YOLO ONNX is a fixed-shape export.

> **Golden export or it doesn't count** (#70 §2): every deployed model must have
> the raw `(1, 84, N)` head with **no NMS/TopK ops** (`end2end=False`). A bad
> export shows `(1, 300, 6)` + a `TopK` op and silently costs 4–9 recall points —
> the app now **refuses to decode** such a head, with an error naming the fix.
> Verify a file with:
> ```
> python -c "import onnx; m=onnx.load('MODEL.onnx'); print(m.graph.output[0].type.tensor_type.shape); print([n.op_type for n in m.graph.node if n.op_type in ('NonMaxSuppression','TopK')])"
> ```

### Running YOLO on a GPU / Intel iGPU (`accelerator`)

The `accelerator` setting (Detection card in the GUI, or `accelerator:` in
`config.yaml`) picks where the net runs:

- `auto` (default since #71) — **CUDA when it genuinely binds, else CPU.** The
  CUDA attempt uses the same verified-provider check as `onnx-cuda` (so "auto"
  can never silently run slow); if CUDA isn't available the fallback to CPU is
  logged loudly. NAS-verified: onnxruntime binds `CUDAExecutionProvider` on the
  3070 at ~2× CPU minimum (up to ~37× on the heavy models).
- `cpu` — OpenCV `cv2.dnn` on the CPU.
- `opencl` — the same net with OpenCV's `OPENCL_FP16` target, so the conv layers
  run on an OpenCL device such as an Intel iGPU. **No extra Python install**, but
  the host needs an OpenCL runtime (e.g. `intel-opencl-icd`). OpenCV silently
  falls back to CPU if there's no usable OpenCL device, so it's safe to try.
- `openvino-gpu` / `openvino-auto` — run the ONNX through Intel's **OpenVINO**
  runtime (optional `openvino` package; `setup.sh`/`setup.ps1` offer it) on the
  `GPU` device, or `AUTO` (GPU with a built-in CPU fallback). Typically 2–4× CPU
  on Intel hardware and the thing that makes `yolo11m` practical.
- `onnx-cuda` — run the ONNX through **onnxruntime-gpu** on an **NVIDIA GPU**
  (CUDAExecutionProvider). Measured **~23 ms/inference on an RTX 3070 vs
  ~485–855 ms on CPU — a ~37× speedup**, which is what makes the heavyweight
  `yolo26x` runnable continuously (≈93 ms/frame at 2×2, ≈210 ms at 3×3). Optional
  `onnxruntime-gpu` package; `setup.sh`/`setup.ps1` offer it. **Use the CUDA-12
  build** to match a CUDA 12.x host — the default pip build targets CUDA 13 and
  fails with `libcudart.so.13 not found`.

**Caveats, honestly:**
- **OpenVINO's GPU plugin is Intel-only** — it does nothing on AMD/ARM and needs
  the host Intel GPU compute drivers installed.
- **`onnx-cuda` is NVIDIA-only, and has a silent-failure trap:** onnxruntime-gpu's
  CUDA provider **silently falls back to CPU** (37× slower, but *looks* like it
  works) unless the CUDA runtime libs (`libcublasLt.so.12`, `libcudnn.so.9`, …) are
  discoverable. The clean fix is torch's bundled `lib/` dir, which already has a
  compatible CUDA 12 / cuDNN 9 set — `run.py` prepends it to `LD_LIBRARY_PATH`
  automatically (and re-execs once) when `onnx-cuda` is selected. If you launch
  another way, export it yourself:
  ```
  export LD_LIBRARY_PATH=$(./venv/bin/python -c "import os,torch;print(os.path.dirname(torch.__file__)+'/lib')"):$LD_LIBRARY_PATH
  ```
  The app **detects the silent CPU fallback** (it checks the session's active
  provider) and **errors loudly** rather than running 37× slow without telling you.
  TensorRT EP isn't wired up — CUDA at ~23 ms is already plenty.

Whatever you pick, the app retries the same model on CPU if the accelerator can't
start, so a missing driver won't break detection. The OpenVINO and onnxruntime
decode paths are verified end-to-end on the CPU in the test suite (identical boxes
to `cv2.dnn`); the real **GPU** speed-ups should be confirmed on your own hardware.

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
# FP16 (the deployment picks, #70 §4 — pair with the auto/onnx-cuda accelerator):
python scripts/export_yolo.py --model yolo26x --imgsz 640 --out yolo26x --half
python scripts/export_yolo.py --model yolo26m --imgsz 640 --out yolo26m --half
```

(If you export by hand instead of the helper, use the full golden recipe —
`m.model.model[-1].end2end = False` before
`m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False,
batch=1, nms=False)` — or the decode guard will refuse the file.)
