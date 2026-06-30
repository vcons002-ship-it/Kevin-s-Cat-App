# Changelog

All notable changes to **Kevin's Cat App** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/).
Version numbers below were assigned retroactively (the repo isn't tagged yet);
everything through the latest entry is on `main`.

## [Unreleased]

_Nothing yet — see [`ROADMAP.md`](ROADMAP.md) for what's planned._

## [0.27.0] — 2026-06-30

### Added
- **Moondream VLM tester: validated load params, an API-key field, and a local/cloud mode
  selector** (#59) — the complete, hardware-validated picture from bringing moondream up on
  an 8 GB RTX 3070.
  - **Validated local load.** Local inference now uses the params that actually work on
    8 GB: `md.vl(api_key=…, local=True, model="moondream2", max_batch_size=4,
    kv_cache_pages=2048)`. The default auto-sized KV cache OOMs (`CUBLAS_STATUS_ALLOC_FAILED`)
    and `kv_cache_pages=256` loads but can't serve; **2048** loads (~4.6 GB) **and** serves.
  - **API key in the GUI.** A password field in the VLM card stores the key in config
    (`moondream_api_key`) — **never logged, never sent back to the browser** (`/api/config`
    exposes only a `has_moondream_api_key` flag), and a **blank field on save keeps the
    stored key**. A missing key shows a clear "⚠ no key set — paste yours" message instead
    of a cryptic 401. `MOONDREAM_API_KEY` still works for headless use.
  - **Mode selector.** Two coherent choices, same `md.vl()` interface:
    *Moondream 2 — local* (default; private, ~0.3 s, fits 8 GB) and *Moondream 3 — cloud
    API* (most accurate, but **sends images off-device** — the UI says so, and shows a
    warning when cloud is picked). Cloud mode needs **no GPU**; only local does.
  - **moondream3 local is not offered** — Photon has no weight quantisation, so the ~9B M3
    loads at bf16 and needs >8 GB; on a small card it's cloud-only (the selector reflects this).
  - **Friendly GPU errors.** CUDA OOM, the "insufficient KV cache" stall, and cuBLAS alloc
    failures map to actionable messages ("lower max_batch_size", "raise kv_cache_pages",
    "this model doesn't fit this GPU's VRAM") instead of a raw traceback.
  - Mode is threaded through every VLM path: single query, the batch tester, and the
    detection-batch "also run VLM" toggle.

### Notes
- Suite: **218 tests** (+7 for the mode selector, the validated local params, the masked
  config key + blank-keeps-stored behaviour, the local-needs-GPU/cloud-doesn't split, and
  the friendly OOM/KV error mapping), all green. The model itself runs on the NAS; here the
  load-call shape, mode plumbing, key masking, and error mapping are verified with mocks.

## [0.26.0] — 2026-06-30

### Added
- **NVIDIA GPU acceleration for YOLO** via a new `onnx-cuda` accelerator (#58). It runs
  the YOLO ONNX through **onnxruntime-gpu** (CUDAExecutionProvider) instead of `cv2.dnn`.
  On the NAS's RTX 3070 this is **~37× faster** (~23 ms/inference vs ~485–855 ms on CPU) —
  fast enough to run the heavyweight `yolo26x` continuously (≈93 ms/frame at 2×2,
  ≈210 ms at 3×3). Selectable in the GUI (Detection card + per-camera editor + Test
  tool) and as `accelerator: onnx-cuda` in `config.yaml`.
  - New `_OnnxRuntimeRunner` in `d20app/yolo.py` — same `infer(blob) → (1,84,N)` contract
    as the cv2.dnn / OpenVINO runners, so the letterbox + NMS decode is unchanged
    (verified: onnxruntime's output decodes to **identical boxes** to cv2.dnn).
  - **Silent-CPU-fallback guard.** onnxruntime-gpu's CUDA provider quietly degrades to
    CPU (37× slower, but looks like it works) if its CUDA runtime libs aren't
    discoverable. The app (a) prepends **torch's bundled CUDA-12 / cuDNN-9 lib dir** to
    `LD_LIBRARY_PATH` — `run.py` does this and re-execs once when `onnx-cuda` is selected;
    `setup.sh`/`setup.ps1`/`models/README.md` document the manual export — and (b) checks
    the session's **active provider** after creation and **raises loudly** if CUDA was
    requested but CPU was chosen, instead of running slow in silence.
  - **Graceful fallback.** If `onnx-cuda` can't start (no onnxruntime-gpu, no CUDA, or the
    silent-CPU trap), the detector retries the **same** model on CPU with a clear warning.
  - Optional dependency: `onnxruntime-gpu` (use the **CUDA-12** build to match a CUDA 12.x
    host; the default pip build targets CUDA 13 and fails with `libcudart.so.13 not found`).
    Not installed by default — `setup.sh`/`setup.ps1` offer it.
  - `check_accelerator.py` now reports the CUDA/onnxruntime status and whether a session
    actually lands on the GPU vs a silent CPU fallback.
- TensorRT EP is intentionally **not** wired up (it needs separate TensorRT libs); CUDA at
  ~23 ms is already plenty. Left as a future "max speed" tier.

### Notes
- Suite: **211 tests** (+6 for the onnx-cuda runner — accelerator registration, the
  no-CUDA-provider error, the silent-CPU-fallback detection, decode parity with cv2.dnn,
  and the detector's CPU fallback), all green. The GPU speedup itself is NAS-only; the
  decode + fallback + error paths are verified here on CPU.

## [0.25.0] — 2026-06-30

### Removed
- **MobileNet-SSD is gone** (#57). It was the original detection backend and the
  automatic fallback when a YOLO model couldn't load, but it lost every benchmark to
  YOLO (notably it scored **0.00** on a dim night frame YOLO11n cleared at ~0.87), and
  keeping a second backend purely as a fallback meant the app could **silently run a
  worse detector**. Deleted: the `mobilenet_ssd.caffemodel` (~23 MB) + `deploy.prototxt`
  weights, the SSD blob path and the `CLASSES`/`PERSON_CLASS_ID`/`person_in_detections`
  helpers in `detector.py`, the `mobilenet_ssd@{300,512,768}` dropdown/benchmark
  variants, and the now-dead `_ssd_size()` / `@size` plumbing.

### Changed
- **No silent fallback — a model that can't load now raises a clear, actionable error**
  (#57, the option chosen for this change). If the selected YOLO ONNX is missing or
  fails to load on CPU, the detector raises a `RuntimeError` naming the model and
  pointing at `d20app/models/` / `models/README.md`, instead of quietly downgrading.
  A **GPU `accelerator`** that can't start still retries the **same** model on CPU first
  (that path is unchanged) — only the cross-*model* fallback is removed.
- **Every detection model is now a YOLO variant**; the model name carries the input size
  (`yolo11n` 320 → `yolo11m` 640 → `yolo11m_960` 960 → `yolo26m`/`yolo26x`). The legacy
  `detect_size` config field is **kept but ignored** so old `config.yaml`s still load.
- `check_camera.py` now runs the configured YOLO model through the normal box path
  (it previously poked the raw SSD blob), so the diagnostic matches what the app does.

### Notes
- Honest fixture finding (verified by eye while re-pointing the accuracy guard at
  yolo11n): two single-cat frames yield a person box — `cat15.jpg` **genuinely contains
  a person** (a hand holding a camera) that SSD had missed, and `cat23.jpg` is a true
  top-down misread, the YOLO analogue of the old cat-cluster misreads (neutralised live
  by the `confirm_frames` temporal gate). Both are documented and excluded by name in
  `tests/test_detection_accuracy.py`, so a *new* cat starting to trigger still fails.
- Suite: **205 tests** (down from 213 — SSD-specific tests removed), all green.

## [0.24.0] — 2026-06-29

### Added
- **Batch VLM tester** (#54). The moondream tester now runs across many images like
  the detection models do: add multiple frames (with the same per-image **"cat present"**
  ground-truth toggle), run **one query per image** (no sweep), and get a per-image
  table (**yes/no + reason + latency**) plus a summary that reports **recall** (on
  cat-present frames) and **false-positive rate** (on no-cat frames) **separately** —
  the same methodology as the detection benchmark. Cancel/abort supported.
  Endpoint: `POST /api/vlm/batch`.
- **Optional "also run VLM" toggle in the detection batch** (#54). Off by default (no
  added cost). On, it runs the VLM **once per image** alongside the model × tiling
  sweep and adds to the cross-image summary:
  - a **"VLM accuracy for batch"** line — recall + false-positive rate, graded against
    the same per-image "cat present" labels that score the YOLO configs;
  - the **frames where the VLM and the best YOLO config disagree** (VLM yes / YOLO miss,
    or VLM no / YOLO hit), each linked to its per-image report — the interesting rows,
    surfaced so you don't have to eyeball every frame.
  - Cost is **+1 VLM call per image** (not per config); VLM latency is reported
    separately from YOLO. If the VLM can't run (no GPU/key), the sweep still completes
    and the report notes the VLM was skipped.

## [0.23.2] — 2026-06-29

### Fixed
- **The moondream VLM tester now actually runs locally** (#52). The old call
  `md.vl(model=path)` was wrong — moondream's `vl()` has no `model=` parameter, so the
  argument was swallowed, `local` stayed `False`, and it silently hit the **cloud** API
  (HTTP 401, and would have sent frames off-machine). It now uses the documented local
  invocation `md.vl(api_key=…, local=True, model=…)`:
  - The model is selected **by name** (`moondream2` default, or `moondream3-preview`),
    not a file path — Photon manages its own Hugging Face weight cache (redirect with
    `HF_HOME`). The GUI has a model picker.
  - The **API key** comes from `MOONDREAM_API_KEY` (authenticates the one-time weight
    download; per-query inference is local — frames don't leave the machine).
  - **Local inference needs a supported GPU** (CUDA/Ampere or Apple Silicon) — there's
    **no CPU path**; the tester now raises a clear "needs a supported GPU" error instead
    of a cryptic Photon failure, and the earlier "multi-second-to-minute on CPU" framing
    is gone.
- **Benchmark XLSX export works out of the box** (#53). `openpyxl` is now a **real
  dependency** (uncommented in `requirements.txt`) instead of a manual `pip install` —
  a button in the shipped GUI shouldn't need an undocumented manual step to function.

### Changed
- **The VLM no longer reports a confidence number** (#54). moondream's self-reported
  confidence proved meaningless ("0-100%", "not sure but yes"), so the output is now just
  **yes/no + the reasoning text** (kept purely as information). The default prompt is
  "Is there a cat in this image? Answer yes or no, then briefly explain," and the parser
  was simplified to extract the yes/no and keep the rest as free-text reason.

## [0.23.1] — 2026-06-29

### Fixed
- **YOLO26 (and any model) now appears in the live/Test picker, not just the
  benchmark** (#50). The Test-detection model dropdown was a **hardcoded** list in the
  template that still stopped at YOLO11 + SSD, so `yolo26m`/`yolo26x` could be
  *swept* but not *selected* for deployment or single-image tests — the mirror of the
  earlier "SSD variants missing from the sweep" drift. Fixed the **pattern**, not just
  the instance: a new `GET /api/models` returns the canonical present-checked
  `{value,label}` list derived from the model registry, and **every** GUI model
  dropdown (the Test picker and the per-camera editor) plus the benchmark sweep are now
  built from that one source — they can't desync again, and any model added to the
  registry shows up everywhere automatically. Export-only variants (yolo26x,
  yolo11m_1280) appear only once their ONNX is present, so the picker never offers a
  model that can't load.

## [0.23.0] — 2026-06-29

### Added
- **Cat-presence (VLM) tester** (#48). A new GUI panel that asks a small
  vision-language model (**moondream**) "*is* there a cat?" about a whole frame —
  a different angle from the box detectors: it *reasons*, so it's strong on the hard
  frames they miss and on the decoys (beds, posters) that open-vocabulary detectors
  false-fire on. v1 is a **manual evaluation tool**, not a live path:
  - **Single `query` pass** with an **editable, format-instructed prompt** (the panel
    doubles as a prompt bench — VLM results are very prompt-sensitive). One call yields
    both the reasoning and a best-effort yes/no (no double-latency two-pass).
  - **The full raw response is always shown** — the reasoning is the point: it reveals
    *why* it answered (e.g. "the round fuzzy object looks like a cat"), which is exactly
    how you catch a decoy confusion. A best-effort **Yes/No** + **stated confidence**
    are parsed out and shown, with confidence labelled as the model's **self-report,
    not a calibrated score**. If the model ignores the format, the fields read
    **unparsed** and the raw text still shows — it never invents an answer.
  - **Latency** is split into one-time **model load** vs **per-query** time, with the
    prompt / model / device shown for reproducibility.
  - Input is an uploaded image or a frame from an uploaded video (reuses the existing
    frame extractor). Endpoints: `GET /api/vlm/status`, `POST /api/vlm/query`.
- **`moondream` is an optional dependency** (mirrors openvino/openpyxl/playsound3):
  `setup.sh`/`setup.ps1` offer it, and the panel degrades with a clear "install
  moondream + point `MOONDREAM_MODEL` at a model" message when it (or the multi-GB
  model) is absent — Run never 500s.

### Notes
- **The live path runs on the NAS**, not here: the model is multi-GB (not committed)
  and CPU inference is multi-second-to-minute. The wiring + the response **parser** are
  unit-tested; the actual moondream call is exercised where the model lives. `detect`
  mode (VLM-guided cropping → YOLO) is deliberately **out of scope** for v1.

## [0.22.0] — 2026-06-29

### Added
- **YOLO26 detection models** (#45). Same COCO lineage as the YOLO11 models (so no
  open-vocabulary false-positive problem), tuned for small-object accuracy — a
  same-lineage candidate for the two-tier locator (cheap always-on + heavyweight
  escalation). Two variants in the model dropdown and the benchmark sweep:
  - **`yolo26m` (640)** — bundled (~79 MB), a head-to-head everyday-model candidate
    against `yolo11m`. Found the cat at **0.85** on the fixture (vs the model's own
    0.85 reference) through the unchanged CPU/`cv2.dnn` path.
  - **`yolo26x` (640)** — the heavyweight (the model that cracked the hard frames in
    hand-testing). Its ONNX is **~213 MB**, over GitHub's 100 MB limit, so it's
    **export-only** (graceful fallback when absent, like the larger YOLO11 exports):
    `python scripts/export_yolo.py --model yolo26x --imgsz 640 --out yolo26x`.
  - **Decode note:** YOLO26 is **NMS-free end-to-end** and its default export emits a
    `(1, 300, 6)` tensor that `cv2.dnn` mis-decodes (near-zero scores). The export now
    forces the **raw `(1, 84, N)` head** (`end2end=False`), which the existing letterbox
    + NMS decoder handles unchanged — verified against the `.pt`'s own predictions.

### Notes
- The real payoff (does `yolo26m` beat `yolo11m` on recall *and* NAS-iGPU speed? is
  `yolo26x` worth it as an escalation tier?) needs the **target hardware** — run the new
  models through the batch benchmark on the NAS to get those numbers. The wiring + CPU
  decode are validated here; the head-to-head is not.

## [0.21.0] — 2026-06-29

### Added
- **The benchmark honours the tester's controls** (#44). The sweep used to run at
  hardcoded defaults, so you couldn't benchmark under the conditions you can hand-tune
  on one image. It now applies the **tile overlap, person confidence, and image
  adjustments (gamma / brightness / contrast / saturation)** from the Test-detection
  panel, **uniformly to every run** (one value each — not a sweep axis), on both the
  single-image and batch endpoints. Defaults match the old behaviour, and the report's
  "held fixed for every run" block shows the **actual values used** — so a seam-straddling
  cat missed at 0.2 overlap can be re-run at 0.45, or a backlit cat re-run with a gamma
  lift, and the two reports are distinguishable on paper.
- **The cross-image summary grid is navigable** (#42). The config×image heatmap's
  column headers (#1–#N) now **link to each image's per-image report** (by slug, so
  the link resolves live and in a downloaded/hosted set) and reveal the **image name +
  a thumbnail on hover**. Expanded miss entries are **numbered** (`#5 backyard.jpg`) so
  the grid column and the miss list share one identifier — from "column #5 is hard" to
  that frame's full report in one click.

## [0.20.1] — 2026-06-29

### Fixed
- **Big batches no longer silently truncate to the last few images** (#40). The
  upload-session store was capped at **4** (sized for single-image testing), so a
  23-image batch had its first 19 uploads **evicted before the batch ran** — and the
  batch then *silently skipped* the missing sessions and built a "successful"
  4-image summary. The session cap is now **tied to the batch ceiling**
  (`_TEST_MAX_SESSIONS == _BENCH_MAX_IMAGES_HARD`), so a batch can never accept more
  images than the store can hold, and the two can't drift apart again.
- **A partial batch can no longer masquerade as a complete one** (#40). If any image's
  upload is genuinely missing, the batch now **reports it** — the response carries
  `requested` / `ran` / `skipped` / `skipped_names`, and the GUI shows a prominent
  "⚠ Ran X of Y — N upload(s) were no longer available" instead of a quietly
  incomplete report.

## [0.20.0] — 2026-06-29

### Fixed
- **Batch summary links survive download/hosting** (#35). The cross-image summary
  linked each per-image report by its hash id, but the reports **download under their
  slug** filename — so once you saved the bundle and hosted it together (e.g. GitHub
  Pages), every miss link 404'd. The summary now links by **slug**, and the serving
  route resolves a slug as well as a hash id, so the links work both live and in a
  downloaded/hosted set.
- **Dark-theme links are legible** (#36). Links were the browser-default bright blue
  (harsh on the dark-blue background) and **unreadable purple once visited**. They're
  now an explicit **cyan**, with a differentiated-but-readable visited colour,
  underline on hover, and a visible keyboard-focus outline. The download "buttons"
  keep their white-on-box look.

### Added
- **"Download all" for a batch** (#35). One click returns a **zip** of the cross-image
  summary plus every per-image report, each named with its slug — unzip into a folder,
  host together, and all the links resolve. Endpoint: `GET /api/test/benchmark/<id>/all.zip`.
- **Soft cap + abort for batches** (#37). The recommended batch size (12) is now a
  **soft cap**: past it the GUI shows the run count and rough time and lets you proceed
  (only an absolute ceiling of 100 is refused). A new **Abort** button stops a running
  batch after the current image and returns the summary of whatever finished
  (`POST /api/test/benchmark/cancel`).
- **Video → "benchmark all frames"** (#38). A video in Test Detection now extracts
  **~1 frame per second** of the clip (was a hard-coded 8) — the count is selectable
  (max 100) with a **Re-extract** button — and a **"Benchmark all N frames"** button
  runs the sweep across every extracted frame into the same cross-image summary,
  reusing the batch endpoint. (Frame extraction feeding the still benchmark — not
  video-feed detection, which stays out of scope.)

## [0.19.0] — 2026-06-28

### Added
- **Batch benchmark + cross-image summary** (#32). The benchmark tool now accepts
  **several images at once** (add as many as 12) and runs the same models × tiling
  sweep on each — you still get the per-image self-contained report, **plus one
  cross-image summary** that answers the question single reports can't: *which config
  is most reliable across many frames?* The summary ranks every config by
  **detection rate** (e.g. `13/13`), with **average combined score** and **average
  inference time**, and is the table you'd use to pick a deploy config.
  - **Misses are traceable, never just counted.** An imperfect rate (e.g. `12/13`)
    is click-to-expand: it lists the **frames it missed** — a small original
    thumbnail, the score it got (`0.31, below 0.50` — near-miss vs total whiff), and
    a link to that image's per-image report. A perfect row has nothing to expand and
    stays quiet, so the eye goes to the configs worth investigating.
  - **config × image heatmap** so the whole pattern is visible at a glance — you can
    instantly see whether misses cluster on one hard frame (everything struggles) or
    are scattered (config-specific weakness).
  - **Size-safe:** each original thumbnail is embedded **once per image** in a catalog
    and referenced by id, so the summary stays ~1–2 MB regardless of how many configs
    miss a hard frame; the heavy annotated detail lives in the linked per-image reports.
  - **Empty-room controls.** Tick a frame "no cat" and the summary reports its
    **false-positive** count separately — a good locator finds the cat *and* stays
    quiet on empty rooms. Endpoint: `POST /api/test/benchmark/batch`.

## [0.18.2] — 2026-06-28

### Fixed
- **Benchmark thumbnails enlarge again** (#30). Clicking a thumbnail (in the HTML
  report *and* the in-GUI table) opened a blank `about:blank` tab — browsers block
  top-level navigation to a `data:` URL. They now enlarge **in-page via a tiny
  lightbox overlay** (click anywhere to dismiss); the report stays self-contained
  and works offline. The **download** link is untouched (the `download` attribute
  still works for `data:` URLs — only *navigation* to them is blocked).
- **SSD model labels no longer show the size twice** (#31). The MobileNet size
  variants read `mobilenet_ssd@512` in the model column next to `512` in the size
  column. The reports/table now show the **base name** (`mobilenet_ssd`) and let the
  size column carry the number, matching the YOLO rows. The `model` key itself is
  unchanged, so re-runs and the benchmark sweep still resolve the size from `@N`.

## [0.18.1] — 2026-06-28

### Fixed
- **Benchmark thumbnails are legible now** (#25). The annotated frames are encoded
  at **680px** (was 240px, which threw the box/label detail away *before* encoding),
  displayed small in the table but **click-to-enlarge** — tidy grid, full detail on
  demand, still a single emailable file (~0.5–1 MB for a 16-run sweep).
- **The benchmark XLSX button no longer sits there dead** (#26). When `openpyxl`
  isn't installed the download control is **visibly disabled with a note explaining
  why and how to fix it** ("XLSX export needs the 'openpyxl' package — pip install
  openpyxl"), instead of a silent no-op link. (`openpyxl` stays optional by design.)
- **Benchmark report downloads have human-readable filenames** (#27). HTML/XLSX are
  served with a `Content-Disposition` slug like `benchmark-living-room-cam-20260628-1432`
  built from the uploaded image name + timestamp, not the opaque report UUID.
- **The default sweep model list no longer drifts from the dropdown** (#28).
  `_benchmark_models()` now returns the present YOLO variants **plus the MobileNet
  size variants** (`mobilenet_ssd`, `@512`, `@768`), so the "include models" checkboxes
  match what the manual model picker offers.

### Added
- **The original (unannotated) frame travels with the report** (#25). The source
  frame is embedded once at full native resolution with a **download link**, so the
  report is a self-contained experiment record — fixed settings + exact input +
  per-run results — and anyone can extract that frame and re-run the identical
  benchmark after a code/model change to compare.

## [0.18.0] — 2026-06-28

### Added
- **"Benchmark this image"** (#21). One click in the Test-detection tool sweeps
  **models × tiling** on the uploaded frame and emits a **shareable report** — a
  **self-contained HTML** file (every annotated thumbnail inlined as base64, no
  external assets, so it emails/opens anywhere off the LAN-only NAS) **and an
  optional XLSX** with embedded thumbnails. Each run records the **best cat score,
  best dog score, combined (cat-or-dog) score**, whether it cleared the cat
  threshold, and **inference time** (summed across tiles). Checkboxes trim the
  matrix; a "settings held fixed for all runs" block makes results unambiguous.
  Reuses the live `detector`/`yolo` code (no parallel detection path). Endpoints:
  `POST /api/test/benchmark`, `GET /api/test/benchmark/<id>.{html,xlsx}`.

### Notes
- **XLSX needs the optional `openpyxl`** (mirrors the openvino/playsound3 pattern):
  `setup.sh`/`setup.ps1` offer it, and it degrades gracefully — the HTML report
  always works and the response carries an `xlsx_error` if openpyxl is absent.
- The sweep runs synchronously and is **capped at 24 runs**; a 4×4 × `yolo11m_960`
  matrix is genuinely heavy on CPU (16 tiles/run) — trim it or run it on the iGPU.
  180 tests (was 176): the sweep matrix, self-contained HTML, XLSX present/absent
  paths, and the run cap.

### Added
- **Independent cat confidence** (#19). Cat detection was gated by `label_floor`
  (shared with every other non-person mover); it now has its own **`cat_confidence`**
  (default 0.5), tunable per camera and in the Test-detection tool — separate from
  `person_confidence` and `label_floor`. The treat/roll path is untouched.
- **"Count dogs as the cat"** (#22). The model often mislabels a cat as a **dog**
  with high confidence, and more resolution raises *dog*, not *cat*. A no-dog house
  can set **`locator_classes: ["cat", "dog"]`** so a dog counts as a locator hit
  (recorded with its real label). Default stays cat-only. New `label` field on cat
  sightings + `/api/cats`.
- **Resolution lives in the model dropdown** (#20). The "Net input size" control was
  a **no-op for YOLO** (fixed-shape exports) and only worked for MobileNet — removed.
  MobileNet sizes are now named variants in the model picker (**`mobilenet_ssd@512`**,
  `@768`); YOLO already encodes its size (320/640/960). `detect_size` is kept only for
  reading old configs.

### Notes
- New config fields `cat_confidence` / `locator_classes` (global + per-camera). The
  net's return-floor now also accounts for `cat_confidence`, so a *lower* cat threshold
  genuinely surfaces fainter cats. 176 tests (was 168). Reproduces #22: with
  `["cat","dog"]`, a cat the model calls a dog becomes a logged sighting.

## [0.16.0] — 2026-06-28

### Added
- **Higher-resolution locator scan** (fixes the still-cat half of issue #17). A
  sleeping cat shrunk into a 640 frame can be **too small for the net to detect at
  all**; the periodic still-cat scan now runs at higher effective resolution while
  the fast treat path is untouched:
  - **Tiling (default `4x4`)** — splits the frame into an overlapping grid and
    detects per tile, so a small/distant cat fills more of the net's input. Merged
    with NMS. Works with the bundled models, no re-export. `off`/`2x2`/`3x3`/`4x4`
    (4×4 is what actually resolved a sleeping cat in testing; fewer tiles = less CPU).
  - **Larger input (`cat_scan_imgsz`)** — MobileNet resizes freely; for YOLO a
    **`yolo11m_960`** export is now bundled (and `scripts/export_yolo.py` +
    `yolo11m_1280` registration for more). Missing export → graceful fallback.
  - Per-camera settings (defaults are global); the **Test detection** tool gains
    **tiling / tile-overlap / accelerator** knobs and an **inference-time** readout
    so you can A/B configs on real frames — the rest of the GUI tester issue #17 asked for.
- **Round-robin camera scheduling.** Optional: watch only **N cameras at a time**
  and **rotate** through the rest on an interval, so many cameras cost about as much
  as a few. Resting cameras release their capture (stop decoding — the CPU win).
  Per-camera **"always watch" (👁)** exempts a camera (e.g. the treat camera); the
  camera you're viewing live never rests. GUI toggle + size + interval; a **💤
  resting** chip per camera. `round_robin` / `round_robin_size` / `round_robin_interval`.

### Notes
- New config: `cat_scan_tiling`/`cat_scan_tile_overlap`/`cat_scan_imgsz` and
  `round_robin*` (global + per-camera `always_watch`/locator). `GET /api/test/detect`
  returns `inference_ms`. 168 tests (was 157): tiling/NMS-merge, locator fallback,
  round-robin rotation/rest/always-watch/viewed-pin, and the new endpoints/config.
- A bundled **`yolo11m_960.onnx`** (~78 MB) ships for Option A; `yolo11m_1280` is
  script-only (run `scripts/export_yolo.py`). Round-robin rests release/reopen the
  capture each rotation — there's reconnect latency, amortised by the interval.

## [0.15.0] — 2026-06-28

### Added
- **"Test detection" tool.** Upload a **photo or a short video** and run the
  detector on it right in the GUI, with boxes drawn and a list of what it found
  (person/cat + score). A video is sampled into an evenly-spaced **filmstrip** you
  step through. The settings that actually affect whether it identifies things —
  **model, net input size, person confidence, notify floor, and the new image
  adjustments** — are surfaced as live sliders; tweak and it re-runs on the spot.
  Save the result to a specific **camera** or to the **global defaults**.
  Endpoints: `POST /api/test/upload`, `POST /api/test/detect`.
- **Image adjustments (gamma / brightness / contrast / saturation).** Applied to
  each frame **before the net runs**, so they can rescue a too-dark or washed-out
  feed into the detector's comfort zone. They're **real per-camera detection
  settings** (defaults are all no-ops) — tune them in the test tool, then save to
  the camera. In the synchronous path the live feed shows the adjusted frame.

### Notes
- New config fields `gamma` / `brightness` / `contrast` / `saturation` (global +
  per-camera). 157 tests (was 143): adjustment maths, `detect_image`, video
  sampling, the new config fields, and the upload/detect endpoints.
- In **smooth-feed** mode the grab thread still publishes the *raw* frame, so the
  live view there stays unadjusted while the net sees the adjusted pixels.

## [0.14.1] — 2026-06-28

### Added
- **"Show cat" detection boost.** Tapping the button now runs the jumped-to
  camera's detector **continuously for ~20 s**, so the live feed keeps drawing a
  box around the cat — even a motionless one between periodic scans, or when
  still-cat scanning is set to _Off_ — while you find it. During multi-room
  rotation, each room is boosted as the feed lands on it. New
  `POST /api/cats/boost {camera}` and `DetectionLoop.boost_detection()`; the boost
  never rolls (no-motion frames stay roll-free). 143 tests (was 141).

## [0.14.0] — 2026-06-28

### Added
- **Still / sleeping-cat detection.** A motionless cat never trips the motion
  pre-filter, so it used to go unseen. A cat-tracking camera now **periodically
  runs the net even with no motion** to catch a still cat — the "Show me the cat!"
  button flashes whether the cat is moving *or* sleeping. The cadence is a GUI
  setting (**Cat cam → "Check for a still cat"**): _Always_ (net every frame, most
  CPU), every 5 s / 15 s / 30 s (default) / 1 / 2 / 5 min, or _Off_ (motion only).
  A still cat is logged once per visit (rising edge), not once per scan.
- **Multi-room Show-cat rotation.** When **more than one camera** has a cat right
  now, tapping "Show me the cat!" rotates the live feed between those rooms; the
  button reads "Cats in N rooms — show me!". Picking a camera manually stops the
  rotation. `GET /api/cats` now returns a `cameras` list of rooms seeing a cat.

### Notes
- A forced still-cat scan **never rolls** — rolling stays gated on real motion, so
  a motionless person found by a scan can't trigger a treat.
- New `cat_scan_interval` config field (default `30.0`). 141 tests (was 133):
  periodic still-cat scan, rising-edge de-dup, always-on/off, the per-room present
  list, and the no-roll-on-still-scan guard.

## [0.13.0] — 2026-06-27

### Added
- **Multi-camera.** Watch several cameras at once, each with **its own role** and
  **its own full detection settings**:
  - **Roles** (per camera): 🎲 *Rolls* — a person there rolls for a treat; 🐱
    *Tracks cats* — its cat sightings are logged. One, both, or neither.
  - **Per-camera settings**: model, accelerator, person-confidence, confirm-frames,
    detection detail, scan rate, notify floor, motion sensitivity, region of
    interest, plus URL/credentials — independent per camera.
  - **GUI**: the Camera card is now a manager — add cameras, tick **Watch** to run
    them at once, expand each to edit its settings/ROI, and see per-camera
    connected/failing chips. The Live-detection card gains a **camera selector**,
    and **Show cat** jumps the feed to the camera that saw the cat.
  - **Detection loop**: one `PersonDetector` + worker thread per watched camera,
    sharing **one treat dispenser** (a single cooldown/roll gate across all
    cameras). One camera failing never stops the others.
  - **Config/API**: per-camera config dicts + `active_cameras`;
    `config.camera_targets()`/`coerce_camera()`; `GET /api/stream?camera=`,
    `GET /api/preview?camera=`, `POST /api/cameras/active`, full per-camera
    `POST /api/cameras/saved`, and per-camera status in `GET /api/status`.

### Notes
- **CPU scales with the number of watched cameras** (each is its own inference
  stream — you can't share one inference across different cameras' frames), but the
  motion pre-filter means idle cameras cost almost nothing, and each camera can be
  tuned lighter (mobilenet / lower scan-rate / tight ROI). Two heavier CPU options —
  a round-robin shared-detector mode and GPU-batched inference — are recorded in
  ROADMAP as future work, not built here.
- **Backwards compatible**: an existing single-camera `config.yaml` (no
  `active_cameras`) runs exactly as before, as one camera that both rolls and
  tracks cats.
- The new threading (one shared cooldown across N camera threads) was put through an
  adversarial multi-lens review before merge, which hardened: a **cat-tracking
  camera keeps watching during another camera's roll-cooldown** (only roll-only
  cameras pause to save CPU); duplicate names in the watch set are de-duped (no two
  threads on one capture); an explicit per-camera `roi: null` means whole-frame; and
  the shared snapshot store no longer serialises workers across a slow-disk write.

## [0.12.0] — 2026-06-27

### Changed
- **GUI reorganised around the cat.** The page now leads with the running
  controls (Start/Stop in the status bar) and a prominent **🐱 Cat cam** card,
  followed by the live feed and the activity log. All the camera/speaker/rules
  **setup moves to the bottom under a "Setup & settings" divider** — it's saved
  and rarely touched, so it no longer dominates the page.
- **Big, fun "Show me the cat!" button** that **flashes green and bounces while a
  cat is actually on camera right now** (and reads "Cat spotted — show me!").
  Tapping it still pulls up the live feed.

### Added
- `cat_present` signal: `GET /api/cats` now returns `present` (a cat is on camera
  this moment, via a fresh cat box above the label floor). The GUI polls it ~1 s
  for a near-real-time flash. Respects `prefers-reduced-motion`.

### Fixed
- The live-feed image no longer renders as a broken-image icon when stopped
  (a CSS specificity slip let `.roi-stage img` override `.hidden`).

### Added
- **Smooth live feed** (optional, off by default). A "Smooth feed" checkbox on the
  Live detection card runs a **dedicated capture thread** that reads the camera
  continuously, decoupled from inference — so the video plays at the camera's
  frame rate instead of stuttering at the scan rate (which is gated by detection
  speed, especially on a slow CPU or the heavier `yolo11m`). Toggles live while
  watching, and persists.
  - `detector.py`: a grab thread becomes the **sole** camera reader in smooth
    mode; the loop samples its latest frame for detection. All capture start/stop
    is reconciled on the **loop thread** (the web request only sets a desired
    flag), so the capture never has two readers — important for USB cameras,
    which can't be opened twice. New `smooth_live_feed` config field.
  - `/api/stream` now re-encodes only when the frame/box **version** changes
    (no fixed ~10 fps cap, no duplicate-frame encodes), so the feed runs at
    whatever rate frames actually arrive. New `POST /api/live/smooth`.

### Notes
- Smoothness is still bounded by the **camera's real output rate** and the LAN —
  smooth mode removes the *inference* bottleneck, not those. It costs a little
  extra CPU and reads the camera continuously; leave it off if you only need the
  occasional frame.
- The threading was put through an adversarial multi-lens review (concurrency,
  lifecycle, regression, edge cases) before merge, which hardened the
  stalled-camera paths: the grab thread self-heals if it ever dies while smooth
  is on, a camera that dies *after* a good frame is now surfaced to the loop
  (not silently frozen on the stale frame), a wedged grabber on toggle-off keeps
  reading rather than exiting, and shutdown never releases the capture out from
  under an in-flight read.

## [0.10.0] — 2026-06-27

### Added
- **Cat tracking + "Show cat".** Cats still never trigger a treat (only people
  roll), but instead of being ignored, every cat sighting is now recorded —
  **when**, on **which camera**, and roughly **where** in the frame (a thirds
  grid: e.g. *bottom-left*) — with an annotated snapshot. A new **Cats** card
  shows the latest sighting and today's count, and a **Show cat** button pulls up
  the live feed of the camera that saw it.
  - New `d20app/cats.py` `CatTracker` (thread-safe, bounded, file-backed like the
    activity log; survives restarts) and a `describe_region()` location helper.
  - New endpoints: `GET /api/cats` (last sighting, today's count, recent list —
    each carrying its `camera`) and `POST /api/cats/clear`.

### Notes
- Built **camera-aware now** for the single camera the app watches today: each
  sighting stores its `camera`, so the planned multi-camera "Show cat" only needs
  to point the live feed at the sighting's camera. Until then the button pulls up
  the one feed.
- Sightings reuse the loop's existing throttle and snapshot, so a pacing cat
  doesn't spam the log or the disk.

## [0.9.0] — 2026-06-27

### Added
- **Live detection feed.** A new "Live detection" card streams a real-time view
  of what the detector sees — the camera frame with boxes drawn around any person
  (green) or cat (orange) as they're recognised — instead of only the per-event
  snapshot thumbnails (which stay, as history). New `GET /api/stream` serves an
  MJPEG (`multipart/x-mixed-replace`) feed the browser renders directly in an
  `<img>`; a "Show live feed" toggle turns it off on slow connections.

### Notes
- The feed **reuses the detection loop's single camera capture** — no second
  stream or extra decode. Frames are JPEG-encoded only while a browser is
  watching (capped ~10 fps), so an unwatched feed costs nothing. It's live only
  while watching (that's when there's recognition to show); stopped shows the
  still preview.
- Update rate is bounded by your **scan rate** (the loop reads that often), so on
  a low `scan_fps` the feed is choppy by design — it shows exactly the frames the
  net actually analysed. Detection boxes expire ~1.5 s after their last refresh so
  a subject who has left doesn't leave a box hanging while the video keeps going.

## [0.8.0] — 2026-06-26

### Added
- **GPU / iGPU acceleration for the YOLO detector** via a new `accelerator`
  setting (Detection card dropdown). Options:
  - `cpu` (default) — OpenCV `cv2.dnn` on the CPU, as before.
  - `opencl` — same net with the `OPENCL_FP16` target so the conv layers run on
    an OpenCL device (e.g. an Intel iGPU). No extra Python dependency; OpenCV
    falls back to CPU on its own if there's no OpenCL device.
  - `openvino-gpu` / `openvino-auto` — run the ONNX through Intel's **OpenVINO**
    runtime (optional `openvino` package) on the `GPU` device, or `AUTO` (GPU
    with built-in CPU fallback). The dependable iGPU path — typically 2–4× CPU on
    Intel hardware, and what makes the heavier `yolo11m` practical.
  The YOLO backend now wraps either engine behind a small inference *runner*, so
  the letterbox + NMS decode is shared across all accelerators.
- Graceful degradation: if a requested GPU backend can't start (no Intel GPU, no
  driver, `openvino` not installed), the detector retries the **same** model on
  CPU before falling back to MobileNet-SSD — a dead accelerator never costs you
  the model.
- `openvino` added as an **optional** dependency (commented in `requirements.txt`;
  offered by `setup.sh` / `setup.ps1`). The core install stays lean.
- **`check_accelerator.py`** diagnostic — reports the compute devices this machine
  exposes, what your configured `accelerator` actually resolves to (a real GPU vs
  a silent CPU fallback), and a CPU-vs-backend ms/frame timing so you can confirm
  the offload is real. Run: `./venv/bin/python check_accelerator.py`.

### Notes
- **Intel-only** for the *GPU*, and it needs the host's Intel GPU compute drivers —
  on AMD/ARM NAS boxes the GPU options stay on CPU. The OpenVINO path was verified
  end-to-end on the CPU device (same detections as `cv2.dnn`); the **iGPU** speed-ups
  are from OpenVINO's published figures, **not yet run on real Intel iGPU hardware
  here** — confirm with `check_accelerator.py` on your box.
- **Bonus measured on a CPU-only box:** OpenVINO's *CPU* runtime alone ran yolo11n
  ~3× and yolo11m ~3× faster than OpenCV's `cv2.dnn` CPU path (yolo11m 465 ms → 150 ms),
  no GPU involved — so `openvino-auto` is a free win even without an iGPU, and it's
  what makes yolo11m practical. (Numbers are from this dev box; relative, not absolute.)

## [0.7.0] — 2026-06-26

### Added
- **Selectable YOLO11m (medium) detection model.** A second YOLO variant
  (`yolo11m`, ~77 MB, exported at 640×640) is now bundled and selectable from the
  Detection-model dropdown alongside the default `yolo11n` and `mobilenet_ssd`.
  The YOLO backend is now a small variant registry (`d20app/yolo.py` `MODELS`)
  mapping each variant to its ONNX file and fixed input size, so adding future
  models is a one-line change.

### Notes
- **Honest trade-off:** `yolo11m` is bigger and much heavier on CPU (~146 ms @320
  / ~500 ms @640, roughly 5–18× nano) and on our own night/day benchmark it did
  **not** beat nano on the night case that motivated the upgrade (nano @320 ~0.865
  vs medium @640 ~0.914 on the night frame — but nano already clears the bar). So
  `yolo11n` stays the **default**; medium is there for users with CPU headroom who
  want the extra capacity on genuinely hard scenes. Flipping the default is a
  one-line change in `config.py` if real-world results warrant it.
- These CPU timings are from this dev box, not the target NAS — treat them as
  relative, not absolute.

## [0.6.0] — 2026-06-26

### Added
- **Local USB / built-in webcam support.** Use a camera plugged into the machine
  running the app — a "USB camera on this PC" picker (Detect button) lists them;
  internally it's stored as `usb:N` and opened by device index with the platform
  backend (DirectShow on Windows, V4L2 on Linux) instead of the FFmpeg/RTSP path.
- **Local PC speaker output.** Play the treat chime/speech on the host computer's
  own speakers via a "This PC (local audio)" entry in the speaker list — pick it,
  a Google Home, or both. Uses the optional **`playsound3`** package (offered by
  `setup.sh` / `setup.ps1`; a clear message tells you to install it if you pick
  local audio without it).

### Notes
- Playing audio out of an IP **camera's own speaker** (ONVIF two-way "backchannel")
  is intentionally not included — it's non-standard and camera-specific. Left as a
  future idea.

## [0.5.3] — 2026-06-26

### Added
- **`start.bat`** — a one-click Windows launcher to run the app after setup
  (checks the venv exists, runs `run.py`, keeps the window open on stop).

## [0.5.2] — 2026-06-26

### Added
- **Windows setup can install Python for you.** If `setup.ps1` doesn't find
  Python 3.11+, it offers to install it per-user (no admin) — via `winget` if
  available, else by downloading the official python.org installer and running it
  silently — then refreshes the session PATH and continues (mirrors how
  `setup.sh` offers the apt install on Debian).

## [0.5.1] — 2026-06-26

### Added
- **Windows setup.** `setup.ps1` (and a double-clickable `setup.bat` wrapper)
  mirror `setup.sh`: find Python 3.11+, create the venv, install deps, generate
  the chime, and create `config.yaml`. README gains a "Run on Windows" note
  (firewall allow-prompt, Task Scheduler for autostart). The app code was already
  cross-platform; only the bash installer was Linux-only.

## [0.5.0] — 2026-06-26

### Added
- **YOLO11n detection model (new default).** A real dim night frame scored
  **0.00** with MobileNet-SSD (person completely missed) but **~0.87** with
  YOLO11n, for only ~1.4× the CPU (≈28 ms vs ≈20 ms per inference on a test box,
  and the net only runs on motion frames). YOLO11n is far better in low light and
  on occluded/odd poses. Choose the model in the GUI (**Detection model**) or via
  `detector_model` in `config.yaml` (`yolo11n` | `mobilenet_ssd`).
  - Runs through OpenCV `cv2.dnn` from a bundled `d20app/models/yolo11n.onnx`
    (~10 MB) — **no PyTorch at runtime**; export tooling is offline-only.
  - New backend in `d20app/yolo.py` (letterbox → decode → NMS, COCO-80) produces
    the same box format as the SSD path, so person triggers, `cat` labelling, and
    annotated snapshots are unchanged.
  - **Graceful fallback:** if the ONNX can't be loaded, the detector logs a
    warning and silently uses MobileNet-SSD.

### Notes
- YOLO is much better at people but, like any strong detector, can still
  occasionally misread an unusual cat pose (e.g. two cats seen top-down) as a
  low-confidence person; the `confirm_frames` gate remains the backstop. The
  MobileNet-SSD cat regression suite is retained (pinned to that model).

## [0.4.0] — 2026-06-25

### Added
- **Motion sensitivity control + advanced motion tuning.** A Low/Medium/High
  preset (with a Custom mode) drives the motion pre-filter so the camera stops
  firing on shadows, lighting changes, or a swaying plant. The raw knobs
  (min change-area, brightness threshold, min blob size) are exposed under
  "Advanced motion tuning" for fine control.
- **Configurable "Notify threshold" (`label_floor`, default raised 0.3 → 0.55).**
  Only confident non-person detections get named in the log/snapshots, so stray
  "pottedplant"/"sofa" guesses no longer clutter the activity log — including at
  the higher detect-size, where a real kitchen frame put a plant at ~0.50. (This
  never affected treats — only a person triggers one — just the labels you saw.)
- **Pause detection during cooldown (on by default).** After a roll, the neural
  net is skipped for the cooldown window (nothing it sees can trigger anyway) —
  a large CPU saving on a NAS — and resumes automatically a few seconds before
  the window reopens, so the next treat is never missed. The camera keeps being
  read so a dropout is still noticed.
- **Saved cameras.** Manually-added cameras (name, URL, username, password) save
  to a dropdown so you can switch between feeds in one click. New endpoints
  `GET/POST /api/cameras/saved`, `…/select`, `…/delete`; passwords are stored
  locally in `config.yaml` (plaintext, same as before) and never sent back to
  the browser.
- **Every setting now has a plain-language note** on its effect on motion
  detection, image-analysis quality, and CPU usage.
- **"Keep speaker connection warm" toggle (off by default).** Optionally loops a
  silent clip every couple of minutes so the Google Home's Cast receiver never
  unloads — then a treat just swaps the audio instead of relaunching the
  receiver, which is what actually removes the "connecting" chime. (Research
  confirmed a held socket alone can't: the receiver tears down ~5 min after
  playback regardless, so only re-asserted audio keeps it loaded.) It yields to
  any other audio so it won't stomp on music, and "don't interrupt playback"
  still distinguishes real media from our own silence. Trade-off: it holds the
  speaker active, so leave it off if you use those speakers for music.

### Changed
- The settings page gained a "5. Motion & CPU" section; Quiet time and Region of
  interest renumber to 6 and 7.

## [0.3.6] — 2026-06-25

### Fixed
- **Treat-cast crash (`name 'speakers_label' is not defined`).** The cast path in
  the detection loop referenced `targets`/`speakers_label` from `_run`'s scope
  while running inside the separate `_loop_body` method, so **every won roll
  crashed the loop**. The cast handling is now a `_cast_for_treat` method that
  takes its speaker arguments explicitly. (Introduced with the multi-speaker work
  in 0.3.0; the 0.3.2 casting revert only touched `caster.py`, so this lived on.)

### Added
- **Persistent speaker connections are back (no "connecting" chime).** The
  `Caster` again caches each speaker's Cast connection and reuses it across
  treats, so only the first cast pays the discover/connect cost. Hardened over the
  original attempt: a cached connection is health-checked before use, and a play
  that fails on a silently-dead socket is dropped and **retried once** on a fresh
  connection before the speaker is reported failed. Held connections are released
  when watching stops (`Caster.close()` from the loop's shutdown). Multiple
  speakers and spoken messages keep working.

## [0.3.5] — 2026-06-25

### Changed
- **Stricter detection defaults to keep cats from ever earning a treat.**
  `person_confidence` 0.4 → **0.5** and `confirm_frames` 3 → **4**. Video testing
  (people, cats, and person+cat clips replayed through the full pipeline) showed
  two things: on still frames 0.5 cleanly separates cats (worst 0.474) from people
  (all ≥ 0.71), but a cat *in motion* can briefly spike much higher — a sprawled
  cat hit person=0.93 for a frame or two. A single high frame is therefore not
  safe to trust, so requiring **4 consecutive** person frames (up from 3) is the
  real guard; the nearest a cat came was 2 in a row. People sustain easily (a
  walking person held 100+ frames), so the extra frame costs ~0.1 s of latency
  and no missed detections. Existing configs are untouched; this only moves the
  defaults for new installs (`d20app/config.py`, `config.example.yaml`).

## [0.3.4] — 2026-06-25

### Reverted
- **Cat-overlap person suppression (from 0.3.3) is removed.** Suppressing a
  low-confidence `person` box that an animal box covers is indistinguishable
  from a person *carrying* a cat, so it risked missing a real person — the one
  failure this app can't tolerate. We now accept that a dense pile of cats may
  occasionally trigger a (harmless) treat-roll rather than ever drop a person.
  The broadened 45-image cat set stays; the multi-cat test now pins the small
  set of tolerated cluster misreads (`tests/test_detection_accuracy.py`,
  `KNOWN_CLUSTER_MISREADS`) so the rate can't grow unnoticed.

## [0.3.3] — 2026-06-25

### Fixed
- **Cat clusters no longer misread as a person** — a group of cats (several
  eating from one bowl, two entangled cats) could make the model emit a weak,
  low-confidence `person` box over the pile. The detector now suppresses a
  `person` box that scores below a trust threshold when an animal detection
  (`cat`/`dog`/`bird`/…) covers it, so cat scenes never trigger a treat. A
  confident person box — e.g. someone *holding* a cat — is always believed, so
  this costs no real-person detections (all people fixtures score ≥ 0.71 and are
  unaffected).

### Changed
- **Broadened the cat regression set from 5 to 45 images** — added ~30 varied
  single cats (breeds, indoor/outdoor, day/night, near/far) and a new
  `tests/fixtures/cats_multi/` of 10 multi-cat scenes, all from Wikimedia Commons
  (credited in `tests/fixtures/cats/CREDITS.md`). New tests assert **0** false
  human flags across the whole set at both 300px and 512px, and a lenient floor
  that the model still recognises cats as cats (so a future model swap can't go
  silently blind to them).

## [0.3.2] — 2026-06-24

### Fixed
- **Casting crash** — reverted the persistent Cast connections from 0.3.0; the
  connection-reuse path could crash. Casts now reconnect each time (reliable).
  Multiple speakers and spoken messages are kept. (The no-reconnect-chime goal
  moves back to the roadmap, to be redone safely.)
- **Camera decode artifacts no longer trigger motion** — the motion filter now
  uses a median blur + morphological opening + a solid-blob (minimum-thickness)
  check, so a thin line of corrupt pixels is ignored instead of firing motion.
  Detection boxes are drawn only at the trigger threshold, so corrupt frames no
  longer litter snapshots with low-confidence boxes.
- **Human-detection regression** — reverted the default detection input size to
  **300** (measured 99–100% person recall vs 98.8% at 512) and lowered the
  default `person_confidence` to **0.4** for margin on hard poses. Verified
  people in hats/helmets/headgear detect at 0.88–1.00 and back-turned
  pedestrians at ~99%, while cats still never trigger. 512 stays selectable for
  distant cats.

## [0.3.1] — 2026-06-24

### Added
- **App version shown in the GUI footer** (and at `GET /api/version`, and printed
  on startup) so it's easy to confirm which build is running when troubleshooting.

## [0.3.0] — 2026-06-24

### Added
- **Multiple speaker output** — pick several Cast devices and the treat plays on
  all of them at once.
- **Optional spoken message** — say something (e.g. "Give the cat a treat!")
  instead of a sound, synthesized with gTTS and cached.
- **Quiet time** — silence chimes during a daily window (wraps past midnight).
- **Region-of-interest picker** — grab a still in the GUI and drag a box to watch
  only part of the view.
- **Detection detail** (net input size) and **scan rate** (fps) controls in the GUI.
- **Annotated snapshots** on every detection — boxes around the person/cat shown
  as clickable thumbnails in the activity log.
- **Live odds readout** — "For those who are mathematically challenged: X%".
- **`ROADMAP.md`** (feature list + roadmap) and this **changelog**.

### Changed
- **Persistent Cast connections** — held open between treats, so there's no
  "connecting" chime or delay; stale connections rebuild automatically.
- Raised the default detection input to **512px** so a cat across the room is
  detected (300px missed it).

### Fixed
- **False positives with no real motion** — frames are Gaussian-blurred so sensor
  noise / a ticking timestamp overlay no longer count as motion, the first frame
  reports no motion, and a person must persist across *N* consecutive frames
  before anything fires.
- Camera **username/password fields overlapping** on narrow/mobile screens.

## [0.2.0] — 2026-06-23

### Added
- **Persistent activity log** in the GUI (survives restarts), colour-coded.
- **Non-human motion logging** — reports what moved (e.g. "cat moved").
- **Camera diagnostics** — `check_camera.py` and a "Camera connected (W×H)"
  heartbeat so a running-but-idle loop isn't silent.
- README **screenshot**, plus stopping/troubleshooting and non-root **systemd**
  documentation.

### Changed
- Open RTSP streams via **FFmpeg over TCP** (authenticates like VLC), with a
  fast-fail connect timeout and quieter decoder logging.

### Fixed
- **Detection model was non-functional** — the bundled weights were a training
  snapshot that didn't match the prototxt, so every detection scored 0 and no
  person was ever detected. Replaced with the matching *deploy* weights
  (~99% recall on 170 real images); added a regression test to guard it.
- **"401 Unauthorized"** when a stream worked in VLC — force the FFmpeg backend
  and percent-encode injected credentials.
- A failing camera **flooding the console** — back off and log the problem once.
- **Camera password leaked** in the "Started watching" log line — now masked.

## [0.1.0] — 2026-06-22

### Added
- Initial release: watch an IP camera, "roll a die" when a **person** enters, and
  cast a celebratory chime to a **Google Home / Nest** speaker on a winning roll —
  while **ignoring the cats**.
- Single-page **web GUI**; **ONVIF** camera and **Google Cast** speaker
  auto-discovery; custom sound upload.
- One-shot **`setup.sh`** (virtualenv, dependencies, bundled detection model,
  config) with a **Python 3.11+** guard and an optional `apt` install of
  `python3-venv` / `pip`.

[Unreleased]: https://github.com/vcons002-ship-it/Kevin-s-Cat-App/compare/main...HEAD
