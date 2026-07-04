# 🎲🐱 Kevin's Cat App — Features & Roadmap

A snapshot of what the app does today and where it could go next. The roadmap is
a list of **ideas, not commitments** — suggestions and PRs welcome.

---

## ✅ Implemented

### Detection
- **Person detection** on CPU via **YOLO11** (OpenCV `cv2.dnn`, ONNX) — no GPU, no
  cloud, no extra services. Detects every clear person in the bundled fixtures.
  (MobileNet-SSD was the original backend; removed in 0.25.0 — it lost every
  benchmark to YOLO. No silent fallback: a model that can't load raises a clear error.)
- **Cats ignored** — `person` and `cat` are separate classes; only people roll.
- **Model = resolution** — a YOLO ONNX is a fixed-shape export, so the model name
  carries its input size (`yolo11n` 320 → `yolo11m` 640 → `yolo11m_960` 960); pick a
  bigger model for distant subjects at more CPU.
- **Motion pre-filter** — median blur + morphological opening + a solid-blob
  check, so sensor noise, compression grain, a ticking timestamp overlay, and
  **thin decode-artifact lines** don't register as motion.
- **False-positive guard** — a person must persist across *N* consecutive frames
  (`confirm_frames`) before anything fires.
- **Region of interest** — draw a box in the GUI to watch only part of the view.
- **Non-human motion logging** — names the mover (e.g. "*cat moved*").

### Camera
- **RTSP / HTTP** streams opened with FFmpeg over **TCP** (authenticates like
  VLC); credentials injected, **percent-encoded**, and **masked** in all logs.
- **ONVIF auto-discovery**, or manual URL entry.
- **Main vs sub feed** choice, plus a **scan-rate** control to balance CPU.
- **Fast-fail** connect timeout, **auto-reconnect** with back-off, and clear
  errors surfaced in the Activity log.
- **Live preview frame** + a standalone **`check_camera.py`** diagnostic.

### Speakers & output
- **Google Cast** — no account, cloud login, or API key.
- **Multiple speakers** at once (multi-select).
- **Speaker-group** detection with a GUI warning.
- **Custom sound upload** *or* an **optional spoken message** (gTTS).
- **"Don't interrupt if already playing"** toggle.

### Game rules
- Configurable **dice size**, **DC**, and **cooldown** between rolls.
- **Live odds** readout ("For those who are mathematically challenged: X%").
- **Quiet time** — silence chimes overnight (window may wrap past midnight).

### Activity log & observability
- **Persistent, file-backed** event log (survives restarts), colour-coded.
- **Live detection feed** — a real-time MJPEG view of what the detector sees,
  with person/cat boxes drawn as they're recognised (reuses the loop's capture).
- **Cat tracking & "Show cat"** — cats don't roll, but every sighting is logged
  (when, which camera, where in the frame) with a snapshot; a button pulls up the
  live feed of the camera that saw it. **Still/sleeping cats** are caught by a
  periodic forced scan (a motionless cat makes no motion to trip the net), on a
  GUI-set cadence (always → off); when several rooms have a cat, the feed
  **rotates** between them.
- **Annotated snapshots** on every detection — boxes around the person/cat,
  shown as clickable thumbnails — the fastest way to debug false positives.
- **"Camera connected (W×H)"** heartbeat so a running-but-idle loop isn't silent.

### App & ops
- **Single-page web GUI** — everything is point-and-click.
- **One-shot `setup.sh`** (venv, deps, model, config) with a Python 3.11+ guard
  and an optional `apt` install of `python3-venv`/`pip`.
- **systemd** autostart instructions for OpenMediaVault.
- **No Docker, no Frigate, no cloud.**
- **368 automated tests**, including a detection-accuracy regression guard over
  45 cat images (incl. multi-cat scenes), a treat-cast regression guard, the
  YOLO11 backend (nano + medium variants, CPU/OpenCL/OpenVINO/CUDA accelerators with
  CPU fallback, a clear error — no silent fallback — when a model can't load, the
  onnxruntime CUDA path's silent-CPU-fallback guard + decode parity with cv2.dnn),
  the live MJPEG feed (frame publish + box-TTL + stream route) and
  the smooth-feed capture thread (toggle reconcile, version gating, error
  surfacing, watchdog respawn, camera-death detection), multi-camera (per-camera
  specs/roles, role-gated rolling/tracking, one shared cooldown across cameras,
  failure isolation), cat-sighting tracking (region labels, store persistence,
  `/api/cats`), the still/sleeping-cat scan (forced no-motion detection,
  rising-edge de-dup, always-on/off, per-room present list, no-roll-on-still-scan
  guard), the "Test detection" tool (image adjustment maths, stateless
  `detect_image`, video frame sampling, upload/detect endpoints), the
  higher-resolution locator scan (tiling + NMS merge, larger-input fallback), the
  round-robin scheduler (rotation, rest/release, always-watch, viewed-pin), per-class
  cat confidence + locator classes (cat/dog), the benchmark sweep (matrix,
  self-contained HTML, XLSX present/absent), local USB camera + local PC speaker
  routing, and saved-camera/cooldown-pause/keep-warm coverage.

---

## 🗺️ Roadmap / ideas

### Detection & accuracy
- [x] **Multi-camera** — watch several feeds at once, each with its own role
      (rolls / tracks cats) and its own detection settings; "Show cat" switches the
      live feed to whichever camera saw the cat (0.13.0). One shared treat dispenser.
- [x] **Round-robin scheduling** — optional CPU cap: only *N* cameras detect at a
      time, rotating sets on an interval, so many cameras cost ~a few; resting cameras
      release their capture. Per-camera "always watch" exempts one; the live-viewed
      camera never rests (0.16.0). Trade-off: slower reaction on resting cameras.
- [ ] **GPU-batched inference** — on the iGPU/OpenVINO path, batch same-model
      cameras' frames into one `forward()` (needs a dynamic-batch ONNX re-export);
      GPU-only win. (Recorded from the 0.13.0 plan.)
- [x] **Test detection on a photo/video + image adjustments** — upload a still or
      a sampled-frame video clip and run the detector on it in the GUI, with the
      impactful settings (model, input size, confidence, notify floor, **gamma /
      brightness / contrast / saturation**, **tiling/overlap/accelerator** + an
      inference-time readout) as live controls; save what works to a camera or the
      global defaults (0.15.0 / 0.16.0). Adjustments apply before the net.
- [x] **Higher-resolution locator scan** — the still-cat scan runs at higher
      effective resolution (**tiling**, default 4×4, and/or a **larger input** like
      the bundled `yolo11m_960`) so a small/distant cat clears the net, while the
      fast treat path stays at the native size (0.16.0, issue #17).
- [x] **Independent cat / person confidence** — the cat (locator) path has its own
      `cat_confidence`, separate from `person_confidence` and `label_floor`, tunable
      per camera and in the test tool (0.17.0, #19).
- [x] **"Animal present" locator classes** — a no-dog household can count a **dog** as
      the cat (`locator_classes: ["cat","dog"]`), since the model often mislabels a cat
      as a dog and resolution raises "dog", not "cat" (0.17.0, #22).
- [x] **Resolution in the model dropdown** — removed the "Net input size" control; the
      model name carries the resolution (YOLO size is its export) (0.17.0, #20). With
      MobileNet-SSD's removal (0.25.0, #57) every model is a YOLO variant.
- [x] **"Benchmark this image"** — one-click sweep of models × tiling on an uploaded
      frame, emitting a shareable **self-contained HTML** report (+ optional XLSX) with
      best cat / dog / combined score, found?, and inference time per run (0.18.0, #21).
      Report polish (0.18.1): legible click-to-enlarge thumbnails, the **full-res
      original frame embedded** for reproducibility, human-readable download filenames,
      and a clearly-disabled (not dead) XLSX button when `openpyxl` is absent
      (#25/#26/#27/#28). Report fixes (0.18.2): in-page lightbox so thumbnails enlarge
      instead of hitting the browser's `data:`-URL navigation block, and SSD labels no
      longer show the size twice (#30/#31).
- [x] **Batch benchmark + cross-image summary** — sweep many images at once and get
      one summary ranking each config by **detection rate across all frames**, with
      click-to-expand **miss traceability** (which frame, what score, link to its
      report), a **config×image heatmap**, and optional **empty-room controls** that
      turn the summary into a false-positive check (0.19.0, #32). Answers "which config
      should the locator use?" inside the app. Portability + scale (0.20.0): a
      **"Download all"** zip with slug-named files whose links resolve when hosted, a
      **soft cap with cost warning** + an **abort** button, slug-based summary links,
      legible dark-theme link colours, and **video → "benchmark all frames"** (extract
      ≈1 fps, run the whole clip through the summary) (#35/#36/#37/#38).
- [x] **Cat-presence VLM tester** — a moondream "is there a cat?" panel that *reasons*
      about the whole frame (good on hard frames/decoys the box detectors miss/false-fire
      on); single editable-prompt `query` pass, raw reasoning + best-effort yes/no +
      load/query latency split, plus a **batch** tester scoring recall + FP separately
      (0.23.0/0.24.0, #48/#54). **Validated on an 8 GB RTX 3070** (0.27.0, #59): the working
      local load params (`max_batch_size=4`, `kv_cache_pages=2048` — the auto cache OOMs),
      an **API-key GUI field** (stored, masked, never logged), and a **local M2 / cloud M3
      mode selector** (cloud sends images off-device; M3 is cloud-only on 8 GB). Friendly
      OOM/KV-cache errors, and a **multi-pass majority vote** (default 3 passes; the vote
      ratio is the honest confidence and non-unanimous frames are flagged for review) since
      moondream's yes/no wobbles run-to-run (0.28.0, #60). Optional dep, model runs on the
      NAS.
- [x] **The escalation ladder** — "zoom in and look again" for missed small/distant cats
      (0.29.0, #66): full-res crops around **motion blobs** (now retained with locations),
      the **last sighting**, and a **CPU-predicted next position** → YOLO re-check (free) →
      moondream `detect` proposes + YOLO/voted-query **confirms** (a bare VLM region never
      records) → voted query as last resort. On-demand only (`/api/vlm/escalate` + GUI;
      live cameras behind the off-by-default `vlm_escalation` toggle); the treat path is
      untouched. Sightings carry a `source` tag so real use shows which rung earns its
      keep. NAS still to verify: real `detect()` quality + VRAM co-residency.
- [x] **The "cat trail"** (0.30.0, #67): diff against the last **null** (no-motion)
      frame so the whole cat silhouette lights up; coloured by recency (blue = entered
      → red = latest) for a path+timing visual; **trail-endpoint targeting** (interior
      endpoint = cat didn't leave → the ladder's strongest hint) and the honest
      **"probable location"** state when detection fails but the trail ends in-room
      (never recorded as a sighting). `GET /api/trail` + 🌈 Show trail in the GUI.
      NAS to verify: real-scene ghosting/lighting-drift behaviour + the basket-cat
      recovery end-to-end. v2 ideas: doorway zones for the exit check (below),
      per-sighting trail images in the report card.
- [x] **Sighting heat maps + semantic zones + time-of-day prior** (0.31.0, #68):
      per-camera density maps from the existing `cats.log` boxes (🔥 Heat map);
      GUI-drawn named zones ("the couch", doorways) so sightings carry a semantic
      spot and exit zones sharpen the trail's left-the-view check; `by_hour` /
      `likely_cameras` rank rooms by historical presence around the current hour
      (a prior, never a tracked state). NAS to verify: zone-draw UX, heat-map
      readability on real scenes, prior usefulness after a few days of data.
- [x] **Temporal VLM analysis — frame mosaic** (0.32.0, #68): each camera keeps a
      small ring of recent downscaled frames (8 × ~1 s apart, ≤480 px);
      `POST /api/vlm/temporal` tiles them into one numbered, age-labelled grid and
      asks moondream a single "did a cat pass through?" voted query (⏱️ button on
      the Test tool's video uploads and on the escalation camera row). NAS to
      verify the core premise: whether moondream can actually reason over a grid
      of frames — the prompt and plumbing are tested, the model's skill is not.
- [x] **Temporal score fusion / track-before-detect** (0.37.0): weak locator
      hits (below `cat_confidence`, decoded in the same forward pass) chain
      across frames by overlap and confirm as ONE sighting (source `track`)
      when the chain is ≥4 hits in 5 s AND net-travels ≥3% of the frame
      diagonal — the movement requirement is the decoy guard (#69/#70: decoys
      don't travel). The recall-raising mirror of `confirm_frames`. Pure YOLO
      evidence (0.33.0 rules); per-camera toggle, on by default. NAS to
      verify: real-footage hit rate and that compression shimmer doesn't fake
      chains (the smooth-chaining + travel checks are the guards).
- [ ] Temporal follow-ups: per-frame voted queries aggregated over the ring;
      per-sighting event clips from the same buffer; true video-LLMs
      (5090-tier experiment) only if the mosaic underperforms on the NAS.
- [ ] **Ideas parked with verdicts**: `caption()` report-card lines; `point()` as a cheap
      presence probe; motion-adaptive tiling; scheduled "cat census" ladder sweeps;
      cat re-ID embeddings (5090 benchmark before judging); **BLE collar tag** = the
      pragmatic per-cat identity (Wi-Fi CSI sensing assessed: people-tuned, new
      hardware — experiment-only, not an app feature).
- [ ] ~~**Super-resolution before detection**~~ — **tested and rejected** (#69,
      maintainer benchmark on the 5090): Real-ESRGAN before an unmodified YOLO
      *reduced* recall 6–8 points at every tiling config and cost ~40× more time.
      Mechanism: SR beautifies for humans and strips the sensor/compression
      texture the detector's features key on — the output is out of YOLO's
      training distribution, and the more SR changed the pixels the more recall
      fell. The narrow escape hatch (fine-tuning a detector *on* SR'd imagery) is
      a much larger project nobody is signing up for. Generative SR (SUPIR/InvSR)
      untested but predicted worse (hallucinated cat texture = false positives).
      Do not re-propose SR-before-detection; the data is in the issue.

#### Future improvements — assessed 2026-07-02 (after #69's SR + VLM decoy data)

The filter that separates good ideas from bad here, learned from #69: a
transformation that **recovers real signal** (more samples, more photons, more
pixels on the cat) can help; one that **synthesizes or beautifies** pushes the
image out of the detector's training distribution and hurts.

**Image-side:**
- [x] **Still-scan frame averaging** (0.34.0): the still-cat scan averages a
      short burst of back-to-back frames (default 3, max 8) before the locator
      runs — sensor noise drops ~√N on a still scene, and a sleeping cat is
      exactly the still subject that benefits. Any movement mid-burst falls back
      to the single sharp frame; the fast treat path never averages. Per-camera
      **Scan frames** knob. NAS to verify: real night-frame recall gain.
- [ ] **CLAHE / adaptive contrast for dark frames only** — apply conditionally
      (mean luminance below a threshold), never globally: on a dark frame it
      mostly restores what the sensor compressed; on a lit frame it's pure
      distribution shift. Add as a Test-tool knob first, measure on night
      frames, only then consider wiring it into the locator path.
- [ ] **Fine-tune YOLO on our own cameras** — the highest-leverage item on this
      list; see the step-by-step guide below.
- Ruled out by the #69 filter: sharpening/unsharp masking, saturation boosts,
  learned denoisers, and (already tested) SR — all synthesize or beautify.

**Equipment (no-constraints list, ranked by leverage per dollar):**
- [ ] **Thermal sensor node per key room** (~$50 MLX90640 32×24 on an ESP32;
      ~$200 FLIR Lepton 160×120): a cat is a ~38 °C blob against a ~21 °C room —
      visible in total darkness, immune to decoys (a plush cat is room
      temperature; thermal is structurally immune to the exact failure mode
      that disqualified the VLMs), and a *still, sleeping* cat radiates heat
      continuously, closing the no-motion gap at the sensor level. Integration:
      "warm blob at region X" enters the app as a **hint box** — the escalation
      ladder already accepts hints from any source. Best first buy.
- [ ] **More light / better night optics** — a warm nightlight per cat room is
      the $10 version (more photons beat every algorithm); "full-color night
      vision" cameras (large sensor, ~f/1.0, ColorVu-class, ~$100–200) are the
      upgrade: colour night frames stay close to YOLO's training distribution
      where grayscale IR is marginal.
- [ ] **BLE collar beacon + room receivers** (~$60 total, espresense-style;
      already parked above) — the only option that gives **per-cat identity**,
      which vision provably can't (#65). Enters as a prior/hint (rank the
      sweep, aim the crops), never a state machine.
- [ ] **mmWave presence sensors** (~$40–80/room, Aqara FP2 / LD2450-class) —
      detect micro-motion incl. breathing of a stationary body; famously
      "false-trigger" on pets, which here is the feature. Another independent,
      dark-proof hint channel.
- [ ] **Doorway break-beam sensors** (~$15/doorway, IR beam at cat height) —
      deterministic "something crossed at time T": the ground-truth version of
      the exit-zone check the trail approximates from pixels. Timestamped
      hints only ("last crossing was *into* the study") — #65's rejection of
      transition state machines applies to the modelling, not the sensor.
- [ ] **NAS GPU upgrade** (used RTX 3090 24 GB, ~$700) — dissolves the
      moondream + CUDA-YOLO VRAM co-residency question on the 8 GB 3070 and
      enables an always-warm VLM for the caption/report-card role. Nice-to-have,
      not a detection win.
- **Assessed and skipped**: PTZ cameras (a moving camera breaks the null-frame
  trail, the zones, and the heat maps — everything assumes fixed POV); Wi-Fi
  CSI sensing (people-tuned, new firmware, cats are marginal RF targets);
  microphones/audio (a weak signal thermal delivers better).
- The pattern: every good option is another **independent hint channel**
  feeding the same only-YOLO-confirms pipeline (0.33.0) — thermal blobs, BLE
  presence, mmWave, beam crossings slot in exactly where motion blobs and
  trail endpoints already do.

#### How to fine-tune YOLO on our own cats (the guide)

The single biggest remaining win: it attacks the residual miss rate *and* decoy
false positives directly, and it makes our exact cameras/lighting/cats the
training distribution by construction. Everything runs on the 5090; the output
is one ONNX file the app already knows how to load.

1. **Assemble the dataset** (the hard 20%, do it well):
   - Start from what exists: the 199-positive/43-null benchmark set, the decoy
     set, and the sightings log — `cats.log` + `snapshots/` is a self-labelling
     pipeline (every YOLO-confirmed sighting has a box already; export as
     starter labels and hand-verify a sample).
   - Bootstrap the rest with the current model: run yolo26x at 3×3 tiling over
     collected frames, take its boxes as **draft labels**, and hand-correct in
     Label Studio or CVAT (free, local). Correcting drafts is ~5× faster than
     labelling from scratch. Aim for 500–1500 boxed cat instances to start —
     small for pretraining, plenty for fine-tuning.
   - **Deliberately over-sample the failure modes**: night/dim frames, the
     curled-up-in-basket poses, partial occlusions, each camera's worst angle.
     A model learns what it sees; the misses are the syllabus.
   - **Hard negatives are half the value**: add the decoy set and the 43 nulls
     as *background images* (present in training with no labels) — this is the
     standard YOLO recipe for "stop firing on the cat-shaped cushion" and it
     directly targets the FP problem.
2. **Split honestly**: hold out ~15% for validation, split by **scene/day, not
   by frame** (two frames of the same nap in train and val is leakage that
   inflates every metric). Keep at least one full camera out if possible.
3. **Choose the class recipe**: fine-tune as a **single-class cat locator**
   (collapse cat/dog into one "cat" class) used *only* for the locator path —
   the treat path keeps the stock person model, so person detection can't
   regress. Don't fine-tune a shared person+cat model with cat-only data: it
   will quietly forget people.
4. **Train** (ultralytics on the 5090, minutes-to-hours):
   ```
   yolo detect train model=yolo26x.pt data=cats.yaml imgsz=1280 epochs=100 \
        batch=8 lr0=0.001 freeze=10 patience=20 close_mosaic=10
   ```
   - `freeze=10` (backbone) + low `lr0`: with a small dataset you're steering,
     not re-learning; this avoids catastrophic forgetting.
   - `imgsz=1280`: matches the tiled-crop resolution the locator actually sees.
   - Keep augmentation mild: default mosaic early (`close_mosaic` turns it off
     for the final epochs), light HSV; **no heavy colour jitter** (it fights
     the night-frame distribution you're trying to learn).
   - `patience=20`: early-stop on the val set; small datasets overfit fast.
5. **Export for the app**: `yolo export model=best.pt format=onnx imgsz=1280
   opset=12` → drop into `models/` (e.g. `cat26x_1280.onnx`), add a registry
   entry in `d20app/yolo.py` — the model-as-resolution convention holds (the
   name carries the input size).
6. **Gate on the benchmark, not vibes**: run the in-app batch benchmark on the
   *held-out* val set + the decoy set, against stock yolo26x at the same
   tiling. Ship only if it beats 91% recall / 0% FP on data it never saw. Check
   the night-frame slice separately — that's the gap it was built to close.
7. **Iterate on misses**: every future miss (frames the still-scan/ladder had
   to escalate on, or "probable" boxes a human confirmed) goes into the next
   training round — the sightings log makes hard-example mining free. Expect
   round 2 to matter more than round 1.
   - Optional later: once a fine-tuned locator exists, the #69 door reopens a
     crack — averaged/CLAHE'd night frames *in the training set* make those
     transforms in-distribution too. Measure, as always.

- [ ] Multiple / per-zone regions of interest.
- [x] **Selectable YOLO11 model size** — `yolo11n` (default) or the bigger
      `yolo11m` for users with CPU headroom (0.7.0). Medium didn't beat nano on
      our night test, so nano stays the default.
- [x] **YOLO26 models** — `yolo26m` (bundled) and `yolo26x` (export-only, ~213 MB),
      same COCO lineage as YOLO11, tuned for small objects; in the dropdown + benchmark
      sweep (0.22.0, #45). Exported with the raw head (`end2end=False`) since YOLO26's
      NMS-free export isn't `cv2.dnn`-decodable. The `26m`-vs-`11m` head-to-head + the
      `26x` escalation question need the NAS iGPU — run them through the benchmark.
- [x] **GPU / Intel iGPU acceleration** for YOLO — an `accelerator` setting with
      OpenCL and Intel **OpenVINO** (GPU/AUTO) backends, falling back to CPU
      (0.8.0). Frees the CPU and makes `yolo11m` practical on Intel hardware.
- [x] **NVIDIA GPU acceleration** — an `onnx-cuda` accelerator that runs the YOLO ONNX
      through **onnxruntime-gpu** (CUDAExecutionProvider). Measured **~37× faster** on
      an RTX 3070 (~23 ms vs ~485–855 ms CPU), which makes the heavyweight `yolo26x`
      runnable continuously (0.26.0, #58). Guards the silent-CPU-fallback trap: prepends
      torch's CUDA libs to `LD_LIBRARY_PATH` and errors loudly if it lands on CPU.
      TensorRT EP left as a future "max speed" tier.
- [ ] Optional **Coral TPU** for hardware-accelerated inference and better
      small-object / low-light accuracy at low CPU.
- [ ] Day/night profiles (different confidence or ROI by time of day).
- [ ] "Trigger on entry only" tracking (ignore someone who lingers).

### Speakers & output
- [x] **No "connecting" chime** — held Cast connections (0.3.6) cut re-discovery,
      but a Google Home still relaunches its receiver after ~5 min idle, so 0.4.0
      adds an optional **"keep speaker warm"** toggle that loops a silent clip to
      keep the receiver loaded (the only thing that actually suppresses the chime).
- [x] **Local PC speaker** — play the treat sound on the host machine's own
      speakers (alongside or instead of a Google Home), via optional `playsound3`.
- [ ] **Play out of an IP camera's own speaker** (ONVIF two-way backchannel) —
      non-standard and camera-specific, so only if a clean approach emerges.
- [ ] **Per-speaker volume**, and a fixed "treat volume" that restores after.
- [ ] **Preset spoken phrases** / a random message from a list.
- [ ] TTS **voice/language** options and an **offline** fallback (e.g. pyttsx3).
- [ ] Play a **chime *and*** a spoken message.

### Notifications & history
- [ ] **Daily/weekly treat-count summary** (in-log, email, or push).
- [ ] **Filterable + downloadable** activity log (CSV).
- [ ] A **snapshot gallery** view.

### Camera
- [x] **Saved cameras** — add several (with credentials), each with its own role
      and detection settings, and **watch several at once** (multi-camera, 0.13.0).
- [x] **Local USB / built-in webcam** on the machine running the app.
- [x] **Live MJPEG feed** in the GUI — a real-time "Live detection" view with
      person/cat boxes drawn as they're recognised (0.9.0), reusing the loop's
      single capture. (A still-grab preview remains for the ROI picker.)
- [x] **Smooth live feed** — optional dedicated capture thread so the feed plays
      at camera rate instead of stuttering at the inference-gated scan rate
      (0.11.0), toggled by a checkbox.
- [ ] **Touch support** for the ROI picker on phones.

### App & ops
- [ ] Optional **GUI password / LAN auth**.
- [ ] **Config export/import** and backup.
- [ ] A **health endpoint** / basic metrics.
- [ ] An **optional Dockerfile** for those who prefer containers.

---

*See [`README.md`](README.md) for setup and usage, and
[`config.example.yaml`](config.example.yaml) for every setting.*
