# 🎲🐱 Kevin's Cat App — Features & Roadmap

A snapshot of what the app does today and where it could go next. The roadmap is
a list of **ideas, not commitments** — suggestions and PRs welcome.

---

## ✅ Implemented

### Detection
- **Person detection** on CPU via MobileNet-SSD (OpenCV `cv2.dnn`) — no GPU, no
  cloud, no extra services. **~99% recall** on 170 real pedestrian images.
- **Cats ignored** — `person` and `cat` are separate classes; only people roll.
- **Configurable detail** — net input size (300 / 512 / 768); **300 default**
  for reliable person detection; 512 recovers distant cats.
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
- **183 automated tests**, including a detection-accuracy regression guard over
  45 cat images (incl. multi-cat scenes), a treat-cast regression guard, the
  YOLO11 backend (nano + medium variants, CPU/OpenCL/OpenVINO accelerators with
  CPU fallback), the live MJPEG feed (frame publish + box-TTL + stream route) and
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
- [x] **Resolution in the model dropdown** — removed the YOLO-no-op "Net input size"
      control; MobileNet sizes are named variants (`mobilenet_ssd@512`), YOLO size is
      its export (0.17.0, #20).
- [x] **"Benchmark this image"** — one-click sweep of models × tiling on an uploaded
      frame, emitting a shareable **self-contained HTML** report (+ optional XLSX) with
      best cat / dog / combined score, found?, and inference time per run (0.18.0, #21).
      Report polish (0.18.1): legible click-to-enlarge thumbnails, the **full-res
      original frame embedded** for reproducibility, human-readable download filenames,
      and a clearly-disabled (not dead) XLSX button when `openpyxl` is absent
      (#25/#26/#27/#28).
- [ ] Multiple / per-zone regions of interest.
- [x] **Selectable YOLO11 model size** — `yolo11n` (default) or the bigger
      `yolo11m` for users with CPU headroom (0.7.0). Medium didn't beat nano on
      our night test, so nano stays the default.
- [x] **GPU / Intel iGPU acceleration** for YOLO — an `accelerator` setting with
      OpenCL and Intel **OpenVINO** (GPU/AUTO) backends, falling back to CPU
      (0.8.0). Frees the CPU and makes `yolo11m` practical on Intel hardware.
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
