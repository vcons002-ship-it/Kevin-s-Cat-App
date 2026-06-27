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
  live feed of the camera that saw it.
- **Annotated snapshots** on every detection — boxes around the person/cat,
  shown as clickable thumbnails — the fastest way to debug false positives.
- **"Camera connected (W×H)"** heartbeat so a running-but-idle loop isn't silent.

### App & ops
- **Single-page web GUI** — everything is point-and-click.
- **One-shot `setup.sh`** (venv, deps, model, config) with a Python 3.11+ guard
  and an optional `apt` install of `python3-venv`/`pip`.
- **systemd** autostart instructions for OpenMediaVault.
- **No Docker, no Frigate, no cloud.**
- **118 automated tests**, including a detection-accuracy regression guard over
  45 cat images (incl. multi-cat scenes), a treat-cast regression guard, the
  YOLO11 backend (nano + medium variants, CPU/OpenCL/OpenVINO accelerators with
  CPU fallback), the live MJPEG feed (frame publish + box-TTL + stream route) and
  the smooth-feed capture thread (toggle reconcile, version gating, error
  surfacing, watchdog respawn, camera-death detection), cat-sighting tracking
  (region labels, store persistence,
  `/api/cats`), local USB camera + local PC speaker routing, and
  saved-camera/cooldown-pause/keep-warm coverage.

---

## 🗺️ Roadmap / ideas

### Detection & accuracy
- [ ] **Multi-camera** — watch several feeds at once; "Show cat" then switches
      the live feed to whichever camera saw the cat (sightings already store the
      camera, so this is the next step on top of 0.10.0's tracking).
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
- [x] **Saved cameras** — add several (with credentials) and switch the active
      feed from a dropdown. (Watching *several at once* is still future work.)
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
