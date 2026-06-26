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
- **Annotated snapshots** on every detection — boxes around the person/cat,
  shown as clickable thumbnails — the fastest way to debug false positives.
- **"Camera connected (W×H)"** heartbeat so a running-but-idle loop isn't silent.

### App & ops
- **Single-page web GUI** — everything is point-and-click.
- **One-shot `setup.sh`** (venv, deps, model, config) with a Python 3.11+ guard
  and an optional `apt` install of `python3-venv`/`pip`.
- **systemd** autostart instructions for OpenMediaVault.
- **No Docker, no Frigate, no cloud.**
- **91 automated tests**, including a detection-accuracy regression guard over
  45 cat images (incl. multi-cat scenes), a treat-cast regression guard, the
  YOLO11n backend (loads, detects people, falls back), local USB camera + local
  PC speaker routing, and
  saved-camera/cooldown-pause/keep-warm coverage.

---

## 🗺️ Roadmap / ideas

### Detection & accuracy
- [ ] Multiple / per-zone regions of interest.
- [ ] Optional **Coral TPU** or a newer model for better small-object and
      low-light/night accuracy.
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
- [ ] **Live MJPEG preview** stream in the GUI (not just a grabbed still).
- [ ] **Touch support** for the ROI picker on phones.

### App & ops
- [ ] Optional **GUI password / LAN auth**.
- [ ] **Config export/import** and backup.
- [ ] A **health endpoint** / basic metrics.
- [ ] An **optional Dockerfile** for those who prefer containers.

---

*See [`README.md`](README.md) for setup and usage, and
[`config.example.yaml`](config.example.yaml) for every setting.*
