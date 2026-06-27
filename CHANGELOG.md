# Changelog

All notable changes to **Kevin's Cat App** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/).
Version numbers below were assigned retroactively (the repo isn't tagged yet);
everything through the latest entry is on `main`.

## [Unreleased]

_Nothing yet — see [`ROADMAP.md`](ROADMAP.md) for what's planned._

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
