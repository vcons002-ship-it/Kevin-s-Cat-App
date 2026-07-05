# Changelog

All notable changes to **Kevin's Cat App** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/).
Version numbers below were assigned retroactively (the repo isn't tagged yet);
everything through the latest entry is on `main`.

## [Unreleased]

_Nothing yet — see [`ROADMAP.md`](ROADMAP.md) for what's planned._

## [0.48.0] — 2026-07-04

### Fixed
- **Live setting changes apply without a stop/start** (#100): the worker used
  to snapshot config at start and never re-read it, so edits (scan rate, model,
  tiling, confidences, motion params, ROI…) needed a restart. Each worker now
  re-reads the saved config every ~2 s and reconfigures its detector in place
  (`PersonDetector.reconfigure`); a model/accelerator/ROI change resets the net
  on the next frame. Roles (roll/track) update live too.
- **"Motion off" runs the LIVE path, not the still-scan path** (#101): it was
  routed through the heavy locator/scan branch (backwards — maximally
  expensive) and mis-tagged detections as `still-scan` (#104). Motion-off now
  runs the camera's live detection every frame, ungated; its finds are tagged
  as the live path (`motion`), and only a genuine forced scan is tagged
  `still-scan`.
- **The live-feed selector never shows the wrong camera** (#103): a specific
  camera request now returns that camera or nothing — it no longer silently
  falls back to the streamed camera's feed when the requested one isn't
  running.
- **Find-my-cat only scans watched cameras** and gives **persistent feedback**
  (#103): a status line under the button says whether it found a cat (where +
  score) or fell back to the last sighting — no more too-fast flash.
- **The workflow line reflects the saved settings** (#104): it re-renders after
  a save, so "still-cat scan every 5s" no longer lingers when the scan is off.
- **GUI pass #2** (#106, #105): the Live-camera escalation toggle moved into
  the Escalation card (it was left behind in the VLM card); the camera **Edit**
  arrow now flips ▾/▴; the status dot is round; the unverified-model label is
  short (no overflow); the four feed toggles are shortened (🌈 Trail / 📍 Last /
  🎞 Smooth / 📺 Live); the **last-known box is violet**, not grey; the API-key
  label stays on one line; camera head toggles align across rows; Set region /
  Add zone sit side by side; sightings thumbnails keep aspect ratio and the
  last-seen image reserves its space (no load flicker).

### Added
- **Live-detection tiling** (#101): a per-camera **Live tiling** / **Live
  overlap** setting, independent of the still-scan's — the live path can now
  tile like the scan does. Default **off** (it multiplies per-frame cost by the
  tile count); the still-scan keeps its own separate tiling.

### Note
The larger #101/#102 restructure — dedicated collapsible settings *sections*
for each mode (live / still-scan / find) with full model+tiling+overlap+
confidence, removing the per-camera scan-model in favour of a Cat-cam
still-scan group, and moving the last-scan indicator there — is the remaining
follow-up; the correctness bugs it named (tiling scan-only, motion-off
routing) are fixed here.

## [0.47.1] — 2026-07-04

### Changed
- **Escalation gets its own GUI section** (#95): the ladder controls moved out
  of the VLM/API-key card into a standalone "🔍 Escalation ladder" card —
  placement only; the (work-in-progress) behavior is untouched, and it stays
  independent of the Find-my-cat button by design.
- **GUI layout & consistency pass** (#97): button rows no longer touch
  adjacent boxes/sections (camera actions, activity Clear, Model-files
  Generate); head quick-toggles align across camera rows; long `<select>`
  text no longer collides with the dropdown arrow or its neighbours; the
  camera Edit button's arrow now flips (▾/▴) like every collapsible; **"Cat
  tiling" is renamed "Tiling"** and the still-scan group (Scan model, Tiling,
  Tile overlap, Scan frames) sits directly under Model/Accelerator; the
  setup divider now says exactly what auto-saves ("cameras save with their
  Save button; everything else saves as you change it") and the Model files
  card moved below the numbered setup steps as an advanced tool.

## [0.47.0] — 2026-07-04

### Added
- **"Show me the cat" can actually search** (#92): with the new **Active scan
  on click** setting, the button runs a real detection pass across the find
  cameras (`POST /api/cats/find`) — a still cat in a motionless room is
  *found*, not guessed at. The pass runs a selectable **Find model** (default:
  each camera's own) with thorough tiling on each camera's latest frame via
  the tester's separate net cache — never the worker's net, so no race with
  the live loop. Finds are recorded (source `find`), the best camera is
  boosted, and the feed jumps there. Off by default: the button then behaves
  exactly as before (jump to current/last sighting). No escalation ladder in
  this button — deliberately, per the issue.
- **Cats-only sightings log** (#93, first slice): a **🐾 Recent sightings**
  block in the Cat-cam card — timestamp, camera, zone/region, score, and a
  **detection-source tag** on every entry (`motion` / `still-scan` / `track` /
  `find` / escalation's rungs), so you can verify each feature is actually
  firing. Worker sightings are now tagged at the source.
- **In-page lightbox everywhere** (#93): sighting thumbnails and activity-log
  snapshots open in the existing click-anywhere-to-dismiss overlay instead of
  a new browser tab.

## [0.46.0] — 2026-07-04

### Added
- **The still-cat scan gets its own model** (#94): a per-camera **Scan model**
  select (`cat_scan_model`, default "same as camera") — the asymmetry the
  issue nailed: live detection gets many frames of a moving cat (a fast model
  is fine), the still scan gets ONE hard static look at a sleeping cat in a
  dark corner (it wants the heavy tiled model). Runs through the existing
  locator slot, resolved to the right precision per the accelerator (#90),
  degrading loudly to the camera's net if the file is missing. The **tile
  overlap** knob is now exposed in the camera card alongside tiling/frames.
- **Still-scan last-run indicator** (#94): each camera card shows
  "still scan: Ns ago — cat found / no cat", fed from the worker via
  `cam_status` — glanceable proof the scan is actually firing.

## [0.45.1] — 2026-07-04

### Fixed
- **Motion "custom" is actually configurable** (#96): selecting Custom now
  reveals the three knobs the detector reads (min area fraction, per-pixel
  diff threshold, min blob px), seeded from the camera's stored values. It
  was a selectable dead end.

### Added
- **Motion gating can be turned OFF** (#96): a new "Off — detect every frame"
  sensitivity runs the net continuously (practical now the accelerated
  workhorse is ~tens of ms). Honesty preserved: the motion pre-filter still
  runs — `outcome.motion` stays true-to-life (rolls still require a real
  entrance), the trail/null-frame bookkeeping keep their real verdicts, and
  the cooldown pause still skips the net. No-motion frames with the gate off
  record cats through the still-scan path (rising edge, never a roll).

## [0.45.0] — 2026-07-04

### Fixed
- **Camera quick-toggles persist on click** (#91): the collapsed-row 🎲/🐱/👁
  toggles now auto-save immediately (a minimal `{name, url, field}` merge —
  they have no visible Save button, so unsaved state used to be silently
  wiped whenever any *other* camera saved and re-rendered the list). A failed
  save flips the toggle back and says so. Saving one camera has never touched
  another's stored settings; now there's no unsaved state to lose either.
- **The log says "cat", not "dog"** (#98): a detection counted as the cat via
  the dog-as-cat toggle was logged as "Still dog seen" under a 🐱. The
  sentence now says cat; the stored sighting keeps the raw label (data stays
  honest — only the wording matches the intent you configured).

### Changed
- **Precision is no longer a model choice** (#90): the model pickers (camera,
  tester/benchmark, sweep) offer the **3 logical models** (26x / 26m / 11n);
  `resolve_variant()` picks the file from the **accelerator** — FP32 for
  cv2.dnn (cpu/opencl), FP16 for onnxruntime/TensorRT/OpenVINO paths when the
  file exists. Legacy configs naming `*_fp16` normalize to the base (and can
  never force FP16 onto cv2.dnn). The `_fp16` registry entries stay for
  provisioning + old configs, hidden from pickers.
- **Fallbacks are visible, never re-labelled** (#90): every runner is stamped
  with what it *actually* ran on and why. The benchmark table shows "ran on X
  (requested Y — reason)" on any fallback; camera chips show `● live · accel`
  and flag a degraded camera (`● live (cpu!)` + tooltip); `cam_status` carries
  `ran_on`/`fallback` for the API.

## [0.44.0] — 2026-07-04

### Added
- **Model provisioning** (#86): the app now knows what model files *should*
  exist and whether the bytes on disk are the ones that were vetted — closing
  the gap where a stale June FP32 26x was silently benchmarked as "the 26x".
  - `models_manifest.json` records each vetted file's sha256 + precision +
    golden-head verdict; the **audit** (GUI "🧰 Model files" card,
    `GET /api/models/audit`, `python -m d20app.provision --audit`) reports
    every settled file as ok / missing / **unverified** (present, unknown
    provenance) / **stale** (changed since vetting). The two bundled models
    ship manifested + golden-verified.
  - **Generate from the GUI or CLI**: the card's button (or
    `python -m d20app.provision`, also run by `setup.sh` when ultralytics is
    present) builds whatever is flagged — golden `end2end=False` head, FP32 +
    FP16 onnx per model, TensorRT engines where the #82 driver gate passes —
    verifies the head (onnx metadata, or a real cv2 forward for FP32), and
    stamps the manifest. `ultralytics` stays a build-time-only dep: without
    it you get instructions, never a surprise install.
  - **Never silent**: a present-but-unverified/stale model's GUI label carries
    a ⚠ flag pointing at the Model files card.
  - `yolo11n_fp16` is now a registered variant (the #86 registry gap), and
    engines are one-per-base-model (`yolo26x` and `yolo26x_fp16` share
    `yolo26x.engine` — engines are always built FP16, the settled precision).
  *(NAS: run one provisioning pass — generation itself needs ultralytics and,
  for engines, the CUDA-13 driver; CI verifies everything around it.)*

## [0.43.2] — 2026-07-04

### Fixed
- **The tester/benchmark Accelerator dropdown is now generated from
  `ACCEL_OPTS`** (#87) — it was a hardcoded HTML list that silently missed
  TensorRT (the exact place you'd benchmark it). One source of truth; it can't
  drift again. The stale hardcoded `MODEL_OPTS` pre-load fallback (dropped
  `yolo11m`/`yolo11m_960`, the old "320" label) is replaced by the two bundled
  models with their registry labels — the real list still comes from
  `/api/models`.
- **Checkbox labels no longer overflow the camera card** (#88): the `.grid`'s
  `input { width: 100% }` rule was stretching the checkbox *itself* to the
  full cell width, shoving "Count dogs as the cat" / "Track fusion" text off
  the card and detaching the tick from its label (measured: a 13 px checkbox
  in a 327 px layout box). A specificity-matched `label.checkbox input
  { width: auto }` restores a normal aligned row; long labels wrap.

## [0.43.1] — 2026-07-04

### Fixed
- **TensorRT driver guard no longer blocks the drivers that enable it** (#85):
  newer NVIDIA drivers (610.x) relabelled `nvidia-smi`'s header field to
  `CUDA UMD Version:`, which the 0.43.0 regex missed — so the guard read
  "CUDA None" on a CUDA-13.3 driver and refused. The parser now accepts both
  labels, and because the header is fragile by nature (relabelled once
  already), a torch that is *actually running* CUDA is accepted as a
  secondary signal (torch only runs if the driver supports its toolkit, so
  `torch.version.cuda` is a valid lower bound).

## [0.43.0] — 2026-07-03

### Added
- **TensorRT accelerator** (#82, opt-in): a `tensorrt` option loads a prebuilt,
  GPU-specific `.engine` — measured by the issue at **1.6×** the onnx-cuda
  workhorse on the NAS 3070 (175 ms → 111 ms at 3×3/0.20, identical 91%/0%;
  26m 1.37×, 11n 1.21×). Engines are a cached one-time build per machine
  (`models/export_trt_engine.py`, golden `end2end=False` head, FP16 default),
  never rebuilt per launch. **Driver guard:** TensorRT needs a CUDA-13-capable
  driver, and installing it on an older one breaks the torch stack — the app
  checks `nvidia-smi`'s driver capability *first*, refuses with instructions,
  and never pip-installs anything itself. Any failure (driver, package,
  engine) logs why and falls back to `auto` (verified CUDA, else CPU) — same
  model, same accuracy, just slower. onnx-cuda/`auto` remains the default;
  switching the default to TensorRT waits on live NAS results, per the issue.
  *(NAS: build engines, then compare a few frames' boxes vs onnx-cuda — the
  runner is written against the TensorRT 10 API but has not run on real
  hardware from this app; also watch VRAM with moondream resident.)*

## [0.42.1] — 2026-07-03

### Changed
- **"Last known location" is now live-tracked and self-effacing** (from the
  hard-mode demo review): the grey box previously followed the *sightings
  log*, which is throttled on purpose (~10 s between motion records,
  rising-edge still scans) — so it could lag mid-room while the cat was
  visibly boxed elsewhere, reading like a second cat. Now (a) the detector
  keeps a drawing-level track updated on **every** ≥`cat_confidence`
  detection (and track-fusion confirm) — no extra log writes, the log keeps
  its throttled cadence — and the overlay draws from that, falling back to
  the log only when the track is empty (e.g. right after a restart); and
  (b) the grey box is **suppressed while a fresh cat box is on screen** —
  a live detection *is* the answer; a stale grey echo of it isn't. Net
  effect: cat visible → no grey box; cat vanishes → the grey box appears
  exactly where she was last confirmed, seconds-fresh.

## [0.42.0] — 2026-07-03

### Added
- **Targeted boost**: when an unconfirmed VLM lead names a *place*, the
  10-second confirm-boost now aims at it instead of just looking harder
  everywhere — forced scans additionally run a **full-resolution zoom crop**
  around the lead's box through the **heaviest model on disk** (26x > 26m >
  the camera's own; FP16 variants only on CUDA). Both choices are #70's
  answer to "heavier model or higher resolution?": heavier model measured
  better (91% vs 82% vs 75%), while raw >640 input measured *worse* — the
  zoom crop **is** the resolution lever that works. The boost never
  downgrades (a 26x camera keeps its 26x), degrades loudly to the camera's
  own net if the heavier file is missing, and its results confirm/record
  through the ordinary YOLO paths only. The feed shows the spot as an orange
  **"checking (lead)"** box while it runs.
- **"Last known location" on the live feed** (default **on**): a grey,
  age-labelled box at the camera's newest recorded sighting, so "where was
  she last?" stays answered even when nothing is detected right now. Fades
  out after 30 min (the age label keeps older boxes honest until then);
  toggled by the **📍 Last known** checkbox by the feed
  (`/api/stream?last_known=0` to hide). Only *confirmed, recorded* sightings
  draw it — an unconfirmed lead shows as the orange "checking" box during
  its window and is never promoted to "last known".
  *(NAS: watch a lead → targeted boost → confirm round-trip on real footage,
  and eyeball 26x latency per forced scan on the 3070/CPU.)*

## [0.41.0] — 2026-07-03

### Changed
- **People don't leave cat trails**: the trail now blanks out the region of any
  fresh **person** detection before stamping silhouettes — a human arm moves in
  cat-sized blobs, and the trail is a *cat* trail. Because the trail stamps each
  frame *before* the net runs, the net's person verdict also **retroactively
  erases** stamps made in the previous few seconds inside the person's box
  (pixels, route points, and the endpoint — which falls back to the previous
  genuinely-cat stamp). Exclusion boxes are padded 10% and expire after 3 s
  without a fresh person hit. *Honest limit:* during the cooldown
  detection-pause the net is skipped, so a person moving through the pause can
  still stamp until detection resumes and the erase catches what falls inside
  the then-current box. *(NAS: watch a real "human in frame" episode — the arm
  test — and confirm the trail stays cat-only.)*

## [0.40.0] — 2026-07-03

### Fixed
- **The default model now ships at its benchmarked size** (#80): the bundled
  `yolo11n` was a **320** export wearing the **640** benchmark's numbers
  (75%/0% — measured on the 640 golden 11n only; the 320 was never benchmarked
  and performed far worse). The bundled file is now a 640 raw-head export and
  the registry/labels/docs say 640 everywhere. Slightly more CPU per frame,
  honest accuracy. *(NAS: run the batch benchmark once against the full set to
  pin this exact file's numbers — the weights are the same as the golden 11n;
  the export flags may differ cosmetically.)*

### Changed
- **Dropped models now fail loudly** (#79, answering the design question):
  `yolo11m` (+ its `_960`/`_1280` locator exports), `yolo11x`, `yolo26n`, and
  `mobilenet_ssd` names raise an actionable error ("dropped in #71 — pick
  yolo26m/yolo11n…") at load and stop the loop like any config error, instead
  of quietly running a worse detector — the 0.25.0 no-silent-fallback stance
  applied consistently, as the issue argued. `yolo11m.onnx` and
  `yolo11m_960.onnx` are removed from the tree (~160 MB less checkout; git
  *history* is deliberately left unrewritten — see the #79 reply).
- **The >640 "locator input" knob is retired** (#79/#70 §5): 960/1280 input
  measured *worse* than 640 across the board, so the per-camera "Cat input
  size" select is gone and `cat_scan_imgsz` is a legacy no-op (old configs
  still load; any value falls back to native + tiling). Tiling is the
  resolution lever.

### Notes
- Suite: **319 tests** (+2 net: the dropped-model coverage replaced the old
  yolo11m variant tests — dropped names raise the actionable error,
  a dropped model in a detector is fatal-not-retried, the 640 floor is
  asserted, decode parity now runs at the export's true size).
- Old configs: a camera that names `yolo11m*` will now stop with a clear
  message naming the replacement — that's the intended behaviour per #79
  (two-person setup, no silent degradation). Everything else coerces as
  before.

## [0.39.0] — 2026-07-03

### Fixed
- **Trail flood on real hardware** (from the first NAS screenshot — a wall of
  green): a camera auto-exposure / white-balance / lighting shift made the
  *whole frame* differ from the remembered null frame, and the entire room was
  stamped as one giant "silhouette" at a single moment (which the recency ramp
  then painted mid-ramp green, edge to edge). Two guards now stop that class of
  failure:
  - **Global-change guard**: a diff covering more than 35% of the frame is
    treated as a lighting event, not an animal — the scene is re-adopted as the
    new null and nothing is stamped.
  - **Silhouette-size ceiling + blob hygiene**: only plausibly-animal-sized
    contour blobs are kept (a cat is neither 3 pixels nor a quarter of the
    room), at most the largest 4 per frame — the trail is shapes, not confetti.

### Added
- **Live trail overlay** (the originally-envisioned display): a **🌈 Trail
  overlay** checkbox by the live feed composites the trail onto the live
  stream (`/api/stream?...&trail=1`) — silhouettes and route update in place
  as the cat moves. Purely visual; detection is unaffected. The 🌈 snapshot
  button remains for a still image.
- **Route line + legend**: the silhouette centroids are joined into a path
  line coloured by the same recency ramp, and every render carries a legend —
  **blue = older → red = newest** with the episode's span in seconds — so the
  image explains itself. Tint opacity lowered 0.55 → 0.35 (the old value was
  unreadable over real footage).

### Notes
- Colour key, for the record: the ramp is a hue sweep blue → cyan → green →
  yellow → red. Green is the *middle* of the age range — in the flooded
  screenshot everything shared one mid-episode stamp time, hence a green room.
- The trail tracks **all** motion (people included) — that's inherent to a
  motion trail and now stated in the GUI help.
- Suite: **317 tests** (+4: a global lighting change resets the null instead
  of flooding, an oversized blob is rejected, the route path is recorded and
  rendered with the legend, the live overlay composites and degrades to the
  plain feed when there's no trail).
- NAS to verify: how often the guard fires on the real cameras (exposure
  hunting), overlay readability at night, and the 35%/25% thresholds on real
  scenes — `trail.GLOBAL_CHANGE_FRAC` / `MAX_SILHOUETTE_FRAC` are the knobs.

## [0.38.0] — 2026-07-02

### Added
- **User & workflow guide** (`docs/USER_GUIDE.md`, served in-app at **`/guide`**
  via the ❓ header link): the app grew a lot of buttons fast, and it wasn't
  clear what runs automatically vs on demand, or what each toggle changes.
  The guide covers: the app in one minute; **the default (benchmark-proven)
  workflow** step by step — motion gate → YOLO on auto-CUDA → track fusion →
  cat trail → still-cat scan at 3×3/0.35 with ×3 averaging — plus the one-time
  26x-FP16 upgrade; a **GUI map** (every card, every button, when to use it);
  **every toggle with what it changes** if flipped; **recipes** ("where's my
  cat", "camera keeps missing", "false-fires on a cushion", night misses,
  zones, model comparisons); and the **trust ladder** (how a sighting earns its
  `source` tag, and what is deliberately never recorded). One source: the
  markdown in the repo *is* the page, rendered by a tiny built-in converter
  (no new dependency).
- **Workflow line in the status bar**: a one-glance summary of what actually
  runs when the loop is on — model, accelerator, track fusion, still-scan
  cadence/tiling/averaging, and whether VLM tools may touch live cameras.
  Updates live as settings change.
- **Card clarity**: the Test-detection and VLM cards now say what they are in
  the heading ("a sandbox — nothing here touches the live loop until you
  save"; "the AI second opinion — never confirms a sighting by itself").

### Notes
- Suite: **313 tests** (+4: /guide serves the rendered markdown with the
  load-bearing sections intact, the GUI links it and carries the workflow
  line, converter basics, HTML escaping).
- The defaults the guide documents ARE the proven workflow from #70/#71 —
  nothing was re-tuned in this release; it documents and surfaces what
  0.35.0–0.37.0 made the default.

## [0.37.0] — 2026-07-02

### Added
- **Temporal score fusion ("track-before-detect")**: YOLO judged every frame
  independently, so a cat scoring 0.35 in eight consecutive frames — the box
  gliding smoothly along a plausible path — was discarded eight times. Now
  weak locator hits (`0.2 ≤ score < cat_confidence`, decoded in the **same
  forward pass** — the floor is a post-filter, not extra inference) chain
  across frames by overlap and confirm as **one sighting** when the chain has
  ≥4 hits inside 5 s **and net-travels ≥3% of the frame diagonal**. The
  movement requirement is the decoy guard: per the benchmark arc (#69/#70),
  the thing that separates a decoy from a cat is that the cat *moves* — and
  its deliberate flip side is that a *stationary* weak cat never fuses (still
  cats remain the averaging/tiling scan's job). Confirmed tracks are recorded
  as ordinary sightings tagged `source: "track"` with the honest **mean** weak
  score, log a "confirmed by track fusion (N weak hits over Ns)" activity
  line, and count toward cat presence. A per-track 30 s cooldown keeps one
  walking cat from spamming the log. Weak boxes never reach the live feed,
  labels, or logs on their own. Pure YOLO evidence — no VLM anywhere in this
  path (0.33.0's rule holds); this is the recall-raising mirror image of
  `confirm_frames`' precision-raising streak. Per-camera **Track fusion**
  checkbox (`track_fusion`, on by default; off = pre-0.37.0 behaviour).

### Notes
- Suite: **309 tests** (+9: moving weak track confirms exactly once,
  stationary decoy never confirms, window pruning, reconfirm cooldown,
  teleporting hits don't chain, detector wires weak hits to the fuser while
  keeping them out of the live feed, fusion-off preserves single-frame
  behaviour, the loop records `source="track"`, config/camera inheritance).
- NAS to verify: the real-footage hit rate on walking cats, and that RTSP
  compression shimmer doesn't fake smooth chains (the IoU-chaining + travel
  requirements are the guards; if shimmer slips through, raise
  `fusion.MIN_TRAVEL_FRAC` or `MIN_HITS`).

## [0.36.0] — 2026-07-02

### Fixed
- **VLM batch ignored the user's prompt** (#72): the batch path ran a baked-in
  prompt no matter what the prompt box said (the "rhino?" test failed), making
  batch prompt-testing impossible. The batch now threads the user's prompt to
  the model exactly like the single-image path; blank falls back to the model's
  validated default.

### Changed
- **Validated default prompt (P6)** (#72): the default is now the bake-off
  winner — *"Ignore plush toys, statues, paintings, reflections, and empty cat
  beds. Is a real live cat visible in this image? Answer with exactly Yes or
  No."* — 97% recall / 2% FP on the full set (prompts without the negative
  exclusions scored 20–73% FP). Every prompt should end with an explicit
  yes/no instruction so the verdict parser can vote on the output.
- **Prompts are model-specific** (#74): per-model defaults live in
  `MODEL_PROMPTS` (moondream3 gets a deliberately short, *unvalidated* form —
  P6 on M3 produced ~100% FP). The GUI swaps the default on a model switch and
  shows a warning when a custom prompt is carried across models.
- **Upload/test-queue cap 100 → 1000** (#73): a full benchmark set (199 cats +
  43 nulls) now runs in one pass. This is the app-level upload queue only —
  moondream's `max_batch_size`/`kv_cache_pages` VRAM params are untouched
  (raising those OOMs the 8 GB card), and a test asserts they stay put.

### Added
- **📍 Where? — detect mode in the tester** (#75): `POST /api/vlm/locate` runs
  moondream's `detect` on the picked frame and draws the proposed regions —
  the diagnostic for false-positive frames ("what feature is it locking
  onto?"), which informs better exclusion prompts. Purely informational:
  nothing is recorded (a bare detect region is never trusted — 0.33.0 rules).
- **Reasoning toggle for moondream3** (#76): `query()` was always called
  without `reasoning`, so M3 ran in its weaker non-reasoning mode. A
  *Reasoning (M3)* checkbox now passes `reasoning=True` through the single,
  voted, and batch paths; the reasoning text is surfaced with the response.
  No effect on moondream2; staying compatible with models whose `query()`
  lacks the kwarg is tested.

### Notes
- Suite: **300 tests** (+8). The P6 numbers (97%/2%) are the maintainer's
  full-set measurements (#70 §6, moondream2 local); the M3 default prompt is
  explicitly unvalidated — M3 remains cloud-only and not the deployment pick.

## [0.35.0] — 2026-07-02

### Changed
- **Benchmark-settled model lineup** (#71, from the authoritative benchmark #70):
  the selectable models are now **26x** (workhorse: 91% recall / 0% FP at
  3×3/0.20), **26m** (lightweight: 82%/0% at 2×2/0.20) and **11n** (floor:
  75%/0%), plus **FP16 variants** of the 26-series (identical accuracy, up to
  2.2× faster on CUDA; export with `scripts/export_yolo.py --half`; pair with
  the auto/CUDA accelerator — `cv2.dnn` FP16 handling is unverified). Dropped:
  11m (beaten by golden 26m) and its `_960`/`_1280` locator exports — their
  registry entries survive so **old configs keep loading**; they just can't be
  picked for new ones. Model labels + the `selectable` flag now live in the
  `yolo.MODELS` registry — one source for every dropdown and the sweep (#50).
- **Golden-export guard** (#71/#70 §2): the decoder now **refuses** an end2end
  (NMS-baked, `(1, 300, 6)`) ONNX head with a clear error naming the fix,
  instead of silently mis-decoding it and quietly costing 4–9 recall points —
  the exact failure that skewed the earlier `.pt`-era model decisions.
  `scripts/export_yolo.py` now passes the full golden recipe explicitly
  (`end2end=False, nms=False, dynamic=False, batch=1`).
- **`auto` accelerator, now the default** (#71): CUDA when it genuinely binds —
  reusing the verified-provider check, so auto can never silently run slow —
  else CPU with a loud log line. NAS-verified CUDA is what makes the heavier
  models everyday-runnable; a machine without a GPU still works and says so.
- **Benchmark-settled scan defaults** (#70 §5): still-cat scan tiling default
  **4×4 → 3×3**, tile overlap **0.2 → 0.35** — 3×3/0.20-0.35 is the clean
  recall-per-ms winner; 4×4 adds ~1 cat at 1.7× the latency and high-overlap
  4×4 buys recall with false positives.

### Notes
- Suite: **292 tests** (+5: end2end head refused with an actionable error, raw
  head still decodes, auto-accelerator CPU fallback, dropped models excluded
  from the picker while their registry entries survive, new defaults).
- NAS follow-ups: re-export the deployed models with the golden recipe (the
  guard will catch any bad one at first use), produce the FP16 exports, and let
  `auto` pick CUDA — `run.py`'s LD_LIBRARY_PATH handling already covers the
  torch-lib path trick.

## [0.34.0] — 2026-07-02

### Added
- **Still-scan frame averaging**: the periodic still-cat scan now averages a
  short burst of back-to-back frames (config/GUI `cat_scan_frames`, default 3,
  max 8) before the locator net runs. On a genuinely still scene sensor noise
  drops ~√N — *real signal recovery* (N samples of the same scene), the
  opposite of the SR-style synthesized detail #69 rejected — which is exactly
  what a dim room and a sleeping cat need. Safety valves: any movement
  mid-burst (cheap downscaled-gray diff) aborts to the single sharp frame, so
  a mover is never smeared; a read failure or size change aborts too; the
  **fast treat path never averages** (bursting is confined to the forced scan,
  which is not latency-sensitive); smooth-feed mode skips it (the grab thread
  is the capture's sole reader). Per-camera **Scan frames** knob on the camera
  card; old configs inherit the default via coercion.

### Docs
- **ROADMAP — future improvements**: recorded the post-#69 assessment as
  actionable items — the recovers-signal vs synthesizes-detail filter,
  CLAHE-for-dark-frames (measure first), the ranked equipment list (thermal
  sensor nodes as hint sources, night lighting/optics, BLE collar, mmWave,
  doorway break-beams, NAS GPU; PTZ/Wi-Fi-CSI/audio assessed and skipped), and
  a **step-by-step guide to fine-tuning YOLO on our own cameras** (dataset
  assembly from the sightings log, hard negatives from the decoy set, honest
  scene-level splits, single-class locator recipe, 5090 training command,
  ONNX export into the model registry, benchmark gating, miss-mining).

### Notes
- Suite: **287 tests** (+6: noise actually drops on a still scene, movement
  mid-burst aborts, read-failure/size-change aborts, forced scan bursts while
  the treat path reads exactly one frame, the knob clamps at 8 / disables at 1,
  new cameras inherit the global knob).
- NAS to verify: the real night-frame recall gain (CI proves the mechanics and
  the math, not the camera's noise profile), and that RTSP burst reads don't
  hiccup on the real streams.

## [0.33.0] — 2026-07-02

### Changed
- **Only YOLO confirms — VLM-only verdicts demoted** (#69): the maintainer's
  decoy benchmarks measured VLMs at **37–42% false positives**; majority voting
  reduces run-to-run *variance*, not that systematic *bias* (three votes just
  agree wrongly on a cat-shaped decoy). Accordingly:
  - **Escalation ladder**: a votes-only VLM "yes" (rung 2's query fallback, all
    of rung 3) can no longer return `found` — it comes back as a `vlm_probable`
    lead and surfaces in the **"probable"** tier (orange box, honest note,
    **never recorded** as a sighting). Rung 2 now also keeps scanning past a
    votes-only "yes": a YOLO confirmation on a later region beats it. A live
    VLM lead **boosts detection** on that camera so real YOLO gets the chance
    to confirm on the next frames — a real cat becomes a normal recorded
    sighting, a decoy dies quietly.
  - **Temporal mosaic**: a "yes" is now labelled an **unconfirmed hint**
    (`hint_note` in the response, shown in the GUI); on a live camera it boosts
    detection instead of standing as a conclusion. Nothing was ever recorded
    from the mosaic — now the response says so and hands off to YOLO.
  - The VLM's confirmed roles are unchanged: proposals for YOLO to check
    (rung 2, decoys filtered by YOLO's 0% FP), and judgment that is already
    labelled inference (the probable tier).

### Notes
- Suite: **281 tests** (+2; three ladder tests rewritten for the demotion, new
  coverage: a later YOLO confirm beats an earlier votes-only yes, a live VLM
  lead records nothing and boosts, temporal hint notes with/without a camera).
- The 37–42% figure comes from the maintainer's own extensive decoy testing
  (issue #69) and was taken as trusted input; the NAS decoy run can still
  measure the voted-moondream setup specifically.

### Docs
- **`DESIGN_RATIONALE.md`** — a reviewer-facing document explaining the *why*
  behind the architecture: the standing invariants (treat path is sacred,
  cheap-first, never record unconfirmed claims, priors not state), the
  reasoning behind each smart-detection slice, the ideas assessed and rejected
  (now including #69's super-resolution verdict), the engineering conventions,
  and the open NAS-validation queue. Linked from `CLAUDE.md` so any Claude
  session finds it.

## [0.32.0] — 2026-07-02

### Added
- **Temporal VLM analysis — frame mosaic** (#68): each running camera now keeps a
  small ring buffer of recent frames (up to 8, spaced ~1 s apart, downscaled to
  ≤480 px — a few MB per camera). `POST /api/vlm/temporal` tiles them into one
  numbered grid image, oldest first, each tile labelled with its age ("1 (-4s)" …
  "N (now)"), and asks moondream a single voted query: *did a cat appear or move
  through the scene, and in which frame(s)?* One query covers ~8 s of history —
  the temporal rung of #65/#68 without any video-LLM dependency. Works on the
  Test tool's **video uploads** (a ⏱️ **Temporal check** button appears when the
  upload is a video: the sampled frames become the grid) and on **live cameras**
  (a ⏱️ button on the escalation ladder's camera row; gated by the same
  `vlm_escalation` toggle and clear 409s when the loop is off or the ring hasn't
  filled yet). The response includes the mosaic itself so you can see exactly
  what the model saw.

### Notes
- Suite: **279 tests** (+7: ring spacing/cap/copy semantics, `read_and_detect`
  feeds the ring, mosaic geometry incl. the 9-tile cap and 1×1/empty edges,
  video-session and live-camera endpoint paths with the privacy gates).
- **Verified in CI vs NAS**: the ring buffer, grid geometry, endpoint gating and
  GUI are tested here. The core premise — that moondream can genuinely reason
  over a grid of numbered frames — is **not** provable with mocks; the
  `TEMPORAL_PROMPT` and tile size need real-hardware validation on the NAS
  before this earns a place in any automatic flow.

## [0.31.0] — 2026-07-02

### Added
- **Semantic zones** (#68): per-camera named rectangles drawn on the preview frame
  (same drag interaction as the ROI picker — a **🚪 Add zone…** button on each camera
  card; zones listed as removable chips). Sightings whose box lands in a zone are
  recorded with its name, and the activity log / Cats card say **"cat seen (the
  couch)"** instead of a grid cell. Zones marked **exit** (doorways) sharpen the
  trail's "may have left the view" check: a trail endpoint inside an exit zone
  suppresses the "probable location" claim even when it isn't at a frame edge.
  Zones live on the saved camera (config `cameras[].zones`); old configs coerce to
  `[]`. Zone boxes are in full-preview coordinates; detection boxes are ROI-crop
  coordinates — `zone_for()` shifts by the ROI origin so the two meet correctly.
- **Sighting heat maps** (#68): `GET /api/cats/heatmap?camera=` renders the
  sighting history (the tracker's 500-cap log) as a Gaussian-softened density field
  colour-mapped over the camera's current frame — the room's hot spots (the basket,
  the couch arm, the sunny patch) at a glance. A **🔥 Heat map** button sits next to
  🌈 Show trail. Pure CPU (`d20app/heatmap.py`).
- **Time-of-day prior** (#68): `CatTracker.by_hour()` + `likely_cameras(hour)` rank
  cameras by historical presence around the current hour (±1, wrapping midnight) —
  exposed as `by_hour`/`likely` in `/api/cats` and shown on the Cats card as
  *"Usually around now: Kitchen (12) · Bedroom (3)"*. Explicitly a **prior** for
  ordering a Find-My-Cat sweep — a hint from history, never a tracked state (the
  fragile house-graph tracking #65 rejected stays rejected).

### Notes
- Suite: **272 tests** (+10: zone naming + ROI shift, exit-zone matching, zone
  persistence with legacy records, the hour histogram + wrap-around ranking, heat-map
  hot/cold pixels + degenerate-box rejection, `/api/cats` prior exposure, heat-map
  endpoint JPEG + 404/409s, exit-zone suppressing "probable", and camera-zone
  round-trip through the saved-cameras API), all green.
- **NAS to verify**: zone-drawing UX on the real cameras (drag on the preview),
  whether the heat map reads well over real scenes, and the prior's usefulness once
  a few days of sightings accumulate.

## [0.30.0] — 2026-07-02

### Added
- **The cat trail** (#67) — slice 2 of the smart-detection roadmap, per the user's
  design. Frame-to-frame diffing only lights the *edges* of motion; diffing against
  the last **null frame** (the scene when nothing was moving) lights the **whole cat
  silhouette**. Each silhouette's pixels are stamped with when they were covered and
  coloured by recency — **blue = where the episode started → red = the latest
  position** — one image showing path, direction, and timing (`d20app/trail.py`,
  fed by every frame read; capped working resolution, a few small-image ops per
  frame, pure CPU).
  - The **null frame re-adopts the scene** after a few still seconds — self-heals
    lighting/auto-exposure drift, and (honestly noted) means a settled cat becomes
    part of "null", so her departure briefly lights her old spot as a "ghost" in the
    trail's oldest colours. A **motion episode** ends after a long stillness; the
    trail keeps rendering until a *new* episode starts.
  - **Trail-endpoint targeting**: the newest silhouette's box is a coordinate the app
    knows numerically. If it's **interior** (not near a frame edge — the v1 exit
    check), the cat plausibly didn't leave: the escalation ladder now takes it as its
    **highest-priority hint** (and as a predictor track fix).
  - **"Probable location"** — a third, honest outcome tier: when the ladder confirms
    nothing but the trail ends in-view, the response says *"no confirmed detection,
    but the last movement ended here Ns ago and no exit was seen — probable
    location"*, with the endpoint boxed in orange on the annotated frame and the
    trail image attached as evidence. **Never recorded as a sighting** — always
    labelled as inference. An endpoint at the frame edge (may have exited) makes no
    claim.
  - **`GET /api/trail?camera=`** serves the recency-coloured trail over the current
    frame (404 until something moves; 409 when not watching). GUI: a **🌈 Show
    trail** button next to the camera check, the probable badge + orange box in the
    escalation results, and the trail image inline when a live run returns one.

### Notes
- Suite: **262 tests** (+11: null adoption, silhouette painting + endpoint coords,
  blue→red recency colouring, edge-vs-interior endpoints, episode reset, endpoint
  staleness expiry, null-refresh absorbing a settled scene, the live escalate
  "probable" path against a real loop+detector with injected frames, no-probable at
  a frame edge, `/api/trail` JPEG + degradations, and the read_and_detect wiring),
  all green.
- **NAS still to verify**: real-scene trail quality — ghosting behaviour, lighting
  drift across hours, and whether the endpoint deep-scrub recovers the basket-cat
  case end-to-end. CI proves the geometry, colouring, episode logic, and API.

## [0.29.0] — 2026-07-02

### Added
- **The escalation ladder** — "zoom in and look again" for the small/distant cats the
  normal pass misses (#17's 0.00-confidence case; the roadmap's "VLM-guided cropping →
  YOLO escalation" next-step). When the quick look finds nothing, the ladder runs
  cheapest-first, stopping at the first **confirmed** find:
  1. **zoom+yolo** (free, no VLM): full-resolution square crops around the "look here"
     hints — where motion just happened, where the cat was last seen, and where a **CPU
     predictor** says a moving cat is headed (`predict_hint_box`: centroid velocity from
     the last timestamped fixes, uncertainty pad grows with staleness; stands down with
     <2 recent fixes — the VLM stays the failsafe when the track is thin) — re-checked
     by the ordinary YOLO net. In a crop the cat is big again.
  2. **vlm detect**: moondream's previously-unused `detect("cat")` proposes regions;
     each is cropped full-res and **confirmed** (YOLO first, voted yes/no query second).
     A bare detect region is never trusted alone — open-vocab detect false-fires on
     cat-shaped decoys.
  3. **vlm query**: last resort — the voted yes/no on the hint crops.
- **Motion location is now retained** (`MotionPrefilter.last_blobs` + timestamp): the
  blob boxes the pre-filter always computed (and threw away) survive as escalation
  hints, including sub-`min_area_frac` movers (a distant cat's small blob) — the motion
  **verdict is unchanged**. Foundation for the upcoming null-frame "cat trail".
- **`POST /api/vlm/escalate`** — on-demand only: a Test-tool frame (works without any
  config), or a live camera's latest frame behind the new **`vlm_escalation`** toggle
  (off by default; 403 when off, 409 when the loop/camera isn't available). The fast
  treat path never touches any of this. A confirmed live find records a sighting
  **tagged with its rung** (`source: zoom+yolo | vlm+yolo | vlm` — sightings now carry
  a `source` field, old records read back as `yolo`), logs an activity line with the
  annotated snapshot, and boosts the camera feed. Response includes the rung table
  (ran/crops/ms/result), every crop as a thumbnail, and the annotated frame.
- **GUI**: an "escalation" section in the VLM card — run on the uploaded frame, a
  "use VLM rungs" toggle (off = pure-CPU zoom+yolo), and a per-camera "Check camera
  now" (shown only when the toggle is on and detection is running); rung table + crop
  thumbnails + annotated result. Settings gains the "Live-camera escalation" checkbox.
- `d20app/escalation.py` is pure math with injected detectors — every ladder decision
  is unit-tested without a net or GPU. `moondream.detect_regions()` wraps the detect
  mode (normalized [0,1] coords per moondream 1.3.0; mapped/clamped/degenerate-rejected
  by `map_normalized_box`).

### Notes
- Suite: **251 tests** (+29: crop math incl. edge-shifting, box mapping round-trips,
  normalized clamping, hint dedupe/caps, the predictor's lead/stand-down/clamp cases,
  the cheap-first guarantee (rung-1 hit → VLM never called), decoy resistance (bare
  detect region never records), rung bookkeeping, endpoint 404/403/409/503 paths, a
  real end-to-end zoom+yolo find on the cat fixture, blob retention/verdict split, and
  sighting-source persistence), all green.
- **Honest verification split**: CI proves the crop math, ladder decisions, and
  degradations with mocks + real yolo11n on fixtures. **Only the NAS can verify**:
  moondream `detect()` output quality/orientation on real frames, VRAM co-residency of
  YOLO-CUDA + moondream2 on the 8 GB 3070 (the ladder's confirm rung defaults to CPU
  YOLO for this reason), and the real-world hit rate on known-miss frames. See the PR
  checklist.

## [0.28.0] — 2026-06-30

### Added
- **VLM multi-pass majority vote** (#60). moondream's yes/no isn't deterministic — the same
  frame can flip run-to-run, and the frames that wobble are the genuinely-hard ones (a black
  cat as a dark blob, a decoy). So each image is now queried **N times** (configurable,
  default **3**, max 9) and the verdict is the **majority vote**:
  - **Vote ratio is the honest confidence** — "4/5 yes" means strong agreement; "3/2" is
    genuinely ambiguous. This is the real number that replaces moondream's meaningless
    self-report (the confidence we'd already dropped). A tie yields no verdict (ambiguous).
  - **Borderline (non-unanimous) frames are flagged** — "split vote — review" — in the
    single tester, the batch table, and the cross-image summary report. Split votes are
    exactly the hard frames worth a human look or a YOLO cross-check, so they're surfaced,
    not buried.
  - Applies to all three paths: the single-image tester, the batch VLM tester, and the
    "also run VLM" toggle in the detection batch. The batch "VLM accuracy" (recall + FP) is
    computed from the **majority-vote** verdicts. A **Passes** control sits next to the
    mode selector; `passes=1` is exactly the old single-pass behaviour.
  - Affordable because queries are fast (~0.3 s on the GPU): 5 passes ≈ 1.5 s/frame, and
    this is a verification tool, not an every-frame live path.

### Notes
- Suite: **222 tests** (+4: the pure `majority_vote` over clear/tie/unanimous/empty inputs,
  the voted query's ratio + borderline + summed latency + verdict-matching reason, the
  single-pass passthrough, and the batch carrying the vote ratio), all green.

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
