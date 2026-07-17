# Review: full-code audit

**Date:** 2026-07-09 · **Reviewed against:** `main` @ 0.50.0 · **Status:** findings only (no fixes applied)

**Method:** five parallel dimension audits (concurrency/threading, web-layer
security, frontend, backend logic/math, config+tests+consistency), each reading
its files in full, plus first-hand verification of the top findings by the
reviewer (three proven at runtime). The earlier ~30-min **memory leak** review
(`2026-07-08-memory-leak.md`) is not repeated here; its two findings still stand.

**Threat model for security items:** LAN-only, no-auth-by-design home app. "No
login page" is in scope by design and not a finding; anything reachable by a
malicious LAN device / crafted camera-name / upload / config value is.

---

## Executive summary

The codebase is in good shape: **XSS is well-defended** (consistent `esc()` +
`textContent`), **YAML is safe** (`safe_load`/`safe_dump`), **config writes are
allowlisted**, **secrets don't leak to the browser** (`moondream_api_key`,
`camera_password`), and the **geometry/decode math** (letterboxing, ROI offset
composition, crop mapping, velocity prediction) is correct. No CRITICAL issues.

The findings cluster in three places, several of them **regressions from the
recent 0.48–0.50 work** (auto-save-everything + config hot-reload):

1. A **500 on routine saves** when a numeric field is cleared/blank (`_coerce`).
2. A **concurrency race** on the new `last_scan()` status path (500 under load).
3. A **credential-masking gap** — inline URL creds leak via two GET APIs.
4. **Frontend lifecycle** brittleness — one failed request at startup bricks the UI.
5. **SSRF** via attacker-set camera URL into FFmpeg.

Ranked findings below. Each is tagged with reviewer verification.

---

## HIGH

### H1 — `_coerce` raises on empty/None numeric values → HTTP 500 on routine saves ✅ proven at runtime
`d20app/config.py:284-298`
```python
if isinstance(default, int):   return int(float(raw))   # int(float("")) → ValueError
if isinstance(default, float): return float(raw)         # float(None)   → TypeError
```
Proven:
```
_coerce('', 0)   → ValueError: could not convert string to float: ''
_coerce(None, 0) → TypeError: float() argument must be ... not 'NoneType'
```
**Why it bites now:** as of 0.50.0 every control auto-saves on `change`, and
`POST /api/config` feeds the raw JSON straight into `update()`. Clearing any
numeric input (`dc`, `cooldown_seconds`, `brightness`, `scan_fps`, `web_port`,
…) to `""`, or any client sending `null`, throws inside the coercion loop →
**HTTP 500**. Not a permanent brick — the exception fires before `save()`, so
`config.yaml` is untouched — but it's an unhandled 500 on a normal action, with
no `try/except` around the coercion and no test covering it.
**Fix:** treat `""`/`None` as "keep the current/default value" (or 422 with a
clear message), and wrap the coercion so one bad field can't 500 the whole save.

### H2 — `last_scan()` iterates `_scan_last` while worker threads insert keys → intermittent 500 ✅ verified by reading
`d20app/loop.py:375` (web thread) vs `:743` (worker thread)
```python
# web thread, via /api/cats → last_scan()
for name, s in self._scan_last.items():          # :375
# worker thread, per still-scan
self._scan_last[name] = {"ts": ..., "found": ...}  # :743  (in-place mutation)
```
`_scan_last` is mutated **in place** (unlike the safe rebind-only pattern used
for `_detectors`). A key insertion during the web thread's `.items()` iteration
raises `RuntimeError: dictionary changed size during iteration` → the `/api/cats`
poll (fired every 1.2 s by the GUI) 500s. Most likely in the first scan cycle of
each watch session; worst with several cat-tracking cameras. The detection loop
itself is unaffected. This is a regression from the 0.49.0 `last_scan()` addition.
**Fix:** snapshot under a lock (or `list(self._scan_last.items())` built under the
existing `_status_lock`), or guard the write/read with a lock.

### H3 — Frontend: one failed request during `init()` permanently bricks the UI ✅ verified by reading
`app.js:1860-1874` + `api()` at `:4-9`
`api()` doesn't wrap `fetch` in try/catch, and `init()` has no `.catch`. A single
network blip / 502 on the first `/api/models` or `/api/config` rejects the
promise, aborts `init()` **before any `setInterval` is registered**, and there's
no recovery — blank/partial UI, no polling, no live feed, until a hard reload.
**Fix:** wrap `init()` in try/catch with a visible retry; make `api()` reject
into a handled path.

---

## MEDIUM

### M1 — Credential-masking gap: inline URL creds returned in cleartext ✅ verified by reading
`d20app/webapp.py:946` (`_public_config`) and `:937` (`_mask_cameras`) strip only
the dedicated `camera_password` / per-camera `password` fields; they return
`camera_url` and `camera_username` verbatim. RTSP creds are commonly pasted
inline (`rtsp://user:pass@cam/stream` — the app itself builds such URLs). Any
unauthenticated LAN device doing `GET /api/config` or `GET /api/cameras/saved`
then receives the full `user:pass@host`. Log paths are clean (`mask_credentials`
is applied on every error/label path); the gap is only these two GET APIs. This
contradicts the project's standing "mask credentials everywhere" invariant.
**Fix:** run `camera_url` / saved `url` through the existing `mask_credentials`
helper in both functions.

### M2 — SSRF / localhost-pivot via attacker-set camera URL → `cv2.VideoCapture(CAP_FFMPEG)` ✅ verified by reading
Sink `detector.py:107`, reached by `GET /api/preview` (`webapp.py:1043-1061`) and
the watch loop. `camera_url` is attacker-writable (`POST /api/config` /
`/api/cameras/saved`). FFmpeg's protocol handlers (`http`/`https`/`rtsp`/`file`)
originate a request from the app host, so a LAN device can pivot to services on
the app's **localhost** or a cloud metadata endpoint. Reachable media streams are
returned as JPEG (not fully blind); `file://` non-media fails to decode (existence
oracle, no byte exfil). Capped at MEDIUM because a trusted-LAN attacker can
already reach LAN hosts directly — the localhost pivot is the real delta.
**Fix:** allowlist the URL scheme (`rtsp`/`http(s)`/`usb:`) and reject loopback /
link-local / `file:` before opening.

### M3 — Frontend: auto-save can serialize a half-populated form (blanks/zeros) ✅ verified by reading (severity nuance corrected)
`app.js:560-603`. `wire()` attaches the `change→saveConfig` handlers
**synchronously**, before the awaited `loadConfig()` populates the scan/find
dropdowns (filled by `innerHTML` at `:535-543`). A user change in that window (or
after a failed `loadConfig`) posts the whole form with `cat_scan_tiling:""`,
`find_tiling:""`, and — note — `cat_scan_frames: Number("") === 0`, **not `NaN`**
(the auditor said NaN; JS `Number("")` is `0`). So the corruption is
empty-strings + zeros, which then hit H1's `_coerce` (`0` is fine; `""` for a
string field is stored as-is). Combined with H1, an empty numeric there is a 500;
an empty dropdown string is silently saved as `""`.
**Fix:** guard `saveConfig` behind a "config loaded" flag; don't wire auto-save
until `loadConfig` resolves.

### M4 — Frontend: whole-form auto-save has no in-flight guard → out-of-order POSTs lose updates ✅ verified by reading
`app.js:593-603`. Every change on ~25 controls POSTs the entire form with no
sequence number and no abort of the prior request. On a slow link, if change A
(sent first) lands after change B, the server's final state is A — reverting B.
`Object.assign(lastCfg, gathered)` per-response compounds the desync.
**Fix:** serialize saves (in-flight flag + coalesce), or send only the changed
field.

### M5 — Class-agnostic NMS suppresses valid different-class detections ✅ verified by reading
`d20app/yolo.py:666` — a single `cv2.dnn.NMSBoxes` across all class ids. Two
strongly-overlapping boxes of **different** classes suppress each other, so a cat
held in a person's lap (IoU ≥ 0.45) can drop the cat box → missed sighting/treat.
Inconsistent with the tiled path's `merge_nms` (`:675`), which is deliberately
per-class. This is the exact person+cat case the app centers on.
**Fix:** per-class NMS (group by label like `merge_nms`, or `NMSBoxesBatched`).

### M6 — `config.example.yaml` drift: 17 dataclass fields missing ✅ (spot-checked)
The example's header claims it "documents every setting" and `setup.sh` copies it
to `config.yaml`. Missing: `label_floor`, `live_tiling`, `live_tile_overlap`,
`cat_scan_model`, `cat_scan_confidence`, the whole `find_*` group,
`motion_sensitivity` + the motion params, `pause_during_cooldown`,
`keep_speakers_warm`. Not a runtime bug (`load()` fills defaults) but the claim is
false and hand-configurers are blind to whole subsystems (find-my-cat, live
tiling, per-mode settings).

### M7 — List/dict config fields get NO coercion; a scalar corrupts silently ✅ proven at runtime
`config.py` `_coerce("Kitchen", [])` returns `"Kitchen"` unchanged (proven). A
client posting `speaker_names: "Kitchen"` is stored verbatim; `speaker_targets()`
then iterates the string into `['K','i','t',…]`. Same shape for `locator_classes`
/ `active_cameras`. Mitigated because these normally arrive as arrays from
structured endpoints, but nothing enforces it and the docstring claims coercion.
**Fix:** coerce non-list → `[]` (or wrap scalars) for the collection fields.

### M8 — `stop()` ignores its join timeout → false "stopped", then won't restart ✅ verified by reading
`loop.py:193-204`. `stop()` joins with `timeout=10`, discards the result, sets
`status.running = False`, returns `True`. A worker inside a Cast call
(`block_until_active(timeout=10)`) or slow reconnect can outlive that; a later
`start()` sees the orchestrator still alive and returns `False`. GUI shows
"stopped" but won't restart until the wedged thread exits.
**Fix:** check the join result; surface "still stopping" instead of a false success.

---

## LOW (abridged — full list in the appendix)

- **L1** Smooth-feed grab thread can end up with two readers on one capture after
  a round-robin rest→rewake if the old grabber is wedged (`_start_grab` re-clears
  `_grab_stop`). Native cv2/FFmpeg two-reader UB. Narrow trigger. `detector.py:1253-1258` / `:1399-1413`.
- **L2** `_TensorRtRunner` never frees its two `cudaMalloc`s or the stream (no
  `__del__`/close) — VRAM leak on runner rebuild. NAS-only path. `yolo.py:440-445`.
- **L3** `save_manifest` is a non-atomic overwrite; a crash mid-write voids the
  provenance record → whole lineup re-exports. `provision.py:105-109`.
- **L4** `sound_file` config value is unvalidated → path traversal into
  `playsound` (reads an arbitrary local file as audio; no exfil). `caster.py:290`.
- **L5** `/snapshots/<path:name>` is a file-existence oracle (two distinguishable
  404 bodies); bytes are not served (`safe_join` protects). `webapp.py:1035-1040`.
- **L6** No `Host`-header validation → DNS-rebinding can drive every endpoint
  from the maintainer's browser (chains to M2). `webapp.py:955-958`.
- **L7** Polling intervals (`loadCats` @1.2 s) never cleared, no in-flight guard →
  request stacking under CPU load. `app.js:1869-1871`.
- **L8** Watch/smooth toggles update UI before the POST and never roll back on
  failure (inconsistent with the quick-toggles, which do). `app.js:320-327`, `:1763`.
- **L9** `provisionPoll` interval leaks on a transient status failure (early
  return before `clearInterval`). `app.js:74-84`.
- **L10** NMS score-threshold boundary: `>=` on the array filter vs strict `>`
  inside `NMSBoxes` — harmless float-equality edge. `yolo.py:651` vs `:666`.
- **L11** Degenerate (≈1 px) frame/crop can crash `cv2.resize` in `_letterbox`;
  escalation rung-2 crop lacks the `<2px` guard `square_crops` applies. `yolo.py:611`, `escalation.py:315`.
- **L12** Discovery: ONVIF client sockets never closed (leak per run); sub-second
  `searchServices` timeout truncated to `0`. `discovery.py:114-137`, `:87`.
- **L13** README says "59 tests" (`README.md:559`); actual ≈393 test functions
  (ROADMAP's "404" is right). Doc-only.
- **L14** `RollGate.cooldown_s` written outside `_roll_lock` (benign: atomic
  float, same value from every worker). `loop.py:592`.

---

## Verified CORRECT / SAFE (so the caller knows what was cleared)

- **Frontend XSS:** every render escapes camera names, log messages, sighting
  labels, zones, error strings, VLM output; snapshot filenames are server-generated;
  the log path uses `createElement`+`textContent`. No injection sink found.
- **Secrets:** `moondream_api_key` never reaches the browser or logs
  (`_public_config` strips it; status returns only `has_api_key`); `camera_password`
  masked; **only the inline-URL path (M1) leaks.**
- **Injection:** no `os.system`/`shell=True`; only subprocess is `nvidia-smi`
  fixed-argv; gTTS/playsound take library args, not shells.
- **YAML/config:** `safe_load`/`safe_dump`; unknown keys filtered on read+write;
  `/api/config` pops `cameras` and preserves blank secrets; `load()` can't be
  bricked (bad values → 500 before save, or reload-as-defaults).
- **Uploads:** `secure_filename` + extension allowlist + `MAX_CONTENT_LENGTH`.
- **`send_from_directory`** traversal-safe (Werkzeug `safe_join`); benchmark HTML
  reports fully `html.escape`d + slugified.
- **Geometry/decode math:** letterbox forward+inverse, ROI offset composition
  (added exactly once end-to-end), `square_crop_box` edge-shift, `map_normalized_box`
  (no x/y swap), `predict_hint_box` velocity (div guarded), `merge_hint_boxes`/IoU,
  golden-export shape guard, heatmap normalization, `majority_vote`, `frame_mosaic`,
  escalation ladder gating (votes-only never `found=True`) — all correct.
- **Concurrency SAFE (verified):** `_detectors` / `_viewing` / `_cat_boost` /
  `_boost_hint` / `MotionPrefilter.last_blobs` rebind-not-mutate; `_live_lock`
  covers all cross-thread frame/box reads; MJPEG path copies frames under the lock;
  caster `_lock`/`_play_lock` never nested (no deadlock); `trail` fully lock-guarded;
  shared `RollGate` only used under `_roll_lock` (no double-roll). `_scan_last`
  (H2) is the one exception.
- **Tested (NOT gaps):** shared-cooldown hot-reload, caster keep-alive, MJPEG
  endpoint, per-camera old-config back-compat, blank-secret preservation.
- **run.py:** `debug` never enabled; `0.0.0.0` intentional; `MAX_CONTENT_LENGTH`
  set. **Repo hygiene:** no secrets tracked; `config.yaml`/logs/snapshots gitignored.

---

## Top-5 test gaps
1. **Camera reconnect / read-failure recovery** — `_ensure_cap` is stubbed in
   every detector test; the `ok=False → drop cap → reopen w/ backoff` path and the
   `CameraError` open-failure branch are never driven. (Also the site of the
   memory-leak review's Finding 1.)
2. **`_coerce` on `""`/`None`/scalar-for-list** — no test; H1/M7 go unguarded.
3. **`camera_targets()` duplicate/whitespace names, non-dict entries** (F5/M-list).
4. **Discovery internals** (ONVIF/WSDiscovery/Cast parsing) — no tests.
5. **Provision download/verify success path** — only audit states tested.

---

## Suggested fix order (fastest ROI first)

1. **H1** — guard `_coerce` against `""`/`None` (+ test). Small, stops live 500s.
2. **H2** — snapshot `_scan_last` under a lock (+ test). Small, stops status 500s.
3. **M1** — mask inline URL creds in the two GET APIs (+ test). Small, closes the
   invariant gap.
4. **H3 / M3 / M4** — frontend: `init()` try/catch + retry; gate `saveConfig` on a
   loaded-flag; in-flight save guard.
5. **M5** — per-class NMS (+ test on an overlapping person+cat fixture).
6. **M2 / L6** — camera-URL scheme allowlist + Host-header allowlist.
7. Docs: **M6** (example.yaml) and **L13** (README count).

## Scope & confidence
- Static evidence is unambiguous for every finding; **H1, M7 proven at runtime**;
  **H2, M1, M5, M8 verified by first-hand reading** of both sides of each path.
- Severity for M2/L4/L5/L6 is calibrated to the trusted-LAN, no-auth model.
- Not reproduced end-to-end: no live camera or multi-camera 30-min run in the
  review environment; the concurrency races are argued from interleavings, not
  observed failures.
</content>
