# Live-testing context — findings from running the app on real cameras

**Purpose:** This doc is the bridge between two bodies of work on this repo:
1. **Fable's code audit** (`docs/reviews/2026-07-09-audit-fixes-handoff.md` + the
   memory-leak and full-audit docs) — static review of the code.
2. **Kevin's live testing** — findings from actually running the app on 7 real
   cameras with real cats, which surfaced behavior a static audit can't see.

The two overlap, and once appeared to **disagree** on NMS — but reading the current
code showed that was a mis-attribution, not a real conflict (see the NMS section
below, now resolved). Read this alongside the audit docs; where they seem to conflict,
**verify against the current code before acting** — don't assume either is right.

Kevin is now the developer; the original maintainer reviews PRs. Kevin has the only
live-camera environment, so runtime-confirmation tasks fall to him.

---

## NMS — per-class or class-agnostic? (RESOLVED — two different sites)

**Earlier framing (wrong): "M5 and issue 31 describe the same code (`yolo.py` ~line
666) and contradict."** They don't. Reading the current code settled it — the two
findings are about **two different NMS sites**, and each is correct about its own:

- **Fable's audit M5** → `detect_boxes` single-pass NMS at **`yolo.py:666`**
  (`cv2.dnn.NMSBoxes` over all classes at once). This one is **class-agnostic**: a cat
  overlapping a person at IoU ≥ 0.45 can suppress the cat box. Applies to the untiled
  path and to the *within-tile* pass of the tiled path.
- **Kevin's live finding (issue 31)** → `merge_nms` at **`yolo.py:675`** (NMS call at
  `:692`), which groups boxes by label first. This one is **per-class** (docstring reads
  "Per-class NMS"). It's what lets a strong `dog` box and a weak `cat` box coexist on the
  *same animal* across tiles — which, with dog-as-cat enabled, drives a false
  track-fusion "cat" sighting.

M5's own text confirms this: it flags `:666` and explicitly contrasts it with
`merge_nms (:675), which is deliberately per-class`. The "same code" error was this
doc's — it collapsed the two line numbers into one.

**The two fixes pull opposite directions — don't unify them:**
- M5 wants *less* cross-class suppression at `detect_boxes:666` (stop dropping the cat
  under a person). Group by label there too, or `NMSBoxesBatched`.
- Issue 31 wants *more* merging at fusion time: when the "count dog as cat" toggle is
  on, overlapping dog+cat boxes are the SAME animal and should be MERGED into one strong
  cat *before* fusion — not kept separate. Toggle off = keep separate. This is a targeted
  transform, **not** a change to `merge_nms`'s per-class nature.

---

## Kevin's live-testing findings (issues 30–34, files exist separately)

These came from live use, not static review. Several are bugs Fable's audit didn't
surface because they only appear at runtime.

- **Issue 30 — provisioning regenerates valid models instead of verify-and-adopt.**
  After `git reset`/update, locally-generated models (26x, FP16, engines) get flagged
  "unknown provenance" and the refresh button *rebuilds* them (slow) instead of
  hashing + verifying + adopting them (fast — proven: a manual hash+verify+adopt
  marked all OK in seconds). Also: the manifest is committed, so resets wipe local
  provenance. Fix: verify-and-adopt for unverified-but-valid files; gitignored local
  manifest (`models_manifest.local.json`).

- **Issue 31 — track fusion class conflation** (see NMS section above). Fusion is fed
  label-stripped boxes, so it can't tell weak-cat from weak-dog hits. Includes a
  request for opt-in `fusion_debug` diagnostic logging (per-hit score+class+box,
  travel, coexisting-strong-box flag, structured `fusion_events.jsonl`). NOTE: Kevin
  believes fusion is near-useless for his setup (cats score ~0.93 clean, rarely need
  weak-hit recovery) and may be a *net FP source* — the debug logging is partly to
  decide whether to keep fusion at all.

- **Issue 32 — find-my-cat leaks settings into the LIVE detector + mis-tags boost.**
  Find reuses `_run_test_detection()`, which mutates the shared live detector in place
  (sets `cat_scan_tiling="3x3"`, confidence, etc.) and never restores it → the live
  feed runs tiled for ~10s after a find (the boost window; `boost_detection(cam,10.0)`
  ~webapp.py:1454), then reverts via hot-reload. ALSO: boost sets `force_scan=True` —
  the same flag the still-scan uses — so boost detections get logged as "still-scan"
  when they're really just live detections. KEEP the boost feature and the
  jump-to-last-sighting fallback; FIX the state leak (run at the camera's own settings)
  and the tag (boost = LIVE, not still-scan). Explicit acceptance test in the issue:
  scan finds a cat → jump → the "last known" box is drawn where the SCAN found it,
  even if the live feed on the camera's own settings can't detect a cat there (sourced
  from the recorded sighting, not live re-detection).

- **Issue 33 — "last known location" box shouldn't fade at 30 min**
  (`_LAST_KNOWN_TTL=1800`). It's most useful for a long-still/asleep cat; staleness is
  already labeled with age. Remove/extend the TTL.

- **Issue 34 — Follow mode + user-enableable second live feed.** Feature. "Follow"
  toggle (alongside trail/last/smooth/live) auto-switches the feed to the most-recent
  cat. Optional second feed the user turns on. Core design principle: sticky,
  debounced, per-feed assignment so feeds don't flip-flop (each feed HOLDS its camera
  until it goes quiet for N seconds; feeds don't compete for "most recent"). Primary
  follows the cat, secondary shows previous room. Expected to be tuned live — build the
  reasonable version, the anti-flicker principle is the part to get right.

Earlier issues (7–29) were mostly implemented by the maintainer's AI across 0.43–0.51
(the #91–#106 PR set); the audit handoff confirms those shipped.

---

## Architectural state (what Kevin's benchmarking established)

- **The model tier collapsed to ONE model: YOLO 26x.** TensorRT made 26x cheap enough
  to run everywhere. Established live: 26x untiled+TRT ≈ 18ms; 26x 3×3+TRT ≈ 93ms;
  both dominate 11n in every config. There is no situation that uses 11n on GPU.
  Structure: 26x untiled = fast live gate; 26x 3×3 = thorough scan; VLM = validator.
- **TensorRT engines** are FP16, golden-headed `(1,84,8400)` no-NMS, and carry an
  Ultralytics metadata header that raw TRT can't deserialize (the app strips it).
  Engines are locked to the exact TRT version + GPU.
- **Driver stack:** 610.43.02 / CUDA UMD 13.3 on the NAS (RTX 3070, headless compute,
  no display — driver display-risks are irrelevant). torch 2.12.1+cu130, TensorRT
  11.1.0.106. Survives kernel updates via DKMS (headers must track the kernel).
- **Motion detection weakness (real finding):** the motion verdict is
  `contourArea(largest blob) >= min_area_frac * H * W`, computed after a 5×5
  MORPH_OPEN. At full resolution (2304×1296, ~3M px) this destroys thin/ragged cat
  motion: a cat crossing produced 95,770 raw changed px but only a 3,926-px
  contourArea verdict (erosion ate ~93%, and contourArea undercounts ragged shapes vs
  a 14,396-px bbox). So distant/thin cat motion under-triggers. Motion runs at
  `scan_fps` cadence (default 10; Kevin's cameras are at 5). Potential fix worth
  filing: verdict on raw changed-pixel count or bbox area instead of contourArea,
  and/or lighten erosion. Kevin tuned live to area 0.001 / diff 30 / blob 18 (100% of
  easy cat clips, 0% FP on his null set) — the non-obvious win was raising diff (reject
  faint lighting changes) while lowering area (catch thin motion).

---

## Open runtime-confirmation Kevin owns (has the only live cameras)

- **The memory-leak fix is UNCONFIRMED against the real 30-min crash.** Fable's fixes
  (capture release on reconnect; MJPEG heartbeat/timeout) address the leaks the review
  *identified*, but the review had no live RTSP feed. Per the handoff: log RSS +
  open-FD + `len(threading.enumerate())` once a minute for 30 min. RSS+FD climbing with
  flat Python heap ⇒ Fix 1 (capture); thread count climbing ⇒ Fix 2 (stream). Kevin's
  observed baseline (26x untiled, all cameras) was stable ~1895 MiB VRAM for >1hr, but
  that run didn't necessarily hit the reconnect trigger. Still needs the deliberate
  extended trace.

---

## Fable's audit open items (from the handoff — not yet fixed)

Not re-filed as GitHub issues; the audit docs are the record. Recommended slice from
the handoff: **H1 → H2 → M1** (all small, one-function fixes + tests).

- **H1 (HIGH):** `_coerce` raises on blank/None numeric → HTTP 500 on routine
  auto-saves (`config.py:284-298`). Proven at runtime. This likely connects to Kevin's
  save-coherence observations (issue 25).
- **H2 (HIGH):** `last_scan()` iterates `_scan_last` while workers insert keys →
  intermittent 500 on the 1.2s `/api/cats` poll (`loop.py:375` vs `:743`).
- **H3 (HIGH):** one failed request during `init()` bricks the UI (`app.js:1860-1874`).
- **M1 (MED):** inline URL credentials leak via `/api/config` + `/api/cameras/saved`
  (only `*_password` masked) (`webapp.py:937,946`).
- **M2 (MED):** SSRF via attacker-set `camera_url` into FFmpeg.
- **M3/M4 (MED):** frontend auto-save serializes half-populated form during init gap;
  no in-flight guard → out-of-order POSTs lose updates.
- **M5 (MED):** the NMS finding — SEE THE RECONCILIATION SECTION; contradicts issue 31.
- **M6 (MED):** `config.example.yaml` missing 17 dataclass fields (find-my-cat, live
  tiling, motion params…).
- **M7 (MED):** list/dict config fields get no coercion; scalar stored + iterated
  char-by-char.
- **M8 (MED):** `stop()` ignores join timeout → false "stopped", won't restart.
- Plus 14 LOW items (TensorRT VRAM cleanup, non-atomic manifest write, etc.).

---

## Suggested first-session sequence

1. **Read** this doc + the three audit docs in `docs/reviews/`.
2. **Resolve the NMS contradiction** — read `yolo.py` merge_nms, determine
   per-class vs class-agnostic, reconcile M5 vs issue 31. Investigative, orients you
   in the code, and unblocks both fixes.
3. **Land the small high-value audit slice** H1 → H2 → M1 (each a one-function fix +
   test), which also overlaps Kevin's save-coherence observations.
4. **Kevin runs the leak runtime-confirmation** on the live cameras (his to do).
5. Then pick from issues 30–34 and the motion-verdict finding.

Kevin's style: verify against the actual code/runtime before asserting; he catches
over-eager hypotheses. Give specific, falsifiable claims he can check against reality.
No implicit hardcoding of models/settings from descriptive labels — expose configurable
mechanisms; the maintainer's AI has a recurring habit of hardcoding usage from
descriptive names, which has caused bugs.
