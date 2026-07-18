# Review: memory leak causing ~30-minute runtime crash

**Date:** 2026-07-08 · **Reviewed against:** `main` @ 0.50.0 · **Status:** findings only (no fix applied yet)

**Reported symptom:** the app crashes after ~30 minutes of continuous runtime,
consistent with a memory leak.

**Method:** four parallel read-only code audits (loop/detector hot path;
trail/fusion/ring buffers; webapp streaming + caster threads; the bounded
stores), plus first-hand reads of the per-frame path and an **empirical
6,000-frame `tracemalloc` probe** driving the detector's hot path with a stubbed
net and fake capture.

---

## Executive summary

- The **per-frame detection path is clean** and **every in-memory store is
  properly bounded** — confirmed both by audit and empirically (6,000 frames →
  +84 GC objects, flat memory).
- Because the steady per-frame path does not grow, a ~30-minute OOM must come
  from an **event-driven path**. Two independent **HIGH-severity** leaks fit,
  both real bugs in always-on paths:
  1. **`cv2.VideoCapture` dropped without `.release()` on every camera reconnect**
     — a native (FFmpeg) leak, invisible to the Python heap.
  2. **MJPEG `/api/stream` generator leaks a thread + closure** per stalled or
     abandoned connection (no `try/finally`; disconnect only detected at `yield`).
- Leading hypothesis: **#1**, because the flat Python heap points to a native
  leak, and RTSP cameras reconnect on a roughly regular cadence (→ predictable
  crash time). #2 is a genuine second leak to fix regardless, most likely if the
  crash correlates with the live feed being open.

---

## 🔴 Finding 1 (HIGH) — VideoCapture not released on reconnect

**Files/lines:** `d20app/detector.py:1276`, `:1176`, `:568–590`

On a failed read the capture reference is nulled to force a reconnect, but the
old object is never released:

```python
# detector.py:1273-1276  (synchronous read path)
ok, frame = cap.read()
if not ok or frame is None:
    self._read_fails += 1
    self._cap = None        # force a reconnect next call  ← old cap never .release()d
```

```python
# detector.py:1176  (smooth-feed grab thread — same pattern)
self._cap = None            # force a reconnect next iteration
```

```python
# detector.py:568-590  (_ensure_cap — reopen site)
if self._cap is None or not self._cap.isOpened():
    cap = _open_capture(self.source)
    ...
    self._cap = cap         # prior object (if any) overwritten, never released
```

**Why it matches a ~30-min crash.** These are RTSP/HTTP streams opened via
`cv2.CAP_FFMPEG` (`_open_capture`, forced TCP, 5 s timeout). Each FFmpeg capture
context holds sockets, decoder threads, and large native decode buffers.
`cv2.VideoCapture.__del__` does **not** reliably tear down the FFmpeg/RTSP
context on garbage collection, so every reconnect abandons a heavy native
allocation. RTSP cameras drop/reconnect on a roughly regular cadence; leaked
contexts accumulate at a steady rate → native-memory OOM at a predictable time.

**Key corroboration:** this leak lives in **native memory**, which is exactly
why the empirical Python-level probe (below) shows a flat heap while the process
would still grow.

**Secondary, same family (MED/LOW):** `detector.py:1408–1413` (`release`) and
`:1211–1220` (`_apply_smooth`) intentionally return early and leak the capture
when the grab thread is wedged in a blocking `read()` (there's an explicit
`log.warning("... leaking the capture")`). Bounded per incident, but compounds #1.

---

## 🔴 Finding 2 (HIGH) — MJPEG stream generator leaks a thread + closure per stalled/abandoned connection

**File/lines:** `d20app/webapp.py:1080–1100`

```python
def frames():
    last_ver = -1
    while loop.is_running():
        loop.note_viewing(name)
        ver = loop.live_version(name)
        if ver != last_ver:
            jpeg = loop.live_jpeg(name, trail=trail, last_known=last_known)
            if jpeg is not None:
                last_ver = ver
                yield (... + jpeg + ...)
        time.sleep(0.03)          # only exit is loop.is_running(); no try/finally
return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
```

**Why it leaks.**
- No `try/finally`; a client disconnect is only observable **at a `yield`** (the
  write is where Werkzeug raises the broken-pipe / `GeneratorExit`).
- When `live_jpeg` returns `None` (camera offline / warming up) or the frame
  version stalls, the `yield` branch is never reached — the generator busy-loops
  `note_viewing → sleep(0.03)` forever.
- The server runs `threaded=True` (`run.py:73`), so each connection is a
  dedicated daemon thread that spins until "Stop watching."
- The GUI reopens the stream on every `trail` / `last_known` / `camera` toggle
  (all query params), and browsers reconnect a dropped MJPEG `<img>`. Each
  reconnect that lands in a no-yield state strands a thread + generator closure
  (pinning `loop`, `name`, the last `jpeg`). Accumulates with runtime.

Leaks Python threads + memory — most likely the culprit if the crash correlates
with the live feed being open / overlay toggling.

---

## 🟡 Secondary findings (not the 30-min crash, worth fixing)

- **`_test_detectors` cache never evicted** — `webapp.py:64`
  (`(model, accelerator) → PersonDetector`, no cap). Bounded by the finite model
  set, so it's a one-way climb (every model ever exercised stays pinned in
  RAM/VRAM), not a steady leak. Populated by the Test tool, benchmark sweeps,
  find-cat, and the escalation ladder.
- **Trail episode reset never fires under *sustained* motion** — `trail.py:184`.
  Behavioral smell only: `self._last_motion_ts` refreshes every frame, so the
  >30 s stillness gap never elapses in a busy scene and the reset never runs.
  Because `_ts_buf` is a fixed-size array and `_path` is independently capped, it
  degrades trail quality / render cost, **not** memory.

---

## ✅ Ruled out (verified bounded)

| Area | Evidence |
|---|---|
| ActivityLog | `deque(maxlen=1000)` — `activitylog.py:37`; entries hold no bytes |
| CatTracker | `deque(maxlen=500)`, trimmed every `record()`; stores image **filename**, not bytes — `cats.py:96,168` |
| SnapshotStore | writes JPEG to disk, prunes to 60 files; no in-memory byte store — `snapshots.py:35,56` |
| Trail `_ts_buf` / `_path` | fixed-size array reused in place; `_path` capped at `PATH_MAX_POINTS=400` — `trail.py:122,208,216` |
| Fusion `_tracks` | capped at `MAX_TRACKS=6`, hits pruned to window unconditionally every `update()` — `fusion.py:110,117` |
| Detector ring buffer | `deque(maxlen=8)`, ≥1 s spacing, ≤480 px — `detector.py:427,1034` |
| Frame/box retention | `_last_frame`/`_live_frame`/`_last_boxes`/`_person_boxes`/`_fused_hit` are single-value reassignments, not lists — `detector.py:1142,1356,1375,1382` |
| Config hot-reload | in-place; net dropped only on real model/accel change, not per reload — `detector.py:496–555`, `loop.py:637–662` |
| Per-camera loop dicts | `_scan_last`/`_cat_boost`/`_viewing`/`_cam_status` keyed by camera name (fixed small set) |
| moondream cache | keyed `(mode,name)`, ≤4 entries, loaded once — `moondream.py:113,219` |
| escalation | pure per-call functions; returned dicts hold boxes, not frame views — `escalation.py:252,286` |
| Caster | connections cached + reused + `_drop`ped (`stop_discovery` + `disconnect`); `start_keepalive` calls `stop_keepalive` first (no thread stacking) — `caster.py:142,188` |
| Session/report caches | `_TEST_SESSIONS` (OrderedDict cap 1000), `_BENCHMARKS` (cap 40), `_BENCH_CANCEL` (discarded) — `webapp.py:1449,1530,1646` |

---

## Empirical probe (per-frame path)

Drove `read_and_detect(detect=True)` for 6,000 frames with a stubbed net + fake
capture (alternating still/moving frames so the trail, motion, ring, and fusion
paths all exercise), measuring `tracemalloc` + `gc` object counts after a
200-frame warmup.

```
frames driven: 6000
gc objects: 23160 -> 23244  (delta 84)      # flat — noise
top allocation growth: detector.py:1034 (the ring buffer's 8 frames), then nil
ring len: 8                                 # bounded
trail _path len: 0 | _ts_buf shape: (480, 640)   # bounded / fixed
```

**Conclusion:** the Python-level per-frame path is flat. This both rules out a
per-frame Python leak and points at a **native** leak (Finding 1) as the most
likely cause of the process OOM.

---

## How to confirm which leak (runtime observation)

> **Outcome (2026-07-17):** confirmed on the real cameras — the 0.51.0 fixes held over a
> **10+ hour live run on all cameras; the reconnect / steady-state leak is resolved.** A
> separate potential leak under heavy live reconfiguration (rapid hot-reload churn) remains
> unverified and low-priority. See `docs/reviews/2026-07-09-audit-fixes-handoff.md`.

The two findings are distinguishable by simple runtime instrumentation:

- **Finding 1 (native / FFmpeg):** crashes even with the browser live feed
  **closed**; process **RSS and file-descriptor/thread count climb** while the
  Python heap (`tracemalloc`) stays flat. Correlates with an RTSP camera that
  reconnects periodically.
- **Finding 2 (Python threads):** crash correlates with the live feed being
  **open** / overlay or camera toggling; `len(threading.enumerate())` climbs.

Quick probe: log `RSS`, open FD count, and `len(threading.enumerate())` once a
minute for the first 30 minutes.

---

## Recommended fixes

1. **Finding 1 (do first):** add a `_release_cap()` helper that calls
   `self._cap.release()` (guarded) before every `self._cap = None` / reopen —
   the three sites at `detector.py:1276`, `:1176`, and inside `_ensure_cap`
   (568–590). Small, safe, high value. Add a regression test that drives a
   reconnect and asserts the prior capture was released (inject a fake capture
   with a `released` flag).
2. **Finding 2:** wrap `frames()` in `try/finally`, and either yield a periodic
   heartbeat part or cap the no-yield wait so a client disconnect is detected
   even when frames stall; ensure the generator returns when the camera is gone.
   Test with a fake `loop` whose `live_jpeg` returns `None` and assert the
   generator terminates / cleans up.
3. **Secondary:** cap/evict `_test_detectors` (LRU by (model, accelerator), or
   drop entries not used recently).

---

## Appendix — scope & confidence

- **Confidence:** HIGH that both findings are real bugs (static evidence is
  unambiguous); the *ranking* between them for THIS crash is a hypothesis, since
  the runtime environment (camera type, whether the feed was open, an RSS-over-
  time trace) was not available during the review.
- **Not reproduced end-to-end:** no live RTSP camera or 30-minute run was
  available in the review environment; the per-frame probe is the only empirical
  measurement. Confirming the culprit needs the runtime observation above.
</content>
