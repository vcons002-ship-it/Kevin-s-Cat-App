# Handoff: audit findings & the fixes that landed

**Date:** 2026-07-09 · **`main` @ 0.51.0** · Pick-up-cold summary of the recent
review pass — what was audited, what got fixed and merged, and what's still open.

> Companion docs: the full findings live in
> [`2026-07-08-memory-leak.md`](2026-07-08-memory-leak.md) and
> [`2026-07-09-full-code-audit.md`](2026-07-09-full-code-audit.md). This file is
> the "what happened / what's left" index over them.

---

## TL;DR

- Two reviews ran: a **memory-leak investigation** (a ~30-min crash) and a
  **full-code audit** (5 dimensions).
- **Fixed & merged to `main` (0.51.0, PR #114):** both memory leaks — capture
  released on reconnect, and the MJPEG stream can no longer spin forever.
- **Documented, NOT yet fixed:** the audit's HIGH/MEDIUM findings (config-save
  500, a status-endpoint race, a credential-masking gap, frontend lifecycle,
  SSRF, class-agnostic NMS, …). Fix order is in the audit doc.
- **Suite:** 410 passing. No CRITICAL issues anywhere; fundamentals (XSS, YAML,
  secrets, geometry math) are clean.

---

## Part 1 — What was FIXED and merged (0.51.0)

Both from the memory-leak review. Verified by 6 new tests; full suite 410 green.

### Fix 1 — camera capture released on every reconnect
- **Was:** an RTSP/FFmpeg `cv2.VideoCapture` context (native sockets + decoder
  buffers, not reliably freed by `__del__`) was dropped with `self._cap = None`
  on every reconnect *without* `.release()` → leaked one context per reconnect →
  native-memory OOM at a predictable time (invisible to the Python heap).
- **Now:** `PersonDetector._release_cap()` (`detector.py`) releases before every
  drop/reopen. Routed through it: both failed-read paths (sync + smooth grab
  thread) and `_ensure_cap`'s stale-handle reopen. `release()` reuses it.
- **Tests:** `tests/test_capture_release.py`.

### Fix 2 — MJPEG stream can't spin forever
- **Was:** a client disconnect is only raised at a `yield`; when `live_jpeg`
  returned `None` or the frame version stalled, `frames()` busy-looped without
  yielding, so its `threaded=True` worker thread spun until "Stop watching" —
  leaking a thread + closure per stalled/abandoned connection.
- **Now:** `frames()` (`webapp.py`) re-emits the current frame on a heartbeat
  cadence, ends a stream that never got a frame after a timeout, and is wrapped
  in `try/finally`. Tunable via `_STREAM_HEARTBEAT_S` /
  `_STREAM_NO_FRAME_TIMEOUT_S`.
- **Tests:** `tests/test_live_feed.py` (stream ends with no frame; heartbeat
  re-emit).

### ⚠️ Still owed on the leaks
The fixes address the leaks the review *identified*. **Confirming they resolve
the specific 30-min crash still needs a runtime check on the real camera** — the
review environment had no live RTSP feed. Per the memory-leak doc's "how to
confirm": log process RSS, open-FD count, and `len(threading.enumerate())` once a
minute for the first 30 minutes. RSS+FD climbing with a flat Python heap ⇒ Fix 1
was the one; thread count climbing ⇒ Fix 2.

---

## Part 2 — What was AUDITED but NOT yet fixed

Full detail + file:line + repro in `2026-07-09-full-code-audit.md`. No CRITICAL
issues. Several are **regressions from the 0.48–0.50 auto-save/hot-reload work**.
Ranked, with the suggested fix order:

| # | Sev | Finding | Location |
|---|-----|---------|----------|
| H1 | HIGH | `_coerce` raises on blank/`None` numeric → **HTTP 500 on routine saves** (every control auto-saves now). Proven at runtime. | `config.py:284-298` |
| H2 | HIGH | `last_scan()` iterates `_scan_last` while workers insert keys → **intermittent 500** on the 1.2 s `/api/cats` poll. | `loop.py:375` vs `:743` |
| H3 | HIGH | One failed request during `init()` **bricks the UI** (no catch, intervals never start). | `app.js:1860-1874` |
| M1 | MED | **Inline URL credentials leak** via `GET /api/config` + `/api/cameras/saved` (only the `*_password` fields are masked). Violates the "mask everywhere" invariant. | `webapp.py:937,946` |
| M2 | MED | **SSRF** / localhost-pivot via attacker-set `camera_url` into FFmpeg. | `detector.py`, `/api/preview` |
| M3/M4 | MED | Frontend auto-save: serializes a half-populated form during the init gap; no in-flight guard → out-of-order POSTs lose updates. | `app.js:560-603` |
| M5 | MED | **Class-agnostic NMS** drops a cat overlapping a person (the exact case the app cares about). | `yolo.py:666` |
| M6 | MED | `config.example.yaml` missing 17 dataclass fields (find-my-cat, live tiling, motion params, …). | `config.example.yaml` |
| M7 | MED | List/dict config fields get no coercion; a scalar is stored + iterated char-by-char. | `config.py` `_coerce` |
| M8 | MED | `stop()` ignores its join timeout → false "stopped", then won't restart. | `loop.py:193-204` |

Plus 14 LOW items (TensorRT VRAM cleanup, non-atomic manifest write, `sound_file`
traversal, snapshot existence oracle, Host-header/DNS-rebind, polling stacking,
optimistic toggles, discovery socket leaks, doc test-count drift, …).

**Suggested fix order (fastest ROI):** H1 → H2 → M1 (all small, each a
one-function fix + test), then the frontend cluster (H3/M3/M4), then M5 (per-class
NMS), then M2/Host-header, then docs (M6, README count).

### Top-5 test gaps (from the audit)
1. **Camera reconnect / read-failure recovery** — `_ensure_cap` is stubbed in
   every detector test. (Note: Fix 1 above added `test_capture_release.py`, which
   now covers the *release* path — but the full `ok=False → backoff → reopen`
   loop and the `CameraError` open-failure branch are still undriven.)
2. `_coerce` on `""`/`None`/scalar-for-list (H1/M7).
3. `camera_targets()` duplicate/whitespace names, non-dict entries.
4. Discovery internals (ONVIF/WSDiscovery/Cast parsing).
5. Provision download/verify success path.

### Verified SAFE (don't re-investigate)
XSS (consistent `esc()` + `textContent`); `moondream_api_key`/`camera_password`
never reach the browser (except the M1 inline-URL path); YAML `safe_load`;
config writes allowlisted, can't brick `load()`; uploads hardened
(`secure_filename` + ext allowlist + max size); `send_from_directory`
traversal-safe; **all geometry/decode math** (letterbox, ROI offset, crop
mapping, velocity, NMS IoU); most concurrency (`_detectors`/`_viewing`/`_live_lock`
patterns) — `_scan_last` (H2) is the lone exception; `run.py` no debug mode; no
secrets tracked in git.

---

## State at handoff

- **Version:** 0.51.0. **`main` @ `afc7746`** (PR #114 merged). `Dev` synced.
- **Tests:** 410 passing (`./venv/bin/python -m pytest -q`, ~3 min).
- **Working tree:** clean.
- **Open issues on GitHub:** the #91–#106 set remains "open" but is all shipped
  (maintainer merges PRs without closing issues); the audit findings above are
  **not** filed as issues — this doc + the audit doc are the record.

## Next actions (pick any)
1. **Runtime-confirm the leak fix** on the real camera (RSS/FD/thread trace).
2. **Land H1 + H2 + M1** — three small, high-value fixes with tests (recommended
   next slice).
3. Frontend lifecycle hardening (H3/M3/M4).
4. Housekeeping: `config.example.yaml` (M6), README test count (L13).
</content>
