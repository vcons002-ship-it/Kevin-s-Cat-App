"""Local VLM (moondream) cat-presence tester — a measurement tool, not a live path.

YOLO/SSD answer "*where* is the cat" (boxes) and miss small/distant/backlit cats;
open-vocabulary detectors false-fire on cat-shaped decoys. A small vision-language
model takes a different angle: *reason* about the whole image to answer "*is* there a
cat". This module wraps `moondream` for a single **query** pass on one frame, so its
answer + reasoning can be measured on real frames before wiring it into any escalation
workflow (issue #48).

**Local inference needs a supported GPU.** moondream's local engine (Photon) requires
an Ampere-or-newer NVIDIA GPU or Apple Silicon — there is **no CPU path** (#52). On a
box without one this raises a clear error rather than the raw Photon failure. Weights are
downloaded once from Hugging Face (authenticated by an API key) and cached locally;
**per-query inference is local** — images don't leave the machine.

`moondream` is an **optional dependency**; the local model is selected **by name**
(`moondream2` or `moondream3-preview`), not a file path — Photon manages its own weight
cache (set `HF_HOME` to redirect it). The API key comes from the GUI (stored in config)
or `MOONDREAM_API_KEY`, and authenticates both the one-time local weight download and
cloud inference.

Two **modes** (#59), same ``md.vl()`` interface:
  * **local** (default) — ``moondream2`` on the box's GPU. Private (frames never leave),
    ~0.3 s/query, fits an 8 GB card. The deployable path.
  * **cloud** — ``md.vl(api_key=…)`` with no ``local=True`` hits moondream's hosted API
    (currently Moondream 3, the most accurate on hard/decoy frames). **Sends images
    off-device** and costs per call — an informed, opt-in choice, useful for validation.

Loading params are the **validated** ones for an 8 GB RTX 3070 (#59): the default
auto-sized KV cache OOMs, so we pin ``max_batch_size`` and ``kv_cache_pages`` to a window
that both loads (~4.6 GB) and serves.

The response **parser** (:func:`parse_vlm_response`) is pure and unit-tested; the model
call needs a GPU/key (local) or network (cloud), so it's exercised where the model lives.
"""

from __future__ import annotations

import os
import re
import threading
import time

# A plain presence question — yes/no plus a short explanation. moondream's self-reported
# confidence proved meaningless (it says "0-100%", "not sure but yes"), so we don't ask
# for or parse a number; the reasoning text is kept purely as information (#54).
DEFAULT_PROMPT = "Is there a cat in this image? Answer yes or no, then briefly explain."

# The temporal-mosaic question (#68): the "image" is a numbered grid of frames from
# ONE camera, oldest → newest (built by escalation.frame_mosaic). Untested premise,
# honestly: whether moondream reasons well over grid images is exactly what the
# tester exists to measure — validate on the NAS before trusting it.
TEMPORAL_PROMPT = (
    "This image is a grid of numbered frames from one camera, in time order "
    "(1 = oldest; the newest is marked 'now'). Did a cat appear or move through the "
    "scene? Answer yes or no, then say in which numbered frame(s) you see it.")

# Local model names Photon can load (not file paths). moondream2 (2B) fits an 8 GB card
# comfortably; moondream3-preview (9B) does NOT fit 8 GB locally (Photon has no weight
# quantisation) — so M3 is offered as a **cloud** choice, not local, on small cards (#59).
MODELS = ("moondream2", "moondream3-preview")
DEFAULT_MODEL = "moondream2"

MODES = ("local", "cloud")
DEFAULT_MODE = "local"

# Moondream's yes/no is non-deterministic — the same frame can flip run-to-run, and the
# frames that wobble are the genuinely-hard ones (#60). Running N passes and taking the
# majority vote averages the wobble out, and the **vote ratio is the honest confidence**
# (agreement across passes) that replaces the model's meaningless self-report.
DEFAULT_PASSES = 3
MAX_PASSES = 9          # a sane ceiling so one request can't run forever

# Validated local-load params for an 8 GB RTX 3070 (#59). The default auto-sized KV cache
# (kv_cache_pages=None) OOMs even at batch 2 (CUBLAS_STATUS_ALLOC_FAILED); 256 pages loads
# but can't serve ("insufficient KV cache capacity"); 2048 loads (~4.6 GB) AND serves at
# batch 4. Bounded-but-adequate defaults; tune for a bigger card if you want more batch.
MAX_BATCH_SIZE = 4
KV_CACHE_PAGES = 2048

# The coherent (mode, model) choices the GUI offers — the single source so the picker
# can't drift from what the backend accepts (#59). Each says whether it leaves the box.
VLM_CHOICES = [
    {"value": "local:moondream2", "mode": "local", "model": "moondream2",
     "label": "Moondream 2 — local (private, ~0.3s, fits 8 GB)", "off_device": False},
    {"value": "cloud:moondream3-preview", "mode": "cloud", "model": "moondream3-preview",
     "label": "Moondream 3 — cloud API (most accurate, but SENDS IMAGES OFF-DEVICE)",
     "off_device": True},
]
DEFAULT_CHOICE = "local:moondream2"

_MODELS: dict = {}                 # (mode, name) -> loaded moondream model (load is slow; cache)
_LOAD_LOCK = threading.Lock()


def is_available() -> bool:
    """True if the ``moondream`` package is importable (a GPU + key may still be needed)."""
    import importlib.util

    return importlib.util.find_spec("moondream") is not None


def majority_vote(answers: list) -> dict:
    """Pure majority vote over a list of ``"yes"``/``"no"``/``None`` pass answers (#60).

    Returns ``{"verdict", "yes", "no", "decisive", "parsed", "unanimous", "borderline"}``:
    ``verdict`` is the winning side (``None`` on a tie or no parsed answers — both
    genuinely ambiguous), ``decisive`` is the winning vote count, ``parsed`` the number of
    yes/no answers, ``unanimous`` True only when every pass agreed, and ``borderline`` True
    whenever the verdict isn't unanimous (a split worth a human look)."""
    yes = sum(1 for a in answers if a == "yes")
    no = sum(1 for a in answers if a == "no")
    parsed = yes + no
    total = len(answers)
    if parsed == 0 or yes == no:
        verdict, decisive = None, max(yes, no)
    else:
        verdict = "yes" if yes > no else "no"
        decisive = max(yes, no)
    return {"verdict": verdict, "yes": yes, "no": no, "decisive": decisive,
            "parsed": parsed,
            "unanimous": verdict is not None and decisive == total and parsed == total,
            "borderline": verdict is None or decisive < total}


def parse_vlm_response(text: str) -> dict:
    """Best-effort yes/no extraction from a VLM reply, with the full text kept as the
    reason. Returns ``{"answer": "yes"|"no"|None, "reason": str, "parsed": bool}``;
    ``parsed`` is True only when a yes/no was found — we never invent an answer (#52/#54).
    No confidence is parsed (moondream's self-report is noise)."""
    text = (text or "").strip()
    m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    answer = m.group(1).lower() if m else None
    return {"answer": answer, "reason": text, "parsed": answer is not None}


def _require_gpu() -> str:
    """Return the local accelerator name, or raise a clear error when there's none —
    moondream's Photon engine has no CPU path (#52)."""
    try:
        import torch
    except Exception:               # noqa: BLE001 — torch ships with moondream
        return "gpu"                # can't check; let Photon decide
    if torch.cuda.is_available():
        return "cuda"
    if getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "moondream local inference needs a CUDA (Ampere+) or Apple-Silicon GPU — there "
        "is no CPU path. Run the VLM tester on a machine with a supported GPU.")


_NO_KEY_MSG = (
    "No Moondream API key set — paste your key in the VLM tester (it authenticates the "
    "one-time local weight download and cloud inference), or set MOONDREAM_API_KEY.")
_NO_PKG_MSG = (
    "The VLM tester needs the 'moondream' package — re-run setup (say yes to the VLM "
    "tester) or: pip install moondream")


def _friendly_model_error(exc: Exception, model: str) -> RuntimeError:
    """Map Photon's raw GPU failures to an actionable message (#59) — OOM / KV-cache
    stall / cuBLAS alloc — instead of surfacing a bare traceback."""
    s = str(exc)
    low = s.lower()
    if "insufficient kv cache" in low or "scheduler stalled" in low:
        return RuntimeError(
            f"moondream {model}: KV cache too small to serve this prompt — raise "
            f"kv_cache_pages (currently {KV_CACHE_PAGES}).")
    if ("out of memory" in low or "cublas_status_alloc_failed" in low
            or "alloc_failed" in low or "cuda error" in low):
        return RuntimeError(
            f"moondream {model} ran out of GPU memory — lower max_batch_size (currently "
            f"{MAX_BATCH_SIZE}), or this model doesn't fit this GPU's VRAM "
            "(moondream3-preview needs 12 GB+ locally; use the cloud mode on a smaller card).")
    return RuntimeError(f"moondream {model} failed: {s}")


def _load_model(model_name: str, api_key: str | None, mode: str = DEFAULT_MODE):
    """Load (and cache) a moondream model via Photon, for the given ``mode``.

    * ``local`` — ``md.vl(api_key=…, local=True, model=name, max_batch_size, kv_cache_pages)``
      with the validated params. *Not* ``md.vl(model=path)``, which has no ``model`` param
      and silently routes to the cloud (401) instead of running locally (#52).
    * ``cloud`` — ``md.vl(api_key=…)`` (no ``local=True``) hits the hosted API; needs no
      GPU, but sends images off-device (#59)."""
    try:
        import moondream as md
    except Exception as exc:        # noqa: BLE001 — optional dependency
        raise RuntimeError(_NO_PKG_MSG) from exc

    key = api_key or os.environ.get("MOONDREAM_API_KEY")
    if not key:
        raise RuntimeError(_NO_KEY_MSG)

    mode = mode if mode in MODES else DEFAULT_MODE
    name = model_name or DEFAULT_MODEL
    with _LOAD_LOCK:
        cache_key = (mode, name)
        m = _MODELS.get(cache_key)
        if m is None:
            try:
                if mode == "cloud":
                    m = md.vl(api_key=key)          # hosted API (Moondream 3); no GPU
                else:
                    _require_gpu()
                    m = md.vl(api_key=key, local=True, model=name,
                              max_batch_size=MAX_BATCH_SIZE, kv_cache_pages=KV_CACHE_PAGES)
            except RuntimeError:
                raise                               # already actionable (_require_gpu / key)
            except Exception as exc:                # noqa: BLE001 — map OOM/KV/cuBLAS
                raise _friendly_model_error(exc, name) from exc
            _MODELS[cache_key] = m
        return m


def preflight(api_key: str | None = None, mode: str = DEFAULT_MODE) -> None:
    """Raise the same actionable error :func:`query_image` would, *before* running a
    whole batch — so a batch fails fast with one clear message (missing package / key /
    GPU) instead of erroring once per image (#54). Cloud mode needs no GPU (#59)."""
    if not is_available():
        raise RuntimeError(_NO_PKG_MSG)
    if not (api_key or os.environ.get("MOONDREAM_API_KEY")):
        raise RuntimeError(_NO_KEY_MSG)
    if (mode if mode in MODES else DEFAULT_MODE) == "local":
        _require_gpu()


def _to_pil(frame_bgr):
    """BGR ndarray → RGB PIL image (what moondream's encoder expects)."""
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def query_image(frame_bgr, prompt: str = DEFAULT_PROMPT, model: str = DEFAULT_MODEL,
                api_key: str | None = None, mode: str = DEFAULT_MODE) -> dict:
    """Run one moondream **query** pass on ``frame_bgr`` and return a result dict:

    ``raw`` (full response text — always shown), the parsed ``answer``/``reason``/
    ``parsed`` from :func:`parse_vlm_response`, ``load_ms`` (one-time model load, 0 if
    cached) split from ``query_ms`` (per-frame), and the ``prompt``/``model``/``mode``/
    ``device`` used. Raises ``RuntimeError`` with an actionable message when moondream, a
    GPU (local mode), the key, or VRAM is unavailable."""
    mode = mode if mode in MODES else DEFAULT_MODE
    name = model or DEFAULT_MODEL
    cached = (mode, name) in _MODELS

    t0 = time.perf_counter()
    m = _load_model(name, api_key, mode)
    load_ms = 0.0 if cached else round((time.perf_counter() - t0) * 1000.0, 1)

    img = _to_pil(frame_bgr)
    t1 = time.perf_counter()
    try:
        encoded = m.encode_image(img)
        result = m.query(encoded, prompt or DEFAULT_PROMPT)
    except Exception as exc:        # noqa: BLE001 — surface model/runtime errors clearly
        raise _friendly_model_error(exc, name) from exc
    query_ms = round((time.perf_counter() - t1) * 1000.0, 1)

    raw = result.get("answer", "") if isinstance(result, dict) else str(result)
    parsed = parse_vlm_response(raw)
    device = "cloud" if mode == "cloud" else _require_gpu()
    return {
        "raw": raw, "load_ms": load_ms, "query_ms": query_ms,
        "prompt": prompt or DEFAULT_PROMPT, "model": name, "mode": mode, "device": device,
        **parsed,
    }


def detect_regions(frame_bgr, obj: str = "cat", model: str = DEFAULT_MODEL,
                   api_key: str | None = None, mode: str = DEFAULT_MODE) -> dict:
    """Run moondream's **detect** mode ("where are the ``obj``s?") on one frame.

    Returns ``{"objects": [{"x_min","y_min","x_max","y_max"}, ...], "detect_ms",
    "load_ms", "model", "mode", "device"}``. Coordinates are **normalized [0,1]**
    (moondream 1.3.0 ``types.py``; same shape on the local Photon and cloud paths) —
    map them with :func:`d20app.escalation.map_normalized_box`.

    Used by the escalation ladder (#66) to propose regions; a bare region is never
    trusted alone — the ladder always confirms with YOLO or a voted query, because
    open-vocabulary detect false-fires on cat-shaped decoys. NB: written against the
    installed package's types and verified with mocks in CI; **unvalidated on real
    GPU hardware until the NAS checklist runs** — treat orientation/quality of real
    detect output as unconfirmed until then.
    """
    mode = mode if mode in MODES else DEFAULT_MODE
    name = model or DEFAULT_MODEL
    cached = (mode, name) in _MODELS

    t0 = time.perf_counter()
    m = _load_model(name, api_key, mode)
    load_ms = 0.0 if cached else round((time.perf_counter() - t0) * 1000.0, 1)

    img = _to_pil(frame_bgr)
    t1 = time.perf_counter()
    try:
        result = m.detect(img, obj)
    except Exception as exc:        # noqa: BLE001 — map OOM/KV/cuBLAS to advice
        raise _friendly_model_error(exc, name) from exc
    detect_ms = round((time.perf_counter() - t1) * 1000.0, 1)

    objects = result.get("objects", []) if isinstance(result, dict) else []
    device = "cloud" if mode == "cloud" else _require_gpu()
    return {"objects": list(objects), "detect_ms": detect_ms, "load_ms": load_ms,
            "model": name, "mode": mode, "device": device}


def query_image_voted(frame_bgr, prompt: str = DEFAULT_PROMPT, model: str = DEFAULT_MODEL,
                      api_key: str | None = None, mode: str = DEFAULT_MODE,
                      passes: int = DEFAULT_PASSES) -> dict:
    """Query the same frame ``passes`` times and take the **majority vote** (#60).

    moondream's yes/no wobbles run-to-run on hard frames, so a single pass is unreliable.
    Returns the single-pass result shape (so callers/UI keep working) with ``answer`` set
    to the majority verdict, plus voting fields: ``votes`` (``{"yes","no"}`` counts),
    ``passes``, ``ratio`` (e.g. ``"4/5"`` — the **honest confidence**, agreement across
    passes), ``unanimous`` and ``borderline`` flags, and ``per_pass`` (each pass's answer).
    ``reason``/``raw`` are taken from a pass that matched the verdict, so the shown
    reasoning explains the verdict. ``passes <= 1`` is exactly the single-pass behaviour."""
    n = max(1, min(int(passes or 1), MAX_PASSES))
    results = [query_image(frame_bgr, prompt=prompt, model=model, api_key=api_key, mode=mode)
               for _ in range(n)]
    answers = [r["answer"] for r in results]
    vote = majority_vote(answers)

    # Show the reasoning from a pass that agrees with the verdict (or the first pass).
    rep = next((r for r in results if r.get("answer") == vote["verdict"]), results[0])
    total_ms = round(sum(r.get("query_ms", 0.0) for r in results), 1)
    return {
        "raw": rep.get("raw", ""), "answer": vote["verdict"], "reason": rep.get("reason", ""),
        "parsed": vote["verdict"] is not None,
        "votes": {"yes": vote["yes"], "no": vote["no"]},
        "passes": n, "ratio": f"{vote['decisive']}/{n}",
        "unanimous": vote["unanimous"], "borderline": vote["borderline"],
        "per_pass": answers,
        "load_ms": results[0].get("load_ms", 0.0), "query_ms": total_ms,
        "prompt": prompt or DEFAULT_PROMPT, "model": model or DEFAULT_MODEL,
        "mode": mode if mode in MODES else DEFAULT_MODE, "device": results[0].get("device", ""),
    }
