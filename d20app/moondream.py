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

`moondream` is an **optional dependency**; the model is selected **by name**
(`moondream2` or `moondream3-preview`), not a file path — Photon manages its own weight
cache (set `HF_HOME` to redirect it). The API key comes from `MOONDREAM_API_KEY`.

The response **parser** (:func:`parse_vlm_response`) is pure and unit-tested; the model
call needs a GPU + key, so it's exercised where the model lives, not in CI.
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

# Local model names Photon can load (not file paths). moondream2 (2B) fits an 8 GB card
# comfortably; moondream3-preview (9B MoE) needs more VRAM.
MODELS = ("moondream2", "moondream3-preview")
DEFAULT_MODEL = "moondream2"

_MODELS: dict = {}                 # model name -> loaded moondream model (load is slow; cache)
_LOAD_LOCK = threading.Lock()


def is_available() -> bool:
    """True if the ``moondream`` package is importable (a GPU + key may still be needed)."""
    import importlib.util

    return importlib.util.find_spec("moondream") is not None


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


def _load_model(model_name: str, api_key: str | None):
    """Load (and cache) a local moondream model by **name** via Photon.

    Uses the documented local invocation ``md.vl(api_key=…, local=True, model=…)`` —
    *not* ``md.vl(model=path)``, which has no ``model`` param and silently routes to the
    cloud (401) instead of running locally (#52)."""
    try:
        import moondream as md
    except Exception as exc:        # noqa: BLE001 — optional dependency
        raise RuntimeError(
            "The local VLM tester needs the 'moondream' package — re-run setup "
            "(say yes to the VLM tester) or: pip install moondream"
        ) from exc

    key = api_key or os.environ.get("MOONDREAM_API_KEY")
    if not key:
        raise RuntimeError(
            "moondream local needs an API key (it authenticates the one-time weight "
            "download from Hugging Face; inference is then local). Set MOONDREAM_API_KEY.")
    _require_gpu()

    name = model_name or DEFAULT_MODEL
    with _LOAD_LOCK:
        m = _MODELS.get(name)
        if m is None:
            m = md.vl(api_key=key, local=True, model=name)
            _MODELS[name] = m
        return m


def _to_pil(frame_bgr):
    """BGR ndarray → RGB PIL image (what moondream's encoder expects)."""
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def query_image(frame_bgr, prompt: str = DEFAULT_PROMPT, model: str = DEFAULT_MODEL,
                api_key: str | None = None) -> dict:
    """Run one moondream **query** pass on ``frame_bgr`` and return a result dict:

    ``raw`` (full response text — always shown), the parsed ``answer``/``reason``/
    ``parsed`` from :func:`parse_vlm_response`, ``load_ms`` (one-time model load, 0 if
    cached) split from ``query_ms`` (per-frame), and the ``prompt``/``model``/``device``
    used. Raises ``RuntimeError`` with an actionable message when moondream, a GPU, or
    the key is unavailable."""
    name = model or DEFAULT_MODEL
    cached = name in _MODELS

    t0 = time.perf_counter()
    m = _load_model(name, api_key)
    load_ms = 0.0 if cached else round((time.perf_counter() - t0) * 1000.0, 1)

    img = _to_pil(frame_bgr)
    t1 = time.perf_counter()
    try:
        encoded = m.encode_image(img)
        result = m.query(encoded, prompt or DEFAULT_PROMPT)
    except Exception as exc:        # noqa: BLE001 — surface model/runtime errors clearly
        raise RuntimeError(f"moondream query failed: {exc}") from exc
    query_ms = round((time.perf_counter() - t1) * 1000.0, 1)

    raw = result.get("answer", "") if isinstance(result, dict) else str(result)
    parsed = parse_vlm_response(raw)
    device = _require_gpu()
    return {
        "raw": raw, "load_ms": load_ms, "query_ms": query_ms,
        "prompt": prompt or DEFAULT_PROMPT, "model": name, "device": device,
        **parsed,
    }
