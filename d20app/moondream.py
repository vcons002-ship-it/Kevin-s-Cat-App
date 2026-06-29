"""Local VLM (moondream) cat-presence tester — a measurement tool, not a live path.

YOLO/SSD answer "*where* is the cat" (boxes) and miss small/distant/backlit cats;
open-vocabulary detectors false-fire on cat-shaped decoys. A small vision-language
model takes a different angle: *reason* about the whole image to answer "*is* there a
cat". This module wraps `moondream` for a single **query** pass on one frame, so its
answer/reasoning/latency can be measured on real frames before wiring it into any
escalation workflow (issue #48).

`moondream` is an **optional dependency** (mirrors the openvino/playsound3/gTTS pattern):
the core install stays lean, and this degrades with a clear "pip install moondream"
message when it (or a local model) isn't present. The model itself is multi-GB, so it's
**not committed** — point ``MOONDREAM_MODEL`` (or the request's ``model`` field) at a
local quantized model file. CPU latency is multi-second-to-minute; this is an occasional
evaluation/escalation tool, not an every-frame path.

The response **parser** (:func:`parse_vlm_response`) is pure and unit-tested; the model
call is not exercised here (no model in CI) — run the live path on the NAS.
"""

from __future__ import annotations

import os
import re
import threading
import time

# A format-instructed presence question: asks for structure *and* reasoning in one pass,
# so a single call yields a best-effort yes/no AND the reasoning (no double-latency two-pass).
# Editable in the GUI — VLM results are very prompt-sensitive, so this doubles as a bench.
DEFAULT_PROMPT = (
    "Is there a cat in this image? Answer in this exact format:\n"
    "ANSWER: yes/no | CONFIDENCE: 0-100% | REASON: <one short sentence>"
)

_MODELS: dict = {}                 # model_path -> loaded moondream model (load is slow; cache)
_LOAD_LOCK = threading.Lock()


def is_available() -> bool:
    """True if the ``moondream`` package is importable (the model may still be absent)."""
    import importlib.util

    return importlib.util.find_spec("moondream") is not None


def parse_vlm_response(text: str) -> dict:
    """Best-effort extraction of ``ANSWER`` / ``CONFIDENCE`` / ``REASON`` from a VLM reply.

    Returns ``{"answer": "yes"|"no"|None, "confidence": int|None, "reason": str|None,
    "parsed": bool}``. ``parsed`` is True only when a structured ``ANSWER:`` was found —
    small VLMs follow format instructions unreliably, so on a miss we report ``parsed:
    False`` and leave the fields ``None`` (the caller shows the raw text). We never
    invent a yes/no or a confidence number (#48)."""
    text = text or ""
    answer = None
    m = re.search(r"ANSWER\s*[:=]\s*(yes|no)\b", text, re.IGNORECASE)
    if m:
        answer = m.group(1).lower()

    confidence = None
    c = re.search(r"CONFIDENCE\s*[:=]\s*(\d{1,3})", text, re.IGNORECASE)
    if c:
        confidence = max(0, min(100, int(c.group(1))))

    reason = None
    r = re.search(r"REASON\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if r:
        # Keep the first line/sentence of the reason; drop trailing format noise.
        reason = r.group(1).strip().splitlines()[0].strip(" |") or None

    return {"answer": answer, "confidence": confidence, "reason": reason,
            "parsed": answer is not None}


def _resolve_model_path(model: str | None) -> str | None:
    """Where the local model file lives: the request value, else ``MOONDREAM_MODEL``."""
    return (model or "").strip() or os.environ.get("MOONDREAM_MODEL") or None


def _load_model(model_path: str | None):
    """Load (and cache) a moondream model. Raises a clear error if the package or the
    model file is missing — both are the user's to provide (optional dep, multi-GB model)."""
    try:
        import moondream as md
    except Exception as exc:        # noqa: BLE001 — optional dependency
        raise RuntimeError(
            "The local VLM tester needs the 'moondream' package — re-run setup "
            "(say yes to the VLM tester) or: pip install moondream"
        ) from exc

    if model_path and not os.path.exists(model_path):
        raise RuntimeError(
            f"moondream model not found at {model_path!r}. Download a quantized model "
            "and set MOONDREAM_MODEL (or the 'model' field) to its path.")

    key = model_path or "__default__"
    with _LOAD_LOCK:
        m = _MODELS.get(key)
        if m is None:
            # The package API drifts between versions; try the documented signatures.
            try:
                m = md.vl(model=model_path) if model_path else md.vl()
            except TypeError:
                m = md.vl(model_path) if model_path else md.vl()
            _MODELS[key] = m
        return m


def _to_pil(frame_bgr):
    """BGR ndarray → RGB PIL image (what moondream's encoder expects)."""
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def query_image(frame_bgr, prompt: str = DEFAULT_PROMPT, model: str | None = None,
                device: str = "cpu") -> dict:
    """Run one moondream **query** pass on ``frame_bgr`` and return a result dict:

    ``raw`` (full response text — always shown), the parsed ``answer``/``confidence``/
    ``reason``/``parsed`` from :func:`parse_vlm_response`, ``load_ms`` (one-time model
    load, 0 if already cached) split from ``query_ms`` (per-frame), and the ``prompt`` /
    ``model`` / ``device`` used (for reproducibility). Raises ``RuntimeError`` with an
    actionable message if moondream or the model is unavailable."""
    model_path = _resolve_model_path(model)
    cached = (model_path or "__default__") in _MODELS

    t0 = time.perf_counter()
    m = _load_model(model_path)
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
    return {
        "raw": raw, "load_ms": load_ms, "query_ms": query_ms,
        "prompt": prompt or DEFAULT_PROMPT,
        "model": os.path.basename(model_path) if model_path else "moondream (default)",
        "device": device or "cpu",
        **parsed,
    }
