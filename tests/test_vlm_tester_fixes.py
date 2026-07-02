"""The VLM tester fixes from the benchmark arc: the batch honours the user's prompt
(#72) with the validated P6 default, the upload cap fits a full benchmark set (#73),
prompts are model-specific (#74), detect mode is exposed for FP diagnosis (#75), and
moondream3's reasoning mode is reachable (#76)."""

import numpy as np

from d20app import moondream as vlm
from d20app import webapp as webapp_mod
from d20app.webapp import create_app


def _session(frames_key="tester-fix", n=1):
    frames = [np.full((120, 160, 3), 60 + 20 * i, np.uint8) for i in range(n)]
    with webapp_mod._TEST_SESSIONS_LOCK:
        webapp_mod._TEST_SESSIONS[frames_key] = frames
    return frames_key


# ---- #72: default prompt is the validated P6; the batch honours the prompt ----
def test_default_prompt_is_p6_with_yes_no_constraint():
    assert "plush toys" in vlm.DEFAULT_PROMPT          # negative exclusions = the lever
    assert "real live cat" in vlm.DEFAULT_PROMPT
    assert vlm.DEFAULT_PROMPT.rstrip().endswith("Yes or No.")   # votable output


def test_batch_threads_the_users_prompt(monkeypatch):
    seen = []
    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)

    def _q(img, **kw):
        seen.append(kw.get("prompt"))
        return {"answer": "no", "reason": "", "raw": "no", "ratio": "1/1",
                "passes": 1, "votes": {"no": 1}, "unanimous": True,
                "borderline": False, "parsed": True, "query_ms": 1.0}
    monkeypatch.setattr(vlm, "query_image_voted", _q)
    c = create_app().test_client()
    sid = _session("batch-prompt")
    body = c.post("/api/vlm/batch", json={
        "items": [{"id": sid, "name": "x.jpg", "has_cat": True}],
        "prompt": "Is there a rhino? Answer Yes or No.", "passes": 1}).get_json()
    assert body["summary"]["n_cat"] == 1
    assert seen == ["Is there a rhino? Answer Yes or No."]    # the bug in #72

    seen.clear()
    c.post("/api/vlm/batch", json={
        "items": [{"id": sid, "name": "x.jpg"}], "passes": 1})
    assert seen == [vlm.DEFAULT_PROMPT]                       # blank → model default


# ---- #73: the upload/test-queue cap fits a full benchmark set -----------------
def test_upload_queue_cap_fits_a_full_set():
    assert webapp_mod._TEST_MAX_SESSIONS >= 1000
    assert webapp_mod._BENCH_MAX_IMAGES_HARD == webapp_mod._TEST_MAX_SESSIONS
    # the VLM VRAM params are NOT the thing that changed (they'd OOM the 8 GB card)
    assert vlm.MAX_BATCH_SIZE == 4 and vlm.KV_CACHE_PAGES == 2048


# ---- #74: prompts are model-specific -------------------------------------------
def test_per_model_default_prompts():
    assert vlm.default_prompt("moondream2") == vlm.DEFAULT_PROMPT
    m3 = vlm.default_prompt("moondream3-preview")
    assert m3 != vlm.DEFAULT_PROMPT and "plush" not in m3     # short form for M3
    assert m3.rstrip().endswith("Yes or No.")
    status = create_app().test_client().get("/api/vlm/status").get_json()
    assert status["model_prompts"]["moondream2"] == vlm.DEFAULT_PROMPT
    assert "moondream3-preview" in status["model_prompts"]


# ---- #75: detect mode exposed for false-positive diagnosis ---------------------
def test_locate_endpoint_maps_and_annotates(monkeypatch):
    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(vlm, "detect_regions",
                        lambda img, obj, **k: {"objects": [
                            {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75}],
                            "detect_ms": 5.0, "model": "moondream2",
                            "mode": "local", "device": "cuda"})
    c = create_app().test_client()
    sid = _session("locate", n=1)
    body = c.post("/api/vlm/locate", json={"id": sid}).get_json()
    assert body["n"] == 1 and body["object"] == "cat"
    x1, y1, x2, y2 = body["boxes"][0]
    assert (x1, y1, x2, y2) == (40, 30, 120, 90)              # normalized → 160×120 px
    assert body["annotated"].startswith("data:image/jpeg")


def test_locate_404_when_session_missing():
    c = create_app().test_client()
    assert c.post("/api/vlm/locate", json={"id": "gone"}).status_code == 404


# ---- #76: moondream3 reasoning mode is reachable --------------------------------
def test_reasoning_flag_reaches_the_model_and_text_surfaces(monkeypatch):
    calls = {}

    class _Model:
        def encode_image(self, img):
            return "enc"

        def query(self, encoded, prompt, reasoning=False):
            calls["reasoning"] = reasoning
            return {"answer": "No",
                    "reasoning": {"text": "I looked carefully; the bed is empty."}}

    monkeypatch.setattr(vlm, "_load_model", lambda *a, **k: _Model())
    monkeypatch.setattr(vlm, "_require_gpu", lambda: "cuda")
    frame = np.zeros((60, 80, 3), np.uint8)
    r = vlm.query_image(frame, reasoning=True)
    assert calls["reasoning"] is True
    assert r["answer"] == "no" and "empty" in r["reasoning"]
    # default stays off — and the model is then called without the kwarg at all
    class _Strict:
        def encode_image(self, img):
            return "enc"

        def query(self, encoded, prompt):                     # no reasoning kwarg
            return {"answer": "Yes"}

    monkeypatch.setattr(vlm, "_load_model", lambda *a, **k: _Strict())
    r2 = vlm.query_image(frame)                               # must not TypeError
    assert r2["answer"] == "yes" and r2["reasoning"] == ""


def test_query_endpoint_passes_reasoning(monkeypatch):
    seen = {}

    def _q(img, **kw):
        seen.update(kw)
        return {"answer": "yes", "reason": "", "raw": "yes", "ratio": "1/1",
                "passes": 1, "votes": {"yes": 1}, "unanimous": True,
                "borderline": False, "parsed": True, "query_ms": 1.0,
                "reasoning": "", "load_ms": 0.0, "prompt": kw.get("prompt", ""),
                "model": "moondream2", "mode": "local", "device": "cuda"}
    monkeypatch.setattr(vlm, "query_image_voted", _q)
    c = create_app().test_client()
    sid = _session("reasoning-q")
    c.post("/api/vlm/query", json={"id": sid, "reasoning": True, "passes": 1})
    assert seen.get("reasoning") is True
