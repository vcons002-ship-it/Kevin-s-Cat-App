"""Local VLM (moondream) cat-presence tester (#48).

The model isn't present in CI (multi-GB, optional), so we exercise the **pure response
parser** thoroughly and assert the endpoint **degrades gracefully** when moondream isn't
installed — never a 500, always an actionable message.
"""

import io
import os

from d20app import moondream as vlm
from d20app.webapp import create_app

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_CAT = os.path.join(_FIXTURES, "cats", "cat01.jpg")


# ---- parser (pure) — yes/no + free-text reason, no confidence (#52/#54) ----
def test_parser_extracts_yes_and_keeps_reason():
    out = vlm.parse_vlm_response("Yes, a tabby cat is sitting on the sofa.")
    assert out["answer"] == "yes" and out["parsed"] is True
    assert "tabby cat" in out["reason"] and "confidence" not in out


def test_parser_extracts_no():
    out = vlm.parse_vlm_response("No — the room is empty.")
    assert out["answer"] == "no" and out["parsed"] is True


def test_parser_is_case_insensitive():
    assert vlm.parse_vlm_response("YES. clearly a cat.")["answer"] == "yes"


def test_parser_unparsed_when_no_yes_or_no():
    # Model never committed to yes/no → unparsed, invent nothing, keep the raw text.
    out = vlm.parse_vlm_response("I think I can see something furry but I'm uncertain.")
    assert out["answer"] is None and out["parsed"] is False
    assert out["reason"].startswith("I think")


def test_parser_takes_the_first_verdict():
    # "yes" before "no" → yes (the model's lead answer).
    assert vlm.parse_vlm_response("Yes. It is not a dog, it's a cat.")["answer"] == "yes"


# ---- endpoints -------------------------------------------------------------
def _client():
    return create_app().test_client()


def _upload(c):
    with open(_CAT, "rb") as fh:
        data = fh.read()
    return c.post("/api/test/upload",
                  data={"file": (io.BytesIO(data), "cat01.jpg")},
                  content_type="multipart/form-data").get_json()


def test_vlm_status_reports_availability_models_and_prompt():
    body = _client().get("/api/vlm/status").get_json()
    assert "available" in body and isinstance(body["available"], bool)
    assert "cat" in body["default_prompt"].lower()
    assert "moondream2" in body["models"] and body["default_model"] == "moondream2"


# ---- #59: mode selector, validated load params, API key, friendly errors ----
def test_vlm_status_exposes_local_and_cloud_choices_and_key_flag():
    body = _client().get("/api/vlm/status").get_json()
    assert "has_api_key" in body and isinstance(body["has_api_key"], bool)
    choices = {c["value"]: c for c in body["choices"]}
    assert body["default_choice"] == "local:moondream2"
    # local M2 stays on-device; cloud M3 is flagged as leaving the box (informed choice).
    assert choices["local:moondream2"]["off_device"] is False
    assert choices["cloud:moondream3-preview"]["off_device"] is True


def test_api_key_is_masked_in_config_and_blank_save_keeps_it(monkeypatch, tmp_path):
    import d20app.config as config_mod
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
    c = create_app().test_client()
    # Save a key, then confirm the GET never echoes it back — only a boolean flag.
    c.post("/api/config", json={"moondream_api_key": "secret-key-123"})
    cfg = c.get("/api/config").get_json()
    assert "moondream_api_key" not in cfg and cfg["has_moondream_api_key"] is True
    assert "secret-key-123" not in str(cfg)
    # A blank field on a later save must NOT wipe the stored key.
    c.post("/api/config", json={"moondream_api_key": "", "dc": 18})
    assert config_mod.load().moondream_api_key == "secret-key-123"


def test_preflight_local_needs_gpu_but_cloud_does_not(monkeypatch):
    monkeypatch.setattr(vlm, "is_available", lambda: True)
    monkeypatch.setattr(vlm, "_require_gpu",
                        lambda: (_ for _ in ()).throw(RuntimeError("no GPU")))
    # local mode hits the GPU check and raises…
    try:
        vlm.preflight("key", mode="local")
        assert False, "local preflight should require a GPU"
    except RuntimeError as e:
        assert "GPU" in str(e)
    # …cloud mode runs off-device, so a key + package is enough (no GPU).
    vlm.preflight("key", mode="cloud")


def test_load_model_uses_validated_local_params_and_caches(monkeypatch):
    import sys, types
    calls = []

    class _Fake:
        def vl(self, **kw):
            calls.append(kw)
            return object()
    fake = types.ModuleType("moondream")
    fake.vl = _Fake().vl
    monkeypatch.setitem(sys.modules, "moondream", fake)
    monkeypatch.setattr(vlm, "_require_gpu", lambda: "cuda")
    monkeypatch.setattr(vlm, "_MODELS", {})

    m1 = vlm._load_model("moondream2", "key", mode="local")
    assert calls[-1] == {"api_key": "key", "local": True, "model": "moondream2",
                         "max_batch_size": vlm.MAX_BATCH_SIZE,
                         "kv_cache_pages": vlm.KV_CACHE_PAGES}
    m2 = vlm._load_model("moondream2", "key", mode="local")     # cached → no second vl()
    assert m1 is m2 and len(calls) == 1


def test_load_model_cloud_call_has_no_local_flag(monkeypatch):
    import sys, types
    calls = []
    fake = types.ModuleType("moondream")
    fake.vl = lambda **kw: calls.append(kw) or object()
    monkeypatch.setitem(sys.modules, "moondream", fake)
    monkeypatch.setattr(vlm, "_MODELS", {})
    # Cloud must NOT pass local=True/model — that's what hits the hosted API.
    vlm._load_model("moondream3-preview", "key", mode="cloud")
    assert calls[-1] == {"api_key": "key"}


def test_friendly_error_maps_oom_and_kv_cache(monkeypatch):
    oom = vlm._friendly_model_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"), "moondream2")
    assert "GPU memory" in str(oom) and "max_batch_size" in str(oom)
    kv = vlm._friendly_model_error(
        RuntimeError("Scheduler stalled: insufficient KV cache capacity"), "moondream2")
    assert "KV cache" in str(kv) and "kv_cache_pages" in str(kv)


def test_query_passes_mode_through(monkeypatch):
    import numpy as np
    seen = {}

    class _M:
        def encode_image(self, img): return "enc"
        def query(self, enc, prompt): return {"answer": "yes, a cat"}
    def _load(name, key, mode=vlm.DEFAULT_MODE):
        seen["mode"] = mode
        return _M()
    monkeypatch.setattr(vlm, "_load_model", _load)
    out = vlm.query_image(np.zeros((8, 8, 3), np.uint8), mode="cloud")
    assert seen["mode"] == "cloud" and out["mode"] == "cloud" and out["device"] == "cloud"
    assert out["answer"] == "yes"


def test_vlm_query_degrades_gracefully_without_model(monkeypatch):
    # With moondream absent (or no model), the query returns a clear 503 message,
    # never a 500 — the GUI tells the user how to enable it.
    monkeypatch.setattr(vlm, "is_available", lambda: False)

    def _boom(*a, **k):
        raise RuntimeError("The local VLM tester needs the 'moondream' package — "
                           "pip install moondream")
    monkeypatch.setattr(vlm, "query_image", _boom)

    c = _client()
    up = _upload(c)
    r = c.post("/api/vlm/query", json={"id": up["id"], "frame_index": 0})
    assert r.status_code == 503 and "moondream" in r.get_json()["error"]


def test_vlm_query_404_when_frame_missing():
    r = _client().post("/api/vlm/query", json={"id": "gone"})
    assert r.status_code == 404


# ---- batch VLM + detection-batch toggle (#54) ------------------------------
import numpy as np                                            # noqa: E402


def _blank_upload(c, name="empty.jpg"):
    import cv2
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))
    return c.post("/api/test/upload",
                  data={"file": (io.BytesIO(buf.tobytes()), name)},
                  content_type="multipart/form-data").get_json()


def _mock_vlm(monkeypatch, always=None):
    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)

    def _q(frame, **kw):
        ans = always or ("yes" if float(np.asarray(frame).std()) > 5 else "no")
        return {"answer": ans, "reason": f"mock {ans}", "query_ms": 12.0}
    monkeypatch.setattr(vlm, "query_image", _q)


def test_vlm_summary_separates_recall_and_fp():
    v = [{"has_cat": True, "answer": "yes"}, {"has_cat": True, "answer": "no"},
         {"has_cat": False, "answer": "yes"}, {"has_cat": False, "answer": "no"},
         {"has_cat": True, "answer": None, "error": "x"}]   # errored → excluded
    from d20app.webapp import _vlm_summary
    s = _vlm_summary(v)
    assert s["n_cat"] == 2 and s["found"] == 1 and s["recall"] == 0.5
    assert s["n_nocat"] == 2 and s["fp"] == 1 and s["fp_rate"] == 0.5
    assert s["errors"] == 1


def test_vlm_batch_scores_recall_and_fp(monkeypatch):
    _mock_vlm(monkeypatch)                       # cat photo → yes, blank → no
    c = _client()
    cat = _upload(c)
    blank = _blank_upload(c)
    body = c.post("/api/vlm/batch", json={"items": [
        {"id": cat["id"], "name": "cat.jpg", "has_cat": True},
        {"id": blank["id"], "name": "empty.jpg", "has_cat": False}]}).get_json()
    assert body["ran"] == 2
    assert body["summary"]["recall"] == 1.0 and body["summary"]["fp_rate"] == 0.0
    assert len(body["verdicts"]) == 2


def test_vlm_batch_503_on_setup_error(monkeypatch):
    monkeypatch.setattr(vlm, "preflight",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("needs a GPU")))
    c = _client()
    up = _upload(c)
    r = c.post("/api/vlm/batch", json={"items": [{"id": up["id"], "has_cat": True}]})
    assert r.status_code == 503 and "GPU" in r.get_json()["error"]


def test_detection_batch_vlm_toggle_off_by_default():
    c = _client()
    up = _upload(c)
    body = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": up["id"], "name": "c.jpg", "has_cat": True}],
        "models": ["yolo11n"], "tilings": ["off"]}).get_json()
    assert body["vlm"] is None              # no VLM cost unless asked


def test_detection_batch_vlm_toggle_flags_disagreements(monkeypatch):
    _mock_vlm(monkeypatch, always="yes")         # VLM says yes on every frame
    c = _client()
    blank = _blank_upload(c)                      # YOLO finds no cat here
    body = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": blank["id"], "name": "empty.jpg", "has_cat": False}],
        "models": ["yolo11n"], "tilings": ["off"], "run_vlm": True}).get_json()
    assert body["vlm"] and body["vlm"]["summary"]["fp"] == 1   # VLM false-fired on no-cat
    dis = body["vlm"]["disagreements"]
    assert any(d["vlm"] == "yes" and d["yolo"] == "miss" for d in dis)
    html = c.get(body["summary_url"]).get_data(as_text=True)
    assert "VLM" in html and "accuracy" in html and "disagreement" in html.lower()


def test_detection_batch_vlm_unavailable_still_runs_sweep(monkeypatch):
    monkeypatch.setattr(vlm, "preflight",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no GPU here")))
    c = _client()
    up = _upload(c)
    body = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": up["id"], "name": "c.jpg", "has_cat": True}],
        "models": ["yolo11n"], "tilings": ["off"], "run_vlm": True}).get_json()
    assert body["configs"]                     # the YOLO sweep still ran
    assert body["vlm"] and "no GPU here" in body["vlm"]["error"]
