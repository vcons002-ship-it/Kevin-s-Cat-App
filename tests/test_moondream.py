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
        "models": ["mobilenet_ssd"], "tilings": ["off"]}).get_json()
    assert body["vlm"] is None              # no VLM cost unless asked


def test_detection_batch_vlm_toggle_flags_disagreements(monkeypatch):
    _mock_vlm(monkeypatch, always="yes")         # VLM says yes on every frame
    c = _client()
    blank = _blank_upload(c)                      # YOLO finds no cat here
    body = c.post("/api/test/benchmark/batch", json={
        "items": [{"id": blank["id"], "name": "empty.jpg", "has_cat": False}],
        "models": ["mobilenet_ssd"], "tilings": ["off"], "run_vlm": True}).get_json()
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
        "models": ["mobilenet_ssd"], "tilings": ["off"], "run_vlm": True}).get_json()
    assert body["configs"]                     # the YOLO sweep still ran
    assert body["vlm"] and "no GPU here" in body["vlm"]["error"]
