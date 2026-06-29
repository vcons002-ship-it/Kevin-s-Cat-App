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
