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


# ---- parser (pure) ---------------------------------------------------------
def test_parser_extracts_a_clean_formatted_response():
    out = vlm.parse_vlm_response(
        "ANSWER: yes | CONFIDENCE: 92% | REASON: a tabby cat sits on the sofa")
    assert out["answer"] == "yes" and out["confidence"] == 92
    assert out["reason"] == "a tabby cat sits on the sofa" and out["parsed"] is True


def test_parser_handles_no_answer():
    out = vlm.parse_vlm_response("ANSWER: no | CONFIDENCE: 40% | REASON: empty room")
    assert out["answer"] == "no" and out["confidence"] == 40 and out["parsed"] is True


def test_parser_is_case_and_separator_tolerant():
    out = vlm.parse_vlm_response("answer = Yes\nconfidence = 5\nreason = faint shape")
    assert out["answer"] == "yes" and out["confidence"] == 5


def test_parser_clamps_confidence():
    assert vlm.parse_vlm_response("ANSWER: yes CONFIDENCE: 250")["confidence"] == 100


def test_parser_unparsed_when_model_rambles():
    # A small VLM that ignored the format → mark unparsed, invent nothing, keep raw.
    out = vlm.parse_vlm_response("I think I can see something furry but I'm not certain.")
    assert out["answer"] is None and out["confidence"] is None and out["parsed"] is False


def test_parser_yes_without_confidence_still_parses():
    out = vlm.parse_vlm_response("ANSWER: yes - pretty sure")
    assert out["answer"] == "yes" and out["confidence"] is None and out["parsed"] is True


# ---- endpoints -------------------------------------------------------------
def _client():
    return create_app().test_client()


def _upload(c):
    with open(_CAT, "rb") as fh:
        data = fh.read()
    return c.post("/api/test/upload",
                  data={"file": (io.BytesIO(data), "cat01.jpg")},
                  content_type="multipart/form-data").get_json()


def test_vlm_status_reports_availability_and_default_prompt():
    body = _client().get("/api/vlm/status").get_json()
    assert "available" in body and isinstance(body["available"], bool)
    assert "ANSWER:" in body["default_prompt"]


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
