"""Escalation ladder (#66): crop math, the CPU predictor, ladder decisions, and the
on-demand endpoint. The ladder's detectors are injected callables, so every decision
is verified here without a net or a GPU; the moondream calls are mocked exactly like
tests/test_moondream.py. What only the NAS can verify: real detect() output quality,
coordinate orientation on real frames, and VRAM co-residency — see the PR checklist.
"""

import io
import os

import numpy as np
import pytest

from d20app import escalation as esc
from d20app import moondream as vlm
from d20app.webapp import create_app

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_CAT = os.path.join(_FIXTURES, "cats", "cat01.jpg")


# ---- crop math (pure) -------------------------------------------------------
def test_square_crop_pads_and_squares():
    box = esc.square_crop_box((500, 300, 600, 350), (1280, 720))
    x1, y1, x2, y2 = box
    assert x2 - x1 == y2 - y1 == esc.MIN_CROP        # small hint → floor size
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    assert abs(cx - 550) <= 1 and abs(cy - 325) <= 1  # centred on the hint


def test_square_crop_shifts_at_edges_instead_of_shrinking():
    for hint in [(0, 0, 30, 30), (1250, 0, 1280, 30), (0, 690, 30, 720),
                 (1250, 690, 1280, 720)]:
        x1, y1, x2, y2 = esc.square_crop_box(hint, (1280, 720))
        assert x2 - x1 == y2 - y1 == esc.MIN_CROP     # full-size window kept
        assert 0 <= x1 and x2 <= 1280 and 0 <= y1 and y2 <= 720


def test_square_crop_respects_max_and_small_frames():
    big = esc.square_crop_box((0, 0, 3000, 3000), (4000, 4000))
    assert big[2] - big[0] == esc.MAX_CROP
    tiny = esc.square_crop_box((10, 10, 50, 50), (200, 100))
    assert tiny[3] - tiny[1] <= 100                   # capped by the frame


def test_map_box_round_trip():
    crop = esc.square_crop_box((400, 300, 500, 380), (1280, 720))
    in_crop = (10, 20, 60, 80)
    back = esc.map_box_to_frame(in_crop, crop)
    assert back == (crop[0] + 10, crop[1] + 20, crop[0] + 60, crop[1] + 80)


def test_map_normalized_box_clamps_and_rejects_degenerate():
    assert esc.map_normalized_box(
        {"x_min": 0.25, "y_min": 0.5, "x_max": 0.5, "y_max": 1.0}, (400, 200)) == \
        (100, 100, 200, 200)
    # out-of-range values clamp to the frame
    assert esc.map_normalized_box(
        {"x_min": -0.5, "y_min": 0.0, "x_max": 1.5, "y_max": 1.0}, (400, 200)) == \
        (0, 0, 400, 200)
    # slivers and junk are rejected, not passed downstream
    assert esc.map_normalized_box(
        {"x_min": 0.5, "y_min": 0.1, "x_max": 0.5001, "y_max": 0.9}, (400, 200)) is None
    assert esc.map_normalized_box({"x_min": "nan?"}, (400, 200)) is None


def test_merge_hint_boxes_dedupes_and_caps():
    merged = esc.merge_hint_boxes(
        [(0, 0, 100, 100), (10, 10, 50, 50),          # contained → dropped
         (5, 5, 95, 95),                               # high IoU → dropped
         (300, 300, 340, 340), (600, 0, 640, 40), (0, 600, 40, 640),
         (900, 900, 940, 940)], max_boxes=4)
    assert merged[0] == (0, 0, 100, 100)               # largest first
    assert (10, 10, 50, 50) not in merged
    assert len(merged) == 4                            # capped


# ---- predict_hint_box: the CPU "cat targeting" ------------------------------
def test_predictor_leads_the_motion():
    # centroid moves +100px/s in x; extrapolating 1s past the last fix → +100 more
    tracks = [(0.0, (100, 100, 140, 140)), (1.0, (200, 100, 240, 140))]
    box = esc.predict_hint_box(tracks, now=2.0, frame_size=(1280, 720))
    cx = (box[0] + box[2]) / 2
    assert 315 <= cx <= 325                            # led to ~x=320
    assert box[3] - box[1] > 40                        # uncertainty pad grew the box


def test_predictor_stands_down_without_enough_data():
    # <2 fixes, stale track, or simultaneous fixes → None (the LLM is the failsafe)
    assert esc.predict_hint_box([(0.0, (0, 0, 10, 10))], now=1.0) is None
    assert esc.predict_hint_box(
        [(0.0, (0, 0, 10, 10)), (1.0, (5, 0, 15, 10))], now=60.0) is None
    assert esc.predict_hint_box(
        [(1.0, (0, 0, 10, 10)), (1.0, (5, 0, 15, 10))], now=2.0) is None
    assert esc.predict_hint_box([], now=1.0) is None


def test_predictor_clamps_at_frame_edges():
    # fast mover headed off-frame: the predicted box clamps inside, never negative
    tracks = [(0.0, (50, 50, 90, 90)), (0.5, (10, 50, 50, 90))]
    box = esc.predict_hint_box(tracks, now=1.5, frame_size=(640, 480))
    assert box is None or (box[0] >= 0 and box[1] >= 0)


# ---- the ladder (stub callables + call counters) ----------------------------
_FRAME = np.zeros((720, 1280, 3), np.uint8)


def _counting(fn):
    def wrapper(*a, **k):
        wrapper.calls += 1
        return fn(*a, **k)
    wrapper.calls = 0
    return wrapper


def test_rung1_hit_never_touches_the_vlm():
    detect = _counting(lambda img: [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}])
    query = _counting(lambda img: {"answer": "yes", "passes": 1, "votes": {"yes": 1}})
    r = esc.escalate(_FRAME, hints=[(600, 300, 700, 400)],
                     run_yolo=lambda img: [("cat", 0.9, (10, 10, 60, 60))],
                     vlm_detect=detect, vlm_query=query)
    assert r["found"] and r["source"] == "zoom+yolo"
    assert detect.calls == 0 and query.calls == 0      # the cheap-first guarantee
    # the found box is mapped back into frame coords, inside the crop window
    assert r["box"][0] >= 440 and r["box"][1] >= 140


def test_rung2_detect_confirmed_by_yolo():
    r = esc.escalate(_FRAME, hints=[],
                     run_yolo=lambda img: [("cat", 0.8, (5, 5, 50, 50))]
                     if img.shape[0] < 720 else [],
                     vlm_detect=lambda img: [{"x_min": 0.4, "y_min": 0.4,
                                              "x_max": 0.5, "y_max": 0.55}],
                     vlm_query=None)
    assert r["found"] and r["source"] == "vlm+yolo" and r["score"] == 0.8


def test_rung2_votes_only_yes_is_a_lead_not_a_find():
    # YOLO silent, voted query says yes → DEMOTED (#69: decoy FP; voting fixes
    # variance, not bias): found stays False, the yes comes back as vlm_probable.
    r = esc.escalate(_FRAME, hints=[],
                     run_yolo=lambda img: [],           # YOLO stays silent
                     vlm_detect=lambda img: [{"x_min": 0.4, "y_min": 0.4,
                                              "x_max": 0.6, "y_max": 0.6}],
                     vlm_query=lambda img: {"answer": "yes", "ratio": "2/3",
                                            "passes": 3, "votes": {"yes": 2, "no": 1}})
    assert r["found"] is False and r["source"] is None
    vp = r["vlm_probable"]
    assert vp and vp["ratio"] == "2/3" and vp["rung"] == "vlm detect"
    assert 0.6 < vp["score"] < 0.7                      # 2/3 as the vote score
    assert "unconfirmed" in r["rungs"][1]["note"]


def test_rung2_later_yolo_confirm_beats_earlier_vlm_yes():
    # Region 1: YOLO silent, VLM votes yes (a lead). Region 2: YOLO confirms.
    # The confirmed find must win — the ladder keeps scanning past a votes-only yes.
    regions = [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
               {"x_min": 0.6, "y_min": 0.6, "x_max": 0.7, "y_max": 0.7}]
    def yolo(img):
        yolo.n += 1
        return [("cat", 0.85, (5, 5, 40, 40))] if yolo.n >= 2 else []
    yolo.n = 0
    r = esc.escalate(_FRAME, hints=[], run_yolo=yolo,
                     vlm_detect=lambda img: regions,
                     vlm_query=lambda img: {"answer": "yes", "ratio": "3/3",
                                            "passes": 3, "votes": {"yes": 3}})
    assert r["found"] and r["source"] == "vlm+yolo" and r["score"] == 0.85


def test_bare_detect_region_never_records_alone():
    # detect proposes, YOLO silent, query says NO → not found (decoy resistance)
    r = esc.escalate(_FRAME, hints=[],
                     run_yolo=lambda img: [],
                     vlm_detect=lambda img: [{"x_min": 0.4, "y_min": 0.4,
                                              "x_max": 0.6, "y_max": 0.6}],
                     vlm_query=lambda img: {"answer": "no", "passes": 3,
                                            "votes": {"no": 3}})
    assert r["found"] is False


def test_rung3_query_yes_is_probable_only():
    r = esc.escalate(_FRAME, hints=[(100, 100, 200, 200)],
                     run_yolo=lambda img: [],
                     vlm_detect=lambda img: [],          # no regions proposed
                     vlm_query=lambda img: {"answer": "yes", "ratio": "3/3",
                                            "passes": 3, "votes": {"yes": 3}})
    assert r["found"] is False                          # #69 demotion
    assert r["vlm_probable"] and r["vlm_probable"]["rung"] == "vlm query"
    assert [x["name"] for x in r["rungs"]] == ["zoom+yolo", "vlm detect", "vlm query"]


def test_vlm_unavailable_rungs_marked_not_run():
    r = esc.escalate(_FRAME, hints=[(0, 0, 50, 50)], run_yolo=lambda img: [],
                     vlm_detect=None, vlm_query=None)
    assert r["found"] is False
    by_name = {x["name"]: x for x in r["rungs"]}
    assert by_name["vlm detect"]["ran"] is False
    assert "unavailable" in by_name["vlm detect"]["note"]


def test_crop_bounds_respected():
    hints = [(i * 150, 0, i * 150 + 60, 60) for i in range(10)]   # 10 spread hints
    seen = _counting(lambda img: [])
    r = esc.escalate(_FRAME, hints=hints, run_yolo=seen,
                     vlm_detect=None, vlm_query=None)
    assert seen.calls <= esc.MAX_CROPS                  # ≤4 crops on rung 1
    assert r["rungs"][0]["crops"] <= esc.MAX_CROPS


def test_detect_exception_does_not_kill_the_ladder():
    def boom(img):
        raise RuntimeError("GPU fell over")
    r = esc.escalate(_FRAME, hints=[(0, 0, 50, 50)], run_yolo=lambda img: [],
                     vlm_detect=boom,
                     vlm_query=lambda img: {"answer": "yes", "passes": 1,
                                            "votes": {"yes": 1}})
    assert r["vlm_probable"] is not None                # rung 3 still ran
    assert r["found"] is False                          # …but a vote can't confirm


# ---- endpoint ---------------------------------------------------------------
def _client():
    return create_app().test_client()


def _upload(c):
    with open(_CAT, "rb") as fh:
        return c.post("/api/test/upload",
                      data={"file": (io.BytesIO(fh.read()), "cat01.jpg")},
                      content_type="multipart/form-data").get_json()


def test_escalate_404_when_session_missing():
    assert _client().post("/api/vlm/escalate", json={"id": "gone"}).status_code == 404


def test_escalate_403_for_live_camera_when_toggle_off():
    r = _client().post("/api/vlm/escalate", json={"camera": "Kitchen"})
    assert r.status_code == 403 and "escalation" in r.get_json()["error"].lower()


def test_escalate_409_when_loop_not_running(monkeypatch, tmp_path):
    import d20app.config as config_mod
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    config_mod.update({"vlm_escalation": True})
    r = _client().post("/api/vlm/escalate", json={"camera": "Kitchen"})
    assert r.status_code == 409


def test_escalate_503_actionable_on_vlm_preflight_failure(monkeypatch):
    monkeypatch.setattr(vlm, "preflight",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("needs a GPU")))
    c = _client()
    up = _upload(c)
    r = c.post("/api/vlm/escalate", json={"id": up["id"], "use_vlm": True})
    assert r.status_code == 503 and "GPU" in r.get_json()["error"]


def test_escalate_zoom_rung_finds_the_fixture_cat_without_vlm():
    # A real end-to-end rung-1 run: hint over the cat → yolo11n on the crop finds it.
    c = _client()
    up = _upload(c)
    body = c.post("/api/vlm/escalate", json={
        "id": up["id"], "use_vlm": False,
        "hints": [[50, 50, 300, 300]]}).get_json()
    assert body["found"] and body["source"] == "zoom+yolo" and body["label"] == "cat"
    assert body["annotated"].startswith("data:image/jpeg;base64,")
    assert body["note"] and "disabled" in body["note"]
    assert all(c["image"].startswith("data:image") for c in body["crops"])


def test_escalate_vlm_detect_path_with_mocks(monkeypatch):
    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(vlm, "detect_regions",
                        lambda img, obj, **k: {"objects": [
                            {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}]})
    c = _client()
    up = _upload(c)
    body = c.post("/api/vlm/escalate", json={"id": up["id"]}).get_json()
    # detect proposed the cat region; the crop is confirmed by real yolo11n
    assert body["found"] and body["source"] == "vlm+yolo"


def test_escalate_live_vlm_lead_probable_boosts_and_never_records(monkeypatch, tmp_path):
    # A votes-only VLM "yes" on a live camera (#69 demotion): the response is the
    # "probable" tier (kind vlm), NOTHING lands in the sightings log, and detection
    # is boosted so real YOLO gets the chance to confirm on the next frames.
    import d20app.config as config_mod
    from d20app.detector import PersonDetector
    from d20app.loop import DetectionLoop

    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    config_mod.update({"vlm_escalation": True})
    loop = DetectionLoop()

    class _Alive:
        def is_alive(self):
            return True

    loop._thread = _Alive()
    det = PersonDetector(source="unused")
    loop._detectors = {"Room": det}
    det._publish_frame(np.full((360, 640, 3), 110, np.uint8))   # blank: YOLO silent

    monkeypatch.setattr(vlm, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(vlm, "detect_regions",
                        lambda img, obj, **k: {"objects": [
                            {"x_min": 0.3, "y_min": 0.3, "x_max": 0.6, "y_max": 0.6}]})
    monkeypatch.setattr(vlm, "query_image_voted",
                        lambda img, **k: {"answer": "yes", "ratio": "3/3",
                                          "passes": 3, "votes": {"yes": 3}})
    before = len(loop.cats.recent(limit=500))
    c = create_app(loop).test_client()
    body = c.post("/api/vlm/escalate", json={"camera": "Room"}).get_json()
    assert body["found"] is False
    assert body["probable"] and body["probable"]["kind"] == "vlm"
    assert "3/3" in body["probable"]["note"]
    assert len(loop.cats.recent(limit=500)) == before   # no sighting recorded
    assert loop._cat_boost.get("Room")                  # YOLO gets its chance
    # Targeted (0.42.0): the boost carries the lead's box, so forced scans zoom
    # that exact spot with the heaviest available model.
    assert det.boost_hint() == tuple(body["probable"]["box"])


def test_vlm_status_exposes_escalation_flag():
    body = _client().get("/api/vlm/status").get_json()
    assert body["escalation"] == {"enabled": False}


def test_config_round_trips_vlm_escalation(monkeypatch, tmp_path):
    import d20app.config as config_mod
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    c.post("/api/config", json={"vlm_escalation": True})
    assert c.get("/api/config").get_json()["vlm_escalation"] is True
    assert c.get("/api/vlm/status").get_json()["escalation"]["enabled"] is True
