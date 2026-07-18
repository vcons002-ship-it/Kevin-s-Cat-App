"""Temporal score fusion (0.37.0): a string of weak, smoothly-moving YOLO hits
confirms a cat no single frame could. Pure math in d20app/fusion.py; detector and
loop wiring here too. The decoy guard is MOVEMENT — a cat-shaped cushion produces
correlated weak hits that never travel."""

import numpy as np

import d20app.config as config_mod
from d20app.detector import PersonDetector
from d20app.fusion import TrackFuser
from d20app.loop import DetectionLoop


def _walk(n, step=40, y=100, size=60, x0=50):
    """n hits walking right: [(score, box)] per frame."""
    return [[(0.35, (x0 + i * step, y, x0 + i * step + size, y + size))]
            for i in range(n)]


def test_moving_weak_track_confirms_exactly_once():
    f = TrackFuser(frame_size=(640, 360))
    confirms = [f.update(1000.0 + i, hits) for i, hits in enumerate(_walk(5))]
    confirms = [c for c in confirms if c]
    assert len(confirms) == 1                    # min-hits confirm, then cooldown
    out = confirms[0]
    assert out["n"] >= 4 and out["travel"] >= 100
    assert 0.3 <= out["score"] <= 0.4            # honest mean of the weak scores
    assert out["box"][0] >= 150                  # the LATEST position, not the first


def test_track_with_a_strong_box_does_not_fuse(monkeypatch):
    # #110: a box that already cleared the normal single-frame bar was confirmed by
    # the normal path — fusion must stand down instead of re-reporting it as
    # "N weak hits" (which is what made a real weak recovery indistinguishable from
    # a redundant misfire).
    f = TrackFuser(frame_size=(640, 360), strong_at=0.5)
    walk = _walk(5)
    walk[2] = [(0.82, walk[2][0][1])]            # one frame is strongly detected
    assert [c for c in (f.update(1000.0 + i, h) for i, h in enumerate(walk)) if c] == []

    # …and with no strong box in the track, the weak recovery still confirms.
    f2 = TrackFuser(frame_size=(640, 360), strong_at=0.5)
    assert [c for c in (f2.update(1000.0 + i, h) for i, h in enumerate(_walk(5))) if c]


def test_strong_guard_off_by_default_keeps_old_behaviour():
    # strong_at=0 (the default) = no guard, so existing callers are unaffected.
    f = TrackFuser(frame_size=(640, 360))
    walk = _walk(5)
    walk[2] = [(0.9, walk[2][0][1])]
    assert [c for c in (f.update(1000.0 + i, h) for i, h in enumerate(walk)) if c]


def test_fusion_debug_records_per_hit_score_label_and_travel():
    # #110 diagnostics: opt-in, structured, and enough to tell a genuine weak
    # recovery from a misfire on the halo of an already-strong detection.
    f = TrackFuser(frame_size=(640, 360), strong_at=0.5, debug=True)
    walk = [[(0.35, b, "cat")] for hits in _walk(5) for _s, b in hits]
    for i, h in enumerate(walk):
        f.update(1000.0 + i, h)
    events = f.take_events()
    assert events and f.take_events() == []          # drained
    ev = next(e for e in events if e["event"] == "confirmed")
    assert ev["n"] >= 4 and ev["track"]
    assert ev["strong_box_present"] is False
    assert ev["travel_px"] >= ev["min_travel_px"]
    assert all(h["label"] == "cat" for h in ev["hits"])
    assert all(0.0 <= h["score"] <= 1.0 for h in ev["hits"])


def test_fusion_debug_records_a_deferral_with_the_strong_flag():
    f = TrackFuser(frame_size=(640, 360), strong_at=0.5, debug=True)
    walk = _walk(5)
    walk[2] = [(0.82, walk[2][0][1])]
    for i, h in enumerate(walk):
        f.update(1000.0 + i, h)
    ev = next(e for e in f.take_events() if e["event"] == "deferred_strong_box")
    assert ev["strong_box_present"] is True and ev["score_max"] >= 0.8


def test_fusion_debug_off_collects_nothing():
    f = TrackFuser(frame_size=(640, 360), strong_at=0.5)
    for i, h in enumerate(_walk(5)):
        f.update(1000.0 + i, h)
    assert f.take_events() == []


def test_stationary_decoy_never_confirms():
    f = TrackFuser(frame_size=(640, 360))
    box = (200, 100, 260, 160)                   # the cushion: weak hits, zero travel
    out = None
    for i in range(10):
        out = f.update(1000.0 + i * 0.5, [(0.4, box)])
        assert out is None


def test_too_few_hits_and_window_pruning():
    f = TrackFuser(frame_size=(640, 360))
    walk = _walk(5)
    for i, hits in enumerate(walk[:3]):          # 3 hits < min 4
        assert f.update(1000.0 + i, hits) is None
    # the 4th and 5th hits arrive AFTER the window: the early ones fall off
    assert f.update(1010.0, walk[3]) is None
    assert f.update(1011.0, walk[4]) is None


def test_reconfirm_cooldown():
    f = TrackFuser(frame_size=(640, 360))
    confirms = [f.update(1000.0 + i, h) for i, h in enumerate(_walk(5))]
    assert any(confirms)
    # keep walking — the track must NOT re-confirm within the cooldown
    for i, hits in enumerate(_walk(4, x0=250)):
        assert f.update(1005.0 + i, hits) is None


def test_teleporting_hits_do_not_chain():
    f = TrackFuser(frame_size=(640, 360))
    # alternate far corners: no IoU, jumps far beyond a box diagonal → separate
    # tracks, each stationary → never confirms
    a, b = (10, 10, 70, 70), (500, 280, 560, 340)
    for i in range(10):
        assert f.update(1000.0 + i, [(0.4, a if i % 2 else b)]) is None


# ---- detector wiring ----------------------------------------------------------
def _detector_with_motion(track_fusion=True):
    det = PersonDetector(source="unused", track_fusion=track_fusion,
                         cat_scan_frames=1)
    frame = np.full((360, 640, 3), 110, np.uint8)

    class _Cap:
        def read(self):
            return True, frame

    det._ensure_cap = lambda: _Cap()
    det._motion.update = lambda gray: True       # every frame "moves"
    return det


def test_detector_fuses_weak_hits_and_filters_them_from_live_boxes():
    det = _detector_with_motion()
    seen_floors = []
    step = {"i": 0}

    def fake_detect(frame, floor):
        seen_floors.append(floor)
        i = step["i"]
        step["i"] += 1
        return [("cat", 0.25, (50 + i * 40, 100, 110 + i * 40, 160))]

    det._detect_boxes = fake_detect
    for _ in range(5):
        det.read_and_detect(detect=True, force=False)
    assert all(f <= det._FUSE_FLOOR for f in seen_floors)     # weak-floor decode
    assert det._last_boxes == []                 # weak boxes never reach the feed
    fused = det.take_fused_hit()
    assert fused and fused["n"] >= 4
    assert det.take_fused_hit() is None          # claimed exactly once
    assert det._cat_last_seen > 0                # presence updated


def _boxes_of(det, frames, hits):
    """Run one detect pass with `hits` as the net's output; return the kept boxes."""
    class _Cap:
        def read(self):
            return True, frames[-1]

    det._ensure_cap = lambda: _Cap()
    det._run_net = lambda img, floor, size=None: list(hits)
    det.read_and_detect(detect=True)                  # motion baseline
    f = frames[-1].copy()
    f[200:280, 300:380] = 220
    frames.append(f)
    det.read_and_detect(detect=True)
    return det._last_boxes


def test_dog_and_cat_merge_into_one_strong_cat_when_toggle_on():
    # #110: with "count dog as the cat" on, a strong dog box and the weak cat halo
    # on the SAME animal are one animal — they must merge into a single strong cat
    # that confirms through the normal path (not dribble into fusion as weak hits).
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = PersonDetector(source="unused", cat_confidence=0.5,
                         locator_classes=["cat", "dog"], track_fusion=False)
    box = (300, 100, 380, 180)
    boxes = _boxes_of(det, frames, [("dog", 0.85, box), ("cat", 0.30, box)])

    cats = [b for b in boxes if b[0] == "cat"]
    assert len(cats) == 1 and not [b for b in boxes if b[0] == "dog"]
    assert cats[0][1] == 0.85                         # the STRONG reading survives
    assert det.last_confirmed()["label"] == "cat"     # confirmed via the normal path


def test_dog_and_cat_stay_separate_when_toggle_off():
    # A real-dog household: dog and cat are different animals — never merge.
    frames = [np.full((360, 640, 3), 110, np.uint8)]
    det = PersonDetector(source="unused", cat_confidence=0.5,
                         locator_classes=["cat"], track_fusion=False)
    box = (300, 100, 380, 180)
    boxes = _boxes_of(det, frames, [("dog", 0.85, box), ("cat", 0.55, box)])

    labels = sorted(b[0] for b in boxes)
    assert labels == ["cat", "dog"]                   # both kept, unmerged
    assert det.last_confirmed()["label"] == "cat"     # and the dog is not "the cat"


def test_detector_fusion_off_keeps_single_frame_behaviour():
    det = _detector_with_motion(track_fusion=False)
    floors = []
    det._detect_boxes = lambda frame, floor: floors.append(floor) or []
    det.read_and_detect(detect=True, force=False)
    normal = min(det.label_floor, det.confidence, det.cat_confidence)
    assert floors == [normal] and normal > det._FUSE_FLOOR   # no weak decode
    assert det.take_fused_hit() is None


# ---- loop record site ----------------------------------------------------------
def test_loop_records_fused_sighting_as_track_source(tmp_path, monkeypatch):
    loop = DetectionLoop()
    loop.cats._path = str(tmp_path / "cats.log")             # isolate the log
    loop.cats._items = []

    class _Det:
        frame_size = (640, 360)
        locator_classes = ("cat",)

        def take_fused_hit(self):
            return {"box": [220, 100, 280, 160], "score": 0.36, "n": 5,
                    "span_s": 4.0, "travel": 160.0, "path": []}

        def annotated_jpeg(self):
            import cv2
            ok, buf = cv2.imencode(".jpg", np.zeros((60, 80, 3), np.uint8))
            return buf.tobytes()

    s = loop._record_fused("Room", "Room", {"zones": [], "roi": None}, _Det())
    assert s and s["source"] == "track" and s["score"] == 0.36
    recent = loop.cats.recent(limit=5)
    assert recent and recent[0]["source"] == "track"


def test_camera_spec_inherits_track_fusion(monkeypatch, tmp_path):
    from d20app.webapp import create_app

    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    c = create_app().test_client()
    assert c.get("/api/config").get_json()["track_fusion"] is True   # default on
    c.post("/api/config", json={"track_fusion": False})
    c.post("/api/cameras/saved", json={"name": "K", "url": "rtsp://1/s"})
    assert c.get("/api/cameras/saved").get_json()[0]["track_fusion"] is False
