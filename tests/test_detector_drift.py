"""Do the LIVE detectors hold the settings you configured?

Find builds its settings from the running detector's in-memory attributes, not
from config.yaml — so a hot-reload that didn't apply, or a `reconfigure` that
misses a field, makes find scan with settings you never set while still reporting
the ones it was handed. That is indistinguishable from outside, and it is a
standing hypothesis for finds that miss cats the Test tool finds in the same
frame. This checks it without needing a cat to walk past.
"""

import d20app.config as config_mod
from d20app.detector import PersonDetector
from d20app.webapp import create_app


def _client(monkeypatch, tmp_path, cam, **overrides):
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    app = create_app()
    c = app.test_client()
    c.post("/api/cameras/saved", json={"name": cam, "url": "rtsp://a/s", **overrides})
    c.post("/api/cameras/active", json={"names": [cam]})
    return app, c


def _detector_for(app, cam):
    spec = next(s for s in config_mod.camera_targets(config_mod.load())
                if s["name"] == cam)
    det = PersonDetector(source=spec["source"], model=spec["model"],
                         accelerator=spec["accelerator"],
                         confidence=spec["person_confidence"],
                         cat_confidence=spec["cat_confidence"],
                         label_floor=spec["label_floor"],
                         locator_classes=spec["locator_classes"],
                         roi=spec["roi"], gamma=spec["gamma"],
                         brightness=spec["brightness"], contrast=spec["contrast"],
                         saturation=spec["saturation"])
    app.config["loop"]._detectors = {cam: det}
    return det


def test_a_detector_built_from_the_spec_reports_no_drift(monkeypatch, tmp_path):
    app, c = _client(monkeypatch, tmp_path, "Office",
                     cat_confidence=0.3, person_confidence=0.5,
                     locator_classes=["cat", "dog"])
    _detector_for(app, "Office")
    body = c.get("/api/diagnostics/detectors").get_json()
    row = body["detectors"][0]
    assert row["camera"] == "Office"
    assert row["drift"] == [], row["drift"]
    assert body["any_drift"] is False
    assert row["live"]["cat_confidence"] == 0.3
    assert row["live"]["locator_classes"] == ["cat", "dog"]


def test_drift_is_reported_field_by_field(monkeypatch, tmp_path):
    # The failure this exists to catch: the saved config says one thing, the
    # running detector another, and find silently uses the running one.
    app, c = _client(monkeypatch, tmp_path, "Office", cat_confidence=0.3,
                     locator_classes=["cat", "dog"])
    det = _detector_for(app, "Office")
    det.cat_confidence = 0.9                 # as if a reload never landed
    det.locator_classes = ("cat",)           # …and the dog toggle never applied

    body = c.get("/api/diagnostics/detectors").get_json()
    drift = {d["field"]: d for d in body["detectors"][0]["drift"]}
    assert body["any_drift"] is True
    assert drift["cat_confidence"]["live"] == 0.9
    assert drift["cat_confidence"]["saved"] == 0.3
    assert drift["locator_classes"] == {"field": "locator_classes",
                                        "live": ["cat"], "saved": ["cat", "dog"]}


def test_the_find_settings_in_play_are_reported_too(monkeypatch, tmp_path):
    app, c = _client(monkeypatch, tmp_path, "Office")
    _detector_for(app, "Office")
    c.post("/api/config", json={"find_tiling": "3x3", "find_tile_overlap": 0.2,
                                "find_confidence": 0.0, "find_model": ""})
    find = c.get("/api/diagnostics/detectors").get_json()["find"]
    assert find["tiling"] == "3x3" and find["tile_overlap"] == 0.2
    # An empty model/confidence means "each camera's own" — say that, don't print 0.
    assert find["model"] == "(each camera's own)"
    assert find["confidence"] == "(each camera's own)"


def test_it_survives_a_stopped_loop(monkeypatch, tmp_path):
    app, c = _client(monkeypatch, tmp_path, "Office")
    app.config["loop"]._detectors = {}
    body = c.get("/api/diagnostics/detectors").get_json()
    assert body["detectors"] == [] and body["any_drift"] is False
