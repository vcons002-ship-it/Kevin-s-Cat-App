"""Saved-camera endpoints: persistence, selection, and password masking."""

import d20app.config as config_mod
from d20app.webapp import create_app


def _client(tmp_path, monkeypatch):
    """A Flask test client whose config reads/writes a throwaway file."""
    cfgfile = str(tmp_path / "config.yaml")
    real_load, real_update = config_mod.load, config_mod.update
    monkeypatch.setattr(config_mod, "load", lambda path=cfgfile: real_load(path))
    monkeypatch.setattr(config_mod, "update",
                        lambda values, path=cfgfile: real_update(values, path))
    return create_app().test_client()


def test_save_list_select_delete_and_masking(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)

    c.post("/api/cameras/saved",
           json={"name": "Kitchen", "url": "rtsp://1.2.3.4/s",
                 "username": "admin", "password": "secret"})
    c.post("/api/cameras/saved", json={"name": "Garage", "url": "rtsp://5.6.7.8/s"})

    saved = c.get("/api/cameras/saved").get_json()
    assert {x["name"] for x in saved} == {"Kitchen", "Garage"}
    assert all("password" not in x for x in saved)        # never leak raw passwords
    assert next(x for x in saved if x["name"] == "Kitchen")["has_password"] is True

    # Selecting makes it the active camera; the response never carries a password.
    cfg = c.post("/api/cameras/saved/select", json={"name": "Kitchen"}).get_json()
    assert cfg["camera_url"] == "rtsp://1.2.3.4/s"
    assert "camera_password" not in cfg
    assert all("password" not in cam for cam in cfg["cameras"])

    left = c.post("/api/cameras/saved/delete", json={"name": "Garage"}).get_json()
    assert {x["name"] for x in left} == {"Kitchen"}


def test_blank_password_on_resave_keeps_the_stored_one(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved",
           json={"name": "Cam", "url": "rtsp://a/s", "password": "secret"})
    c.post("/api/cameras/saved", json={"name": "Cam", "url": "rtsp://a/s2"})  # no pw
    cam = next(x for x in config_mod.load(str(tmp_path / "config.yaml")).cameras
               if x["name"] == "Cam")
    assert cam["password"] == "secret" and cam["url"] == "rtsp://a/s2"


def test_main_config_post_cannot_clobber_camera_store(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={"name": "Cam", "url": "rtsp://a/s"})
    c.post("/api/config", json={"cameras": [], "scan_fps": 8})
    assert len(config_mod.load(str(tmp_path / "config.yaml")).cameras) == 1


def test_config_endpoints_never_leak_a_password(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/api/config", json={"camera_password": "topsecret"})
    got = c.get("/api/config").get_json()
    assert "camera_password" not in got


def test_inline_url_credentials_are_masked_in_get_responses(tmp_path, monkeypatch):
    # M1: creds pasted inline in the stream URL (rtsp://user:pass@host) must not
    # leak in cleartext via GET /api/cameras/saved or GET /api/config.
    c = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved",
           json={"name": "Cam", "url": "rtsp://joe:hunter2@1.2.3.4/stream"})

    saved = c.get("/api/cameras/saved").get_json()
    cam = next(x for x in saved if x["name"] == "Cam")
    assert cam["url"] == "rtsp://joe:***@1.2.3.4/stream"   # password masked, host/user kept
    assert "hunter2" not in str(saved)

    # Selecting promotes it to the legacy single-camera camera_url — GET /api/config
    # must mask it there too (both the top-level field and the expanded list).
    c.post("/api/cameras/saved/select", json={"name": "Cam"})
    got = c.get("/api/config").get_json()
    assert got["camera_url"] == "rtsp://joe:***@1.2.3.4/stream"
    assert "hunter2" not in str(got)


def test_resaving_a_masked_camera_url_keeps_the_real_credentials(tmp_path, monkeypatch):
    # The GUI loads the masked URL and re-saves the camera unchanged; the masked
    # "***" must not overwrite the stored inline password (mirrors the
    # blank-password-on-resave guard).
    c = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={"name": "Cam", "url": "rtsp://joe:hunter2@1.2.3.4/s"})
    masked = next(x for x in c.get("/api/cameras/saved").get_json()
                  if x["name"] == "Cam")["url"]
    assert masked == "rtsp://joe:***@1.2.3.4/s"

    c.post("/api/cameras/saved", json={"name": "Cam", "url": masked})   # echo it back
    stored = next(x for x in config_mod.load(str(tmp_path / "config.yaml")).cameras
                  if x["name"] == "Cam")
    assert stored["url"] == "rtsp://joe:hunter2@1.2.3.4/s"             # real creds preserved


def test_editing_a_camera_url_still_saves_the_new_value(tmp_path, monkeypatch):
    # A genuine URL edit (not the masked echo) must go through unchanged.
    c = _client(tmp_path, monkeypatch)
    c.post("/api/cameras/saved", json={"name": "Cam", "url": "rtsp://joe:hunter2@host/s"})
    c.post("/api/cameras/saved", json={"name": "Cam", "url": "rtsp://amy:pw3@other/s"})
    stored = next(x for x in config_mod.load(str(tmp_path / "config.yaml")).cameras
                  if x["name"] == "Cam")
    assert stored["url"] == "rtsp://amy:pw3@other/s"


def test_config_post_with_masked_camera_url_keeps_stored(tmp_path, monkeypatch):
    # Same round-trip guard for the legacy camera_url via POST /api/config.
    c = _client(tmp_path, monkeypatch)
    c.post("/api/config", json={"camera_url": "rtsp://joe:hunter2@1.2.3.4/s"})
    masked = c.get("/api/config").get_json()["camera_url"]
    assert masked == "rtsp://joe:***@1.2.3.4/s"

    c.post("/api/config", json={"camera_url": masked, "scan_fps": 7})   # routine save
    cfg = config_mod.load(str(tmp_path / "config.yaml"))
    assert cfg.camera_url == "rtsp://joe:hunter2@1.2.3.4/s"            # creds preserved
    assert cfg.scan_fps == 7                                          # other fields still land


def test_local_cameras_endpoint(tmp_path, monkeypatch):
    import d20app.discovery as discovery
    monkeypatch.setattr(discovery, "probe_local_cameras",
                        lambda *a, **k: [{"value": "usb:0", "label": "USB camera 0"}])
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/cameras/local").get_json() == [
        {"value": "usb:0", "label": "USB camera 0"}]
