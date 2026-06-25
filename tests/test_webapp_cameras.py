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
