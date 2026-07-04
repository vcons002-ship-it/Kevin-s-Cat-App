"""Model provisioning (#86): the audit states, manifest round-trips, lineup
derivation, and the GUI endpoints. Actual generation needs ultralytics (+ a
GPU for engines) and is NAS-only — CI covers everything around it.
"""

import json
import os

import pytest

from d20app import provision, yolo
from d20app.webapp import create_app


# ---- lineup derives from the registry (no drift) --------------------------------
def test_onnx_lineup_covers_every_selectable_variant():
    lineup = {i["variant"]: i for i in provision.onnx_lineup()}
    assert "yolo11n_fp16" in lineup            # the #86 registry gap, closed
    assert lineup["yolo26x_fp16"]["precision"] == "fp16"
    assert lineup["yolo26m"]["precision"] == "fp32"
    selectable = {v for v, m in yolo.MODELS.items() if m.get("selectable", True)}
    assert set(lineup) == selectable


def test_engine_lineup_is_one_per_base_model():
    files = {i["file"] for i in provision.engine_lineup()}
    assert files == {"yolo11n.engine", "yolo26m.engine", "yolo26x.engine"}


def test_engine_path_shared_between_precisions():
    # engines are always FP16 (#86): both variants map to the one cached file
    assert yolo.engine_path("yolo26x_fp16") == yolo.engine_path("yolo26x")
    assert yolo.engine_path("yolo26x").endswith("yolo26x.engine")


# ---- audit states ----------------------------------------------------------------
def _write(mdir, name, data=b"model-bytes"):
    with open(os.path.join(mdir, name), "wb") as fh:
        fh.write(data)


def test_audit_states_missing_unverified_stale_ok(tmp_path):
    mdir = str(tmp_path)
    by_file = lambda rows, f: next(r for r in rows if r["file"] == f)  # noqa: E731

    rows = provision.audit(models_dir=mdir)
    assert all(r["status"] == "missing" for r in rows)

    _write(mdir, "yolo26m.onnx")                       # present, no manifest
    rows = provision.audit(models_dir=mdir)
    assert by_file(rows, "yolo26m.onnx")["status"] == "unverified"

    manifest = {"yolo26m.onnx": {
        "sha256": provision._sha256(os.path.join(mdir, "yolo26m.onnx")),
        "precision": "fp32", "kind": "onnx"}}
    provision.save_manifest(manifest, models_dir=mdir)
    rows = provision.audit(models_dir=mdir)
    assert by_file(rows, "yolo26m.onnx")["status"] == "ok"

    _write(mdir, "yolo26m.onnx", b"DIFFERENT bytes")   # changed since vetting
    rows = provision.audit(models_dir=mdir)
    assert by_file(rows, "yolo26m.onnx")["status"] == "stale"


def test_engine_rows_are_optional(tmp_path):
    rows = provision.audit(models_dir=str(tmp_path))
    assert all(r["optional"] for r in rows if r["kind"] == "engine")
    assert not any(r["optional"] for r in rows if r["kind"] == "onnx")


def test_bundled_models_are_manifested_ok():
    # the committed manifest vouches for the two bundled files — a fresh clone
    # starts with a verified lineup, not "unverified" bundled models
    by_file = {r["file"]: r for r in provision.audit()}
    assert by_file["yolo11n.onnx"]["status"] == "ok"
    assert by_file["yolo26m.onnx"]["status"] == "ok"


def test_manifest_roundtrip_and_corruption(tmp_path):
    mdir = str(tmp_path)
    provision.save_manifest({"a.onnx": {"sha256": "x"}}, models_dir=mdir)
    assert provision.load_manifest(models_dir=mdir) == {"a.onnx": {"sha256": "x"}}
    with open(os.path.join(mdir, provision.MANIFEST_NAME), "w") as fh:
        fh.write("not json {")
    assert provision.load_manifest(models_dir=mdir) == {}


# ---- provisioning guardrails -------------------------------------------------------
def test_provision_without_ultralytics_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "ultralytics_available", lambda: False)
    with pytest.raises(RuntimeError) as exc:
        provision.provision(models_dir=str(tmp_path))
    msg = str(exc.value)
    assert "ultralytics" in msg and "never installs" in msg


# ---- webapp: audit + provision endpoints -------------------------------------------
def test_api_models_audit_shape():
    body = create_app().test_client().get("/api/models/audit").get_json()
    assert isinstance(body["items"], list) and body["items"]
    assert {"file", "kind", "status", "precision"} <= set(body["items"][0])
    assert isinstance(body["can_provision"], bool)
    assert body["running"] is False


def test_api_provision_without_ultralytics_503(monkeypatch):
    monkeypatch.setattr(provision, "ultralytics_available", lambda: False)
    r = create_app().test_client().post("/api/models/provision", json={})
    assert r.status_code == 503 and "ultralytics" in r.get_json()["error"]


def test_api_provision_status_idle():
    body = create_app().test_client().get("/api/models/provision/status").get_json()
    assert body["running"] is False


def test_model_options_flag_unverified(monkeypatch):
    import d20app.webapp as webapp

    monkeypatch.setattr(provision, "audit", lambda models_dir=None: [
        {"file": "yolo26m.onnx", "kind": "onnx", "status": "unverified"}])
    labels = {m["value"]: m["label"] for m in webapp._model_options()}
    assert "unverified" in labels["yolo26m"]          # says so, never silent
    assert "unverified" not in labels["yolo11n"]
