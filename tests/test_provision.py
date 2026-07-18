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
def test_onnx_lineup_is_both_precisions_of_every_base_model():
    # #90 hid the _fp16 entries from pickers, but they stay in the LINEUP:
    # base + fp16 for each of the 3 logical models.
    lineup = {i["variant"]: i for i in provision.onnx_lineup()}
    assert "yolo11n_fp16" in lineup            # the #86 registry gap, closed
    assert lineup["yolo26x_fp16"]["precision"] == "fp16"
    assert lineup["yolo26m"]["precision"] == "fp32"
    assert set(lineup) == {"yolo11n", "yolo11n_fp16", "yolo26m", "yolo26m_fp16",
                           "yolo26x", "yolo26x_fp16"}


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


# ---- #109: verify-and-adopt instead of rebuilding a valid file ----------------------
def _explode(*a, **k):
    raise AssertionError("rebuilt a file that should have been adopted")


def test_unverified_file_is_adopted_not_rebuilt(tmp_path, monkeypatch):
    # A present-but-unmanifested file that passes its own verification is stamped
    # into the manifest in place — no multi-minute regeneration (#109).
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    monkeypatch.setattr(provision, "_verify_existing",
                        lambda path, **kw: (True, "golden head (stubbed)"))
    monkeypatch.setattr(provision, "_require_ultralytics", _explode)

    msgs = []
    rows = provision.provision(targets=["yolo26x.onnx"], models_dir=mdir,
                               progress=msgs.append)

    assert {r["file"]: r for r in rows}["yolo26x.onnx"]["status"] == "ok"
    assert any("adopted" in m for m in msgs)
    # Adoption is recorded as local provenance, not in the committed repo manifest.
    assert "yolo26x.onnx" in provision.load_local_manifest(models_dir=mdir)
    assert "yolo26x.onnx" not in _read_repo_manifest(mdir)


def test_unverifiable_file_is_never_silently_adopted(tmp_path, monkeypatch):
    # The invariant: a file we cannot vouch for is rebuilt, never blessed.
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    monkeypatch.setattr(provision, "_verify_existing",
                        lambda path, **kw: (False, "NOT a golden export"))
    monkeypatch.setattr(provision, "ultralytics_available", lambda: False)

    msgs = []
    with pytest.raises(RuntimeError):        # falls through to the rebuild path
        provision.provision(targets=["yolo26x.onnx"], models_dir=mdir,
                            progress=msgs.append)
    assert provision.load_local_manifest(models_dir=mdir) == {}     # nothing stamped
    assert any("can't adopt" in m for m in msgs)


def test_stale_file_is_rebuilt_not_adopted(tmp_path, monkeypatch):
    # `stale` (hash differs from its entry) means the file changed since vetting —
    # it must regenerate, not take the adopt shortcut (#109 is only for unverified).
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    provision.save_manifest({"yolo26x.onnx": {"sha256": "not-the-real-hash",
                                              "kind": "onnx", "precision": "fp32"}},
                            models_dir=mdir)
    assert {r["file"]: r for r in provision.audit(models_dir=mdir)
            }["yolo26x.onnx"]["status"] == "stale"
    monkeypatch.setattr(provision, "_verify_existing", _explode)   # must not be consulted
    monkeypatch.setattr(provision, "ultralytics_available", lambda: False)
    with pytest.raises(RuntimeError):
        provision.provision(targets=["yolo26x.onnx"], models_dir=mdir,
                            progress=lambda m: None)


def test_adoption_needs_no_build_time_deps(tmp_path, monkeypatch):
    # Adopting a valid file must not require ultralytics at all — that dep is only
    # needed to actually build something.
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    monkeypatch.setattr(provision, "ultralytics_available", lambda: False)
    monkeypatch.setattr(provision, "_verify_existing", lambda path, **kw: (True, "ok"))
    rows = provision.provision(targets=["yolo26x.onnx"], models_dir=mdir,
                               progress=lambda m: None)
    assert {r["file"]: r for r in rows}["yolo26x.onnx"]["status"] == "ok"


# ---- #109: local manifest survives a repo-manifest reset ----------------------------
def _read_repo_manifest(mdir):
    return provision._read_json(os.path.join(mdir, provision.MANIFEST_NAME))


def test_local_manifest_survives_a_repo_manifest_reset(tmp_path):
    # Simulates `git reset --hard`: the committed manifest is replaced by upstream's
    # (which only knows the bundled files), but local provenance is untouched.
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    sha = provision._sha256(os.path.join(mdir, "yolo26x.onnx"))
    provision.save_local_manifest(
        {"yolo26x.onnx": {"sha256": sha, "kind": "onnx", "precision": "fp32"}},
        models_dir=mdir)
    provision.save_manifest({"yolo11n.onnx": {"sha256": "upstream"}}, models_dir=mdir)

    by_file = {r["file"]: r for r in provision.audit(models_dir=mdir)}
    assert by_file["yolo26x.onnx"]["status"] == "ok"     # still vouched for locally


def test_local_manifest_entry_wins_over_the_repo_one(tmp_path):
    mdir = str(tmp_path)
    _write(mdir, "yolo26x.onnx")
    sha = provision._sha256(os.path.join(mdir, "yolo26x.onnx"))
    provision.save_manifest({"yolo26x.onnx": {"sha256": "stale-upstream-hash"}},
                            models_dir=mdir)
    provision.save_local_manifest({"yolo26x.onnx": {"sha256": sha}}, models_dir=mdir)
    assert provision.load_manifest(models_dir=mdir)["yolo26x.onnx"]["sha256"] == sha
    assert {r["file"]: r for r in provision.audit(models_dir=mdir)
            }["yolo26x.onnx"]["status"] == "ok"


def test_verify_existing_rejects_a_garbage_file(tmp_path):
    # Real (unstubbed) verification: a non-model file is rejected rather than
    # raising out of provision() — the caller rebuilds it.
    p = tmp_path / "yolo26x.onnx"
    p.write_bytes(b"definitely not an onnx model")
    ok, note = provision._verify_existing(str(p), kind="onnx", size=640,
                                          precision="fp32")
    assert ok is False and note


def test_verify_engine_is_fail_safe(tmp_path):
    # No tensorrt (or a bad file) must report "can't vouch", never adopt on faith.
    p = tmp_path / "yolo26x.engine"
    p.write_bytes(b"not a serialized engine")
    ok, note = provision._verify_engine(str(p))
    assert ok is False and note


def test_local_manifest_is_gitignored():
    # The whole point of the split: git must not track this machine's provenance.
    with open(".gitignore", encoding="utf-8") as fh:
        assert provision.LOCAL_MANIFEST_NAME in fh.read()


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
