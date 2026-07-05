"""Model provisioning (#86): audit what's on disk vs the settled lineup, and
(re)generate what's missing, stale, or unverifiable.

The failure this closes: model files used to arrive three inconsistent ways —
committed (11n/26m, FP32), manually exported (26x), or never generated at all
(every FP16 onnx, every engine) — and the app ran whatever bytes were present
with no check. A stale June FP32 26x was silently benchmarked as "the 26x".
That's the no-silent-worse-detector failure mode at the provisioning layer.

Design:
- ``models_manifest.json`` (committed, next to the models) records each
  generated/vetted file's sha256 + precision + head verdict. The **audit**
  compares disk against it: match → *ok*; hash mismatch → *stale*; file with
  no entry → *unverified* (exactly the stale-manual-export case); absent →
  *missing*. Engines are per-GPU, so their manifest entries are written at
  build time on that machine and never committed.
- **provision()** regenerates whatever the audit flags, from the Ultralytics
  ``.pt`` weights (downloaded on demand), with the golden recipe
  (``end2end=False`` → raw ``(1, 84, 8400)`` head, ``nms=False``), FP16/FP32
  per the variant, then verifies the head and stamps the manifest.
  ``ultralytics`` (+ ``tensorrt``/``cuda-python`` for engines) are BUILD-time
  deps only — the lean install never needs them, and nothing here ever
  pip-installs anything.
- Head verification prefers the ``onnx`` package (shape + no NMS/TopK ops,
  works for FP16 too); without it, FP32 files are checked by a real cv2.dnn
  forward pass and FP16 files are stamped "head unverified here" honestly
  (cv2.dnn's FP16 handling is itself unverified).

Verified in CI: audit states, manifest round-trips, error wording. NAS-only:
actual generation (needs ultralytics + GPU for engines).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

from . import yolo

_log = logging.getLogger("d20.provision")

MANIFEST_NAME = "models_manifest.json"

# sha cache so repeated audits don't re-hash ~100 MB files: path -> (mtime, size, sha)
_SHA_CACHE: dict = {}


def _models_dir(models_dir: str | None) -> str:
    return models_dir or os.path.dirname(yolo.model_path("yolo11n"))


def onnx_lineup() -> list:
    """The settled ONNX set, derived from the registry so it can't drift: both
    precisions of every selectable base model (#90 hid the ``_fp16`` entries
    from pickers — precision is the accelerator's business — but the files are
    still very much part of the lineup)."""
    out = []
    for v, m in yolo.MODELS.items():
        if not m.get("selectable", True):
            continue
        out.append({"variant": v, "file": m["file"], "size": m["size"],
                    "precision": "fp32"})
        fp16 = yolo.MODELS.get(f"{v}_fp16")
        if fp16:
            out.append({"variant": f"{v}_fp16", "file": fp16["file"],
                        "size": fp16["size"], "precision": "fp16"})
    return out


def engine_lineup() -> list:
    """One (FP16) engine per base model — GPU-specific, optional (#82)."""
    bases = sorted(v for v, m in yolo.MODELS.items() if m.get("selectable", True))
    return [{"variant": b, "file": os.path.basename(yolo.engine_path(b))}
            for b in bases]


def _sha256(path: str) -> str:
    st = os.stat(path)
    key = (st.st_mtime, st.st_size)
    cached = _SHA_CACHE.get(path)
    if cached and cached[0] == key:
        return cached[1]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    _SHA_CACHE[path] = (key, sha)
    return sha


def load_manifest(models_dir: str | None = None) -> dict:
    path = os.path.join(_models_dir(models_dir), MANIFEST_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_manifest(manifest: dict, models_dir: str | None = None) -> None:
    path = os.path.join(_models_dir(models_dir), MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")


def audit(models_dir: str | None = None) -> list:
    """Disk vs manifest, one row per settled file.

    Statuses: ``ok`` (present, hash matches the manifest), ``stale`` (present
    but hash differs — the file changed since it was vetted), ``unverified``
    (present with no manifest entry — unknown provenance, the silent-stale-26x
    case), ``missing``. Engine rows carry ``optional: True`` (TensorRT-only)."""
    mdir = _models_dir(models_dir)
    manifest = load_manifest(models_dir)
    rows = []
    for item in onnx_lineup():
        rows.append(_audit_one(mdir, manifest, item["file"], kind="onnx",
                               variant=item["variant"],
                               precision=item["precision"]))
    for item in engine_lineup():
        row = _audit_one(mdir, manifest, item["file"], kind="engine",
                         variant=item["variant"], precision="fp16")
        row["optional"] = True
        rows.append(row)
    return rows


def _audit_one(mdir, manifest, filename, *, kind, variant, precision) -> dict:
    path = os.path.join(mdir, filename)
    row = {"file": filename, "kind": kind, "variant": variant,
           "precision": precision, "optional": False}
    entry = manifest.get(filename)
    if not os.path.exists(path):
        row["status"] = "missing"
        return row
    if not entry or not entry.get("sha256"):
        row["status"] = "unverified"
        row["note"] = "present but not in the manifest — unknown provenance"
        return row
    if _sha256(path) != entry["sha256"]:
        row["status"] = "stale"
        row["note"] = "file changed since it was vetted — regenerate"
        return row
    row["status"] = "ok"
    row["note"] = entry.get("note", "")
    return row


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401 — build-time optional dep

        return True
    except Exception:       # noqa: BLE001 — absent or broken: same answer
        return False


def _verify_golden_head(path: str, size: int, precision: str):
    """(verdict, note): the raw ``(1, 84, N)`` head check, best tool available.

    Prefers the ``onnx`` package (metadata check, FP16-safe); else a real
    cv2.dnn forward for FP32; FP16 without ``onnx`` is honestly 'unverified
    here' (cv2.dnn's FP16 handling is itself unverified)."""
    try:
        import onnx

        m = onnx.load(path)
        bad_ops = [n.op_type for n in m.graph.node
                   if n.op_type in ("NonMaxSuppression", "TopK")]
        dims = [d.dim_value for d in
                m.graph.output[0].type.tensor_type.shape.dim]
        if bad_ops or (len(dims) == 3 and dims[1] > dims[2]):
            raise RuntimeError(
                f"{os.path.basename(path)} is NOT a golden export "
                f"(output {dims}, ops {bad_ops}) — the end2end head the "
                "pipeline mis-decodes (#70 §2)")
        return True, "golden head (onnx metadata)"
    except ImportError:
        pass
    if precision == "fp16":
        return False, ("head unverified here — cv2.dnn FP16 is itself "
                       "unverified; pip install onnx for a metadata check")
    import cv2
    import numpy as np

    net = cv2.dnn.readNetFromONNX(path)
    net.setInput(np.zeros((1, 3, size, size), np.float32))
    out = net.forward()
    if getattr(out, "ndim", 0) != 3 or out.shape[1] > out.shape[2]:
        raise RuntimeError(
            f"{os.path.basename(path)} is NOT a golden export (output shape "
            f"{getattr(out, 'shape', '?')}) — re-export with end2end=False (#70 §2)")
    return True, "golden head (cv2 forward)"


def provision(targets=None, force: bool = False, progress=None,
              models_dir: str | None = None) -> list:
    """Generate every flagged file (or ``targets``); returns the refreshed audit.

    Needs ``ultralytics`` (build-time only) — raises an actionable
    ``RuntimeError`` if absent rather than installing anything. Engines are
    skipped (with a message) unless the driver clears the #82 CUDA-13 gate."""
    say = progress or (lambda msg: _log.info("%s", msg))
    if not ultralytics_available():
        raise RuntimeError(
            "model provisioning needs the 'ultralytics' package (build-time "
            "only; it fetches the .pt weights). Install it on this machine — "
            "pip install ultralytics — then retry. The app never installs it "
            "for you.")
    from ultralytics import YOLO

    mdir = _models_dir(models_dir)
    manifest = load_manifest(models_dir)
    rows = audit(models_dir)
    todo = [r for r in rows
            if (force or r["status"] != "ok")
            and (not targets or r["file"] in targets or r["variant"] in targets)]
    driver = yolo._driver_cuda_version()
    for row in todo:
        path = os.path.join(mdir, row["file"])
        if row["kind"] == "engine":
            if driver is None or driver < yolo._TRT_MIN_CUDA:
                say(f"skip {row['file']}: driver CUDA {driver} < "
                    f"{yolo._TRT_MIN_CUDA:g} — engines need the #82 driver "
                    "upgrade; the onnx lineup still provisions")
                continue
        say(f"building {row['file']} ({row['status']}) …")
        base = row["variant"].replace("_fp16", "")
        model = YOLO(f"{base}.pt")
        model.model.model[-1].end2end = False           # golden head (#70 §2)
        size = yolo.MODELS.get(row["variant"], {}).get("size", 640)
        if row["kind"] == "engine":
            built = model.export(format="engine", imgsz=size, half=True,
                                 simplify=True, dynamic=False, batch=1)
        else:
            built = model.export(format="onnx", imgsz=size, opset=12,
                                 simplify=True, dynamic=False, batch=1,
                                 nms=False, half=(row["precision"] == "fp16"))
        import shutil

        if os.path.abspath(str(built)) != os.path.abspath(path):
            shutil.move(str(built), path)
        note = "generated by provision"
        if row["kind"] == "onnx":
            ok, head_note = _verify_golden_head(path, size, row["precision"])
            note = f"{note}; {head_note}"
            if not ok:
                say(f"  ⚠ {row['file']}: {head_note}")
        manifest[row["file"]] = {
            "sha256": _sha256(path), "precision": row["precision"],
            "kind": row["kind"], "imgsz": size, "note": note,
            "ts": time.time(),
        }
        save_manifest(manifest, models_dir)
        say(f"  done {row['file']}")
    if not todo:
        say("nothing to do — lineup is complete and verified")
    return audit(models_dir)


def _main() -> None:      # pragma: no cover — thin CLI over provision()
    import argparse

    ap = argparse.ArgumentParser(
        description="Audit / generate the settled model lineup (#86).")
    ap.add_argument("targets", nargs="*",
                    help="specific files/variants (default: everything flagged)")
    ap.add_argument("--force", action="store_true", help="regenerate even 'ok' files")
    ap.add_argument("--audit", action="store_true", help="report only, build nothing")
    args = ap.parse_args()
    if args.audit:
        for row in audit():
            print(f"{row['file']:24} {row['kind']:6} {row['status']:10} "
                  f"{row.get('note', '')}")
        return
    provision(targets=args.targets or None, force=args.force, progress=print)


if __name__ == "__main__":
    _main()
