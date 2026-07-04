"""TensorRT accelerator (#82): driver guard, engine metadata handling, loud
fallback. The runner itself needs an NVIDIA GPU + a per-machine engine, so CI
covers everything AROUND it (guards, parsing, fallback chain, error wording);
the actual engine inference is NAS-only and flagged in the PR checklist.
"""

import json
import struct
import sys
import types

import pytest

from d20app import yolo


# ---- registry / config surface -------------------------------------------------
def test_tensorrt_is_a_known_accelerator():
    assert "tensorrt" in yolo.ACCELERATORS


def test_engine_path_is_per_variant():
    assert yolo.engine_path("yolo26x_fp16").endswith("yolo26x_fp16.engine")


# ---- driver-capability parsing --------------------------------------------------
def test_parse_cuda_version_from_nvidia_smi_header():
    header = ("| NVIDIA-SMI 610.43.02    Driver Version: 610.43.02    "
              "CUDA Version: 13.3 |")
    assert yolo._parse_cuda_version(header) == 13.3
    assert yolo._parse_cuda_version("... CUDA Version: 12.2 ...") == 12.2
    assert yolo._parse_cuda_version("no gpu here") is None
    assert yolo._parse_cuda_version("") is None


def test_parse_cuda_version_accepts_the_new_umd_label():
    # Driver 610.x relabelled the header field (#85) — the guard must read both.
    header = ("| NVIDIA-SMI 610.43.02    KMD Version: 610.43.02    "
              "CUDA UMD Version: 13.3 |")
    assert yolo._parse_cuda_version(header) == 13.3


def test_driver_probe_survives_missing_nvidia_smi(monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setitem(sys.modules, "torch", None)   # no torch fallback either
    assert yolo._driver_cuda_version() is None


def test_driver_probe_falls_back_to_a_working_cuda_torch(monkeypatch):
    # The nvidia-smi header is fragile (relabelled once already, #85): a torch
    # that is actually RUNNING CUDA proves the driver's ceiling as a fallback.
    import subprocess

    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_torch.version = types.SimpleNamespace(cuda="13.0")
    monkeypatch.setattr(subprocess, "run", no_smi)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert yolo._driver_cuda_version() == 13.0


# ---- ultralytics engine metadata header ------------------------------------------
def test_strip_engine_metadata_removes_ultralytics_header():
    meta = json.dumps({"description": "Ultralytics", "batch": 1}).encode()
    payload = b"\x7fTRT-serialized-engine-bytes"
    wrapped = struct.pack("<I", len(meta)) + meta + payload
    assert yolo._strip_engine_metadata(wrapped) == payload


def test_strip_engine_metadata_passes_bare_engines_through():
    bare = b"\x00\x00\x00\x99not-json" + b"x" * 64      # length points at non-JSON
    assert yolo._strip_engine_metadata(bare) == bare
    tiny = b"abc"
    assert yolo._strip_engine_metadata(tiny) == tiny


# ---- the guard ladder: driver -> package -> engine file ---------------------------
def test_old_driver_refuses_with_the_breaks_torch_warning(monkeypatch):
    monkeypatch.setattr(yolo, "_driver_cuda_version", lambda: 12.2)
    with pytest.raises(RuntimeError) as exc:
        yolo._load_tensorrt("yolo26m")
    msg = str(exc.value)
    assert "12.2" in msg and "break torch" in msg and "NOT pip-install" in msg


def test_no_driver_refuses_cleanly(monkeypatch):
    monkeypatch.setattr(yolo, "_driver_cuda_version", lambda: None)
    with pytest.raises(RuntimeError) as exc:
        yolo._load_tensorrt("yolo26m")
    assert "no working NVIDIA driver" in str(exc.value)


def test_missing_package_names_the_install_and_its_precondition(monkeypatch):
    monkeypatch.setattr(yolo, "_driver_cuda_version", lambda: 13.3)
    # tensorrt genuinely isn't installed in CI — the import fails naturally
    with pytest.raises(RuntimeError) as exc:
        yolo._load_tensorrt("yolo26m")
    msg = str(exc.value)
    assert "pip install tensorrt" in msg and "never on an older driver" in msg


def test_missing_engine_points_at_the_build_script(monkeypatch):
    monkeypatch.setattr(yolo, "_driver_cuda_version", lambda: 13.3)
    # satisfy the imports with stub modules so the check reaches the engine file
    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))
    cuda_pkg = types.ModuleType("cuda")
    cuda_pkg.cudart = types.ModuleType("cuda.cudart")
    monkeypatch.setitem(sys.modules, "cuda", cuda_pkg)
    monkeypatch.setitem(sys.modules, "cuda.cudart", cuda_pkg.cudart)
    with pytest.raises(RuntimeError) as exc:
        yolo._load_tensorrt("yolo26m")
    msg = str(exc.value)
    assert "export_trt_engine.py" in msg and "GPU-specific" in msg


# ---- load_net: tensorrt degrades to 'auto', never dies, never installs -----------
def test_load_net_tensorrt_falls_back_loudly(caplog):
    # CI has no NVIDIA driver: the tensorrt path must fall back to 'auto'
    # (which lands on CPU here) and still return a WORKING runner for the
    # same model — accelerator degradation never costs the model (#82).
    import logging

    with caplog.at_level(logging.WARNING):
        net = yolo.load_net("yolo11n", "tensorrt")
    assert net is not None and hasattr(net, "infer")
    assert any("tensorrt unavailable" in r.message for r in caplog.records)


def test_export_script_parses():
    # the build script must at least be valid Python (it only runs on the NAS)
    import py_compile

    py_compile.compile("d20app/models/export_trt_engine.py", doraise=True)
