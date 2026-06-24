"""Tests for the snapshot store (save, prune, serve path)."""

import os

from d20app.snapshots import SnapshotStore


def test_save_returns_filename_and_writes_file(tmp_path):
    store = SnapshotStore(directory=str(tmp_path))
    name = store.save(b"\xff\xd8jpegbytes")
    assert name and name.endswith(".jpg")
    with open(store.path(name), "rb") as fh:
        assert fh.read() == b"\xff\xd8jpegbytes"


def test_save_none_or_empty_returns_none(tmp_path):
    store = SnapshotStore(directory=str(tmp_path))
    assert store.save(None) is None
    assert store.save(b"") is None


def test_prune_keeps_only_max_files(tmp_path):
    store = SnapshotStore(directory=str(tmp_path), max_files=5)
    for i in range(20):
        store.save(f"img{i}".encode())
    remaining = [f for f in os.listdir(tmp_path) if f.endswith(".jpg")]
    assert len(remaining) == 5
