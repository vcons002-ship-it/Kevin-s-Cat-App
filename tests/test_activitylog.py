"""Unit tests for the persistent activity log (no GUI, no camera)."""

from d20app.activitylog import ActivityLog


def test_add_and_newest_first(tmp_path):
    log = ActivityLog(path=str(tmp_path / "a.log"))
    log.add("info", "first")
    log.add("treat", "second")
    msgs = [e["message"] for e in log.entries()]
    assert msgs == ["second", "first"]          # newest first


def test_limit_and_order(tmp_path):
    log = ActivityLog(path=str(tmp_path / "a.log"))
    for i in range(5):
        log.add("roll", f"r{i}")
    assert [e["message"] for e in log.entries(limit=2)] == ["r4", "r3"]
    assert [e["message"] for e in log.entries(newest_first=False)] == [
        "r0", "r1", "r2", "r3", "r4"
    ]


def test_unknown_kind_becomes_info(tmp_path):
    log = ActivityLog(path=str(tmp_path / "a.log"))
    entry = log.add("bogus", "hi")
    assert entry["kind"] == "info"
    assert log.add("treat", "yum")["kind"] == "treat"


def test_bounded_in_memory(tmp_path):
    log = ActivityLog(path=str(tmp_path / "a.log"), max_entries=10)
    for i in range(50):
        log.add("roll", f"r{i}")
    entries = log.entries()
    assert len(entries) == 10
    assert entries[0]["message"] == "r49"       # newest kept
    assert entries[-1]["message"] == "r40"      # oldest-kept


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "a.log")
    first = ActivityLog(path=path)
    first.add("treat", "saved event")
    # A fresh instance loads the history back from disk.
    second = ActivityLog(path=path)
    assert [e["message"] for e in second.entries()] == ["saved event"]


def test_file_compacted_on_reload(tmp_path):
    path = str(tmp_path / "a.log")
    writer = ActivityLog(path=path, max_entries=5)
    for i in range(20):
        writer.add("roll", f"r{i}")
    # Reloading with the same small cap should trim the file to the kept tail.
    reloaded = ActivityLog(path=path, max_entries=5)
    assert len(reloaded.entries()) == 5
    with open(path, encoding="utf-8") as fh:
        assert len([ln for ln in fh if ln.strip()]) == 5


def test_clear_empties_memory_and_file(tmp_path):
    path = str(tmp_path / "a.log")
    log = ActivityLog(path=path)
    log.add("info", "x")
    log.clear()
    assert log.entries() == []
    assert ActivityLog(path=path).entries() == []   # nothing reloaded
