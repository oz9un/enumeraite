import json
import os
import stat
from pathlib import Path

import pytest

from enumeraite.core import scan_state
from enumeraite.core.scan_state import ScanStore, default_state_path


def test_state_roundtrip_and_private_permissions(tmp_path):
    path = tmp_path / "private" / "scan.json"
    state = {"target": "example.com", "nodes": [{"value": "api.example.com"}], "amount": 0.25}
    store = ScanStore(path)
    store.save(state)
    assert store.load() == state
    assert not list(path.parent.glob(".scan-*"))
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_store_replaces_atomically_and_preserves_previous_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "scan.json"
    store = ScanStore(path)
    store.save({"stage": "before"})

    def fail_replace(source, destination):
        assert Path(destination) == path
        assert json.loads(path.read_text()) == {"stage": "before"}
        assert json.loads(Path(source).read_text()) == {"stage": "after"}
        raise OSError("simulated replace failure")

    monkeypatch.setattr(scan_state.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        store.save({"stage": "after"})
    assert store.load() == {"stage": "before"}
    assert not list(tmp_path.glob(".scan-*"))


def test_state_flush_failure_preserves_old_file_and_cleans_temporary(tmp_path, monkeypatch):
    store = ScanStore(tmp_path / "scan.json")
    store.save({"old": True})

    def fail_fsync(descriptor):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(scan_state.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="disk failure"):
        store.save({"new": True})
    assert store.load() == {"old": True}
    assert not list(tmp_path.glob(".scan-*"))


def test_invalid_serialization_cannot_damage_existing_state(tmp_path):
    store = ScanStore(tmp_path / "scan.json")
    store.save({"old": True})
    with pytest.raises(ValueError):
        store.save({"amount": float("nan")})
    assert store.load() == {"old": True}
    assert not list(tmp_path.glob(".scan-*"))


@pytest.mark.parametrize("raw", ["not-json", "[]", "null", '"text"'])
def test_load_rejects_invalid_json_or_nonobject(tmp_path, raw):
    path = tmp_path / "scan.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        ScanStore(path).load()


def test_state_read_has_a_size_limit(tmp_path):
    path = tmp_path / "scan.json"
    with path.open("wb") as handle:
        handle.truncate(32_000_001)
    with pytest.raises(ValueError, match="byte limit"):
        ScanStore(path).load()


def test_lock_excludes_other_scan_and_is_released_after_exception(tmp_path):
    path = tmp_path / "scan.json"
    first = ScanStore(path)
    second = ScanStore(path)
    with pytest.raises(RuntimeError, match="interrupted"):
        with first.locked(resume=False):
            with pytest.raises(ValueError, match="already running"):
                with second.locked(resume=False):
                    pytest.fail("concurrent scan acquired a locked state")
            raise RuntimeError("interrupted")
    with second.locked(resume=False):
        second.save({"recovered": True})
    assert second.load() == {"recovered": True}


def test_existing_state_requires_explicit_resume(tmp_path):
    store = ScanStore(tmp_path / "scan.json")
    store.save({"previous": True})
    with pytest.raises(ValueError, match="already exists"):
        with store.locked(resume=False):
            pytest.fail("old state was silently reused")
    with store.locked(resume=True):
        assert store.load() == {"previous": True}


@pytest.mark.skipif(os.name == "nt", reason="Unix symlink permissions")
def test_state_symlink_is_rejected_for_reads_and_writes(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"private": true}')
    path = tmp_path / "scan.json"
    path.symlink_to(target)
    store = ScanStore(path)
    with pytest.raises(ValueError, match="symbolic link"):
        store.load()
    with pytest.raises(ValueError, match="symbolic link"):
        store.save({"replace": True})
    assert target.read_text() == '{"private": true}'


@pytest.mark.skipif(os.name == "nt", reason="Unix symlink permissions")
def test_lock_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("unchanged")
    path = tmp_path / "scan.json"
    path.with_suffix(".lock").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        with ScanStore(path).locked(resume=False):
            pytest.fail("followed a lock symlink")
    assert target.read_text() == "unchanged"


@pytest.mark.skipif(os.name == "nt", reason="Unix symlink permissions")
def test_load_refuses_symlink_swapped_between_check_and_open(tmp_path, monkeypatch):
    path = tmp_path / "scan.json"
    path.write_text('{"safe": true}')
    other = tmp_path / "private.json"
    other.write_text('{"private": true}')
    original_open = os.open

    def swap_then_open(name, flags, *args, **kwargs):
        if Path(name) == path:
            path.unlink()
            path.symlink_to(other)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(scan_state.os, "open", swap_then_open)
    with pytest.raises((ValueError, OSError)):
        ScanStore(path).load()


@pytest.mark.skipif(os.name == "nt", reason="Unix FIFOs")
def test_nonregular_state_and_lock_are_rejected(tmp_path):
    path = tmp_path / "scan.json"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        ScanStore(path).load()
    path.unlink()
    os.mkfifo(path.with_suffix(".lock"))
    with pytest.raises((ValueError, OSError)):
        with ScanStore(path).locked(resume=False):
            pytest.fail("nonregular lock target was accepted")


def test_default_state_paths_are_unique_and_resume_selects_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    first = default_state_path("example.com")
    second = default_state_path("example.com")
    assert first.parent == tmp_path / "enumeraite" / "example.com"
    assert first != second
    with pytest.raises(ValueError, match="No saved scan"):
        default_state_path("example.com", resume=True)
    ScanStore(first).save({"first": True})
    ScanStore(second).save({"second": True})
    assert default_state_path("example.com", resume=True) == max(first, second)
