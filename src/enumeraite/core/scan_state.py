"""Private, atomic checkpoints for resumable domain scans."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .bounded_io import open_regular_file


def default_state_path(target: str, resume: bool = False) -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    directory = base / "enumeraite" / target
    if resume:
        matches = sorted(directory.glob("scan-*.json"))
        if not matches:
            raise ValueError(f"No saved scan for {target}; start a scan first.")
        return matches[-1]
    from datetime import datetime, timezone
    from secrets import token_hex

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return directory / f"scan-{stamp}-{token_hex(3)}.json"


class ScanStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if self.path.is_symlink():
            raise ValueError("Scan state must not be a symbolic link")
        with open_regular_file(self.path, no_follow=True) as handle:
            try:
                state = json.load(handle, parse_constant=_invalid_json_constant)
            except (ValueError, RecursionError) as exc:
                raise ValueError("Invalid scan checkpoint JSON") from exc
        if not isinstance(state, dict):
            raise ValueError("Scan checkpoint must be a JSON object")
        return state

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("Scan state must not be a symbolic link")
        descriptor, name = tempfile.mkstemp(prefix=".scan-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=True, allow_nan=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def locked(self, *, resume: bool) -> Iterator[None]:
        """Advisory lock released by the OS even if a scan process crashes."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = self.path.with_suffix(".lock")
        if lock_path.is_symlink():
            raise ValueError("Scan lock must not be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        with os.fdopen(descriptor, "r+b") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Scan lock must be a regular file")
            try:
                if os.name == "nt":
                    import msvcrt

                    if os.fstat(handle.fileno()).st_size == 0:
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    getattr(msvcrt, "locking")(handle.fileno(), getattr(msvcrt, "LK_NBLCK"), 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("This scan is already running in another process") from exc
            if self.path.exists() and not resume:
                raise ValueError("State file already exists; use --resume or another --state path")
            yield


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
