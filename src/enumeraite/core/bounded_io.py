"""Open-once helpers for bounded, regular input files."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Union

PathLike = Union[str, bytes, Path]


def read_bounded_regular_file(
    path: PathLike,
    max_bytes: int,
    *,
    not_regular_message: str = "input path must be a regular file",
    too_large_message: str = "input file exceeds the byte limit",
    no_follow: bool = False,
) -> bytes:
    """Read one regular file object under a hard byte limit.

    The descriptor is opened once and then inspected with ``fstat``.  This
    avoids checking one path target and reading another if a file is swapped
    between ``stat`` and ``open``.  Non-blocking mode also prevents a FIFO from
    hanging the process before its type can be rejected.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    flags = os.O_RDONLY
    if no_follow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    for flag_name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NONBLOCK"):
        flags |= getattr(os, flag_name, 0)

    descriptor = os.open(os.fspath(path), flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(not_regular_message)
        if metadata.st_size > max_bytes:
            raise ValueError(too_large_message)

        handle = os.fdopen(descriptor, "rb")
        descriptor = -1  # The file object now owns and closes it.
        with handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(too_large_message)
        return raw
    finally:
        if descriptor >= 0:
            os.close(descriptor)
