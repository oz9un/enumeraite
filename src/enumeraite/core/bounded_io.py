"""Open-once helpers for regular input files and bounded reads."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Union

PathLike = Union[str, bytes, Path]


@contextmanager
def open_regular_file(
    path: PathLike,
    *,
    not_regular_message: str = "input path must be a regular file",
    no_follow: bool = False,
) -> Iterator[BinaryIO]:
    """Open and inspect the same file descriptor, rejecting FIFOs without waiting."""
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

        handle = os.fdopen(descriptor, "rb")
        descriptor = -1  # The file object now owns and closes it.
        with handle:
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_bounded_regular_file(
    path: PathLike,
    max_bytes: int,
    *,
    not_regular_message: str = "input path must be a regular file",
    too_large_message: str = "input file exceeds the byte limit",
    no_follow: bool = False,
) -> bytes:
    """Read one regular file object under a hard byte limit, even if it grows."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    with open_regular_file(
        path, not_regular_message=not_regular_message, no_follow=no_follow
    ) as handle:
        if os.fstat(handle.fileno()).st_size > max_bytes:
            raise ValueError(too_large_message)
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(too_large_message)
    return raw
