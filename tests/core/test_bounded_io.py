"""Tests for open-once bounded input handling."""

import os

import pytest

from enumeraite.core.bounded_io import read_bounded_regular_file


def test_bounded_reader_accepts_regular_file_and_rejects_oversize(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"exact bytes")

    assert read_bounded_regular_file(source, 11) == b"exact bytes"
    with pytest.raises(ValueError, match="byte limit"):
        read_bounded_regular_file(source, 10)


def test_bounded_reader_rejects_non_regular_input(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        read_bounded_regular_file(tmp_path, 100)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available")
def test_bounded_reader_rejects_fifo_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        read_bounded_regular_file(fifo, 100)
