"""Run isolated browser-asset regressions without a browser or live API."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_frontend_fake_api() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is optional; needed only for dashboard asset regressions")
    script = Path(__file__).with_name("dashboard_frontend.test.cjs")
    result = subprocess.run(
        [node, "--test", str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
