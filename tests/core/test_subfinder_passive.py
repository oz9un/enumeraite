"""Optional subfinder integration tested with local fixture executables only."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from enumeraite.core import passive


@pytest.fixture(autouse=True)
def public_sources_are_fixtures(monkeypatch):
    monkeypatch.setattr(
        passive,
        "_fetch_bytes",
        lambda url, *args, **kwargs: b"[]" if "crt.sh" in url else b"No records found",
    )
    monkeypatch.setattr(passive.shutil, "which", lambda name: None)


def install_executable(monkeypatch, tmp_path, code):
    script = tmp_path / "fixture-subfinder"
    script.write_text(f"#!{sys.executable}\n" + code, encoding="utf-8")
    script.chmod(0o700)
    monkeypatch.setattr(
        passive.shutil, "which", lambda name: str(script) if name == "subfinder" else None
    )
    original_popen = passive.subprocess.Popen
    processes = []

    def record_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append((process, args, kwargs))
        return process

    monkeypatch.setattr(passive.subprocess, "Popen", record_process)
    return processes


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_installed_subfinder_merges_scoped_names_and_provenance(monkeypatch, tmp_path):
    processes = install_executable(
        monkeypatch,
        tmp_path,
        "print('z.example.test\\nA.example.test\\n*.dev.example.test\\nexample.test.evil.test\\nexample.test\\na.example.test\\nshop.bücher.example.test')\n",
    )
    monkeypatch.setattr(
        passive,
        "_fetch_bytes",
        lambda url, *args, **kwargs: (
            b'[{"name_value":"a.example.test"}]'
            if "crt.sh" in url
            else b"mail.example.test,192.0.2.1"
        ),
    )
    result = passive.collect_passive("EXAMPLE.test.", timeout=1)
    assert result.names == [
        "a.example.test",
        "dev.example.test",
        "mail.example.test",
        "shop.xn--bcher-kva.example.test",
        "z.example.test",
    ]
    assert result.sources["subfinder"] == [
        "a.example.test",
        "dev.example.test",
        "shop.xn--bcher-kva.example.test",
        "z.example.test",
    ]
    assert result.sources["crt.sh"] == ["a.example.test"]
    assert not result.errors and not result.truncated
    assert processes[0][0].returncode == 0


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_arguments_environment_and_configs_are_keyless_isolated_and_removed(monkeypatch, tmp_path):
    capture = tmp_path / "captured-fixture.json"
    code = "import json, os, sys\nfrom pathlib import Path\n"
    code += f"Path({str(capture)!r}).write_text(json.dumps({{'args':sys.argv, 'env':dict(os.environ), 'config':Path(os.environ['SUBFINDER_CONFIG']).read_text(), 'providers':Path(os.environ['SUBFINDER_PROVIDER_CONFIG']).read_text()}}))\n"
    code += "print('safe.example.test')\n"
    processes = install_executable(monkeypatch, tmp_path, code)
    for name in [
        "MINIMAX_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HTTP_PROXY",
        "SUBFINDER_PROVIDER_CONFIG",
        "SUBFINDER_CONFIG",
    ]:
        monkeypatch.setenv(name, "must-not-be-inherited-fixture")
    result = passive.collect_passive("example.test", timeout=1)
    assert result.names == ["safe.example.test"]
    captured = json.loads(capture.read_text())
    assert "must-not-be-inherited-fixture" not in json.dumps(captured)
    args = captured["args"]
    assert args[args.index("-d") + 1] == "example.test"
    assert "-silent" in args and "-duc" in args
    assert args[args.index("-max-time") + 1] == "1"
    assert 1 <= int(args[args.index("-timeout") + 1]) <= 10
    assert not any(flag in args for flag in ["-active", "-all", "-up", "-o"])
    assert captured["config"] == captured["providers"] == "{}\n"
    for name in ["HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "APPDATA"]:
        assert captured["env"][name] == captured["env"]["HOME"]
    assert not Path(captured["env"]["HOME"]).exists()
    _, _, kwargs = processes[0]
    assert kwargs["shell"] is False
    assert kwargs["stdin"] == passive.subprocess.DEVNULL
    assert kwargs["start_new_session"]


def test_absent_executable_is_optional_not_an_error(monkeypatch):
    monkeypatch.setattr(
        passive.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("No optional binary should run"),
    )
    result = passive.collect_passive("example.test")
    assert "subfinder" not in result.sources
    assert result.errors == []


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_relative_path_entry_still_uses_the_discovered_executable(monkeypatch, tmp_path):
    install_executable(monkeypatch, tmp_path, "print('found.example.test')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(passive.shutil, "which", lambda name: "./fixture-subfinder")
    result = passive.collect_passive("example.test", timeout=1)
    assert result.names == ["found.example.test"]
    assert not result.errors


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_result_limit_retains_sorted_partial_names_and_kills_and_reaps(monkeypatch, tmp_path):
    processes = install_executable(
        monkeypatch,
        tmp_path,
        "import time\nprint('z.example.test\\na.example.test\\nb.example.test',flush=True)\ntime.sleep(20)\n",
    )
    start = time.monotonic()
    result = passive.collect_passive("example.test", timeout=1, max_results=2)
    assert time.monotonic() - start < 0.8
    assert result.names == ["a.example.test", "z.example.test"]
    assert result.sources["subfinder"] == result.names
    assert result.truncated
    assert any("result limit" in error for error in result.errors)
    assert processes[0][0].returncode is not None
    assert processes[0][0].stdout.closed and processes[0][0].stderr.closed


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_timeout_retains_partial_names_and_reaps_before_return(monkeypatch, tmp_path):
    processes = install_executable(
        monkeypatch,
        tmp_path,
        "import time\nprint('known.example.test',flush=True)\ntime.sleep(20)\n",
    )
    start = time.monotonic()
    result = passive.collect_passive("example.test", timeout=0.4)
    assert time.monotonic() - start < 0.7
    assert result.names == ["known.example.test"]
    assert result.truncated
    assert any("timed out" in error for error in result.errors)
    assert processes[0][0].returncode is not None


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_both_output_streams_share_a_byte_cap_and_never_expose_stderr(
    monkeypatch, tmp_path, stream
):
    code = "import sys,time\nprint('known.example.test',flush=True)\ntime.sleep(0.02)\n"
    code += f"sys.{stream}.write('private-fixture-secret-' * 2000);sys.{stream}.flush()\ntime.sleep(20)\n"
    processes = install_executable(monkeypatch, tmp_path, code)
    monkeypatch.setattr(passive, "_MAX_RESPONSE_BYTES", 1024)
    result = passive.collect_passive("example.test", timeout=1)
    assert result.names == ["known.example.test"]
    assert result.truncated
    assert any("safety limit" in error for error in result.errors)
    assert "private-fixture-secret" not in str(result)
    assert processes[0][0].returncode is not None


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_nonzero_exit_retains_names_but_never_logs_secret_stderr(monkeypatch, tmp_path):
    processes = install_executable(
        monkeypatch,
        tmp_path,
        "import sys\nprint('found.example.test')\nprint('fake-private-key and /private/file',file=sys.stderr)\nsys.exit(7)\n",
    )
    result = passive.collect_passive("example.test", timeout=1)
    assert result.names == ["found.example.test"]
    assert result.truncated
    assert result.errors == [
        "subfinder: optional collector exited unsuccessfully; results are incomplete"
    ]
    assert "fake-private-key" not in str(result) and "/private/file" not in str(result)
    assert processes[0][0].returncode == 7


@pytest.mark.skipif(os.name == "nt", reason="Fixture executables use POSIX shebangs")
def test_malformed_long_unicode_and_out_of_scope_lines_are_filtered(monkeypatch, tmp_path):
    code = "import sys\nsys.stdout.buffer.write(b'bad_line.example.test\\n'+b'x'*5000+b'\\n'+b'\\xff.example.test\\n'+b'bad.example.test.evil\\n'+b'valid.example.test')\n"
    install_executable(monkeypatch, tmp_path, code)
    result = passive.collect_passive("example.test", timeout=1)
    assert result.names == ["valid.example.test"]
    assert result.truncated
    assert result.errors == [
        "subfinder: ignored malformed collector output; results are incomplete"
    ]


def test_start_failure_is_generic_and_never_leaks_path(monkeypatch):
    monkeypatch.setattr(
        passive.shutil, "which", lambda name: "/private/nonexistent/confidential-subfinder"
    )
    result = passive.collect_passive("example.test", timeout=1)
    assert result.sources["subfinder"] == []
    assert result.truncated
    assert "could not run" in result.errors[0]
    assert "/private/" not in str(result)


def test_expired_deadline_never_starts_installed_binary(monkeypatch):
    monkeypatch.setattr(passive.shutil, "which", lambda name: "/fake/subfinder")
    monkeypatch.setattr(
        passive.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Expired collector must not launch"),
    )
    result = passive.collect_passive("example.test", deadline=time.monotonic() - 1)
    assert result.sources["subfinder"] == []
    assert result.truncated


def test_invalid_domain_cannot_become_subprocess_flags(monkeypatch):
    monkeypatch.setattr(
        passive.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Invalid target must not launch"),
    )
    with pytest.raises(ValueError):
        passive.collect_passive("example.test;touch /tmp/unwanted")
