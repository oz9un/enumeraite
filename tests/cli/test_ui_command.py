from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from enumeraite.cli.main import cli


def test_ui_command_help_does_not_start_server():
    result = CliRunner().invoke(cli, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--no-open" in result.output
    assert "Local use only" in result.output
    assert "--state-dir" in result.output


def test_ui_port_validation():
    result = CliRunner().invoke(cli, ["ui", "--port", "65536"])
    assert result.exit_code != 0


@pytest.fixture
def launch(monkeypatch):
    state = SimpleNamespace(
        events=[],
        managers=[],
        servers=[],
        browser_urls=[],
        bind_error=False,
        browser_error=None,
        serve_error=None,
    )

    class Manager:
        def __init__(self, *, storage_dir):
            self.storage_dir = storage_dir
            self.closed = False
            state.managers.append(self)
            state.events.append("manager-created")

        def close(self):
            self.closed = True
            state.events.append("manager-closed")

    class Server:
        def __init__(self, manager, *, port):
            if state.bind_error:
                raise OSError("fixture private bind diagnostic")
            self.manager = manager
            self.port = port
            self.closed = False
            self.url = "http://127.0.0.1:54321/"
            self.poll_interval = None
            state.servers.append(self)
            state.events.append("server-created")

        def serve_forever(self, *, poll_interval):
            self.poll_interval = poll_interval
            state.events.append("serving")
            if state.serve_error is not None:
                raise state.serve_error

        def server_close(self):
            self.closed = True
            state.events.append("server-closed")

    def open_browser(url):
        state.browser_urls.append(url)
        state.events.append("browser-opened")
        if state.browser_error is not None:
            raise state.browser_error
        return True

    monkeypatch.setattr("enumeraite.web.manager.ScanManager", Manager)
    monkeypatch.setattr("enumeraite.web.server.DashboardServer", Server)
    monkeypatch.setattr("enumeraite.cli.ui_command.webbrowser.open", open_browser)
    return state


def test_ui_no_open_prints_launch_url_and_closes_resources(launch, tmp_path):
    result = CliRunner().invoke(
        cli, ["ui", "--no-open", "--port", "0", "--state-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert not launch.browser_urls
    assert launch.managers[0].storage_dir == tmp_path
    assert launch.servers[0].port == 0
    assert launch.servers[0].poll_interval == 0.2
    assert launch.servers[0].url in result.output
    assert "No dashboard login needed" in result.output
    assert launch.managers[0].closed and launch.servers[0].closed


def test_ui_opens_browser_by_default_with_same_plain_local_url(launch):
    result = CliRunner().invoke(cli, ["ui"])
    assert result.exit_code == 0, result.output
    assert launch.managers[0].storage_dir is None
    assert launch.servers[0].port == 8765
    assert launch.browser_urls == [launch.servers[0].url]
    assert launch.browser_urls[0] in result.output
    assert "#" not in launch.browser_urls[0]
    assert "token" not in launch.browser_urls[0]
    assert launch.events.index("browser-opened") < launch.events.index("serving")
    assert launch.managers[0].closed and launch.servers[0].closed


def test_ui_ctrl_c_pauses_manager_and_closes_server(launch):
    launch.serve_error = KeyboardInterrupt()
    result = CliRunner().invoke(cli, ["ui", "--no-open"])
    assert result.exit_code == 0, result.output
    assert "Pausing active work and closing the dashboard" in result.output
    assert launch.managers[0].closed and launch.servers[0].closed
    assert launch.events.count("manager-closed") == 1
    assert launch.events.count("server-closed") == 1


def test_ui_ctrl_c_during_browser_launch_still_closes_resources(launch):
    launch.browser_error = KeyboardInterrupt()
    result = CliRunner().invoke(cli, ["ui"])
    assert result.exit_code == 0, result.output
    assert "Pausing active work and closing the dashboard" in result.output
    assert "serving" not in launch.events
    assert launch.managers[0].closed and launch.servers[0].closed


def test_ui_bind_error_is_friendly_closes_manager_and_never_opens_browser(launch):
    launch.bind_error = True
    result = CliRunner().invoke(cli, ["ui"])
    assert result.exit_code == 1
    assert "Could not open local port; try enumeraite ui --port 0" in result.output
    assert "fixture private bind diagnostic" not in result.output
    assert launch.managers[0].closed
    assert launch.browser_urls == [] and launch.servers == []


def test_ui_browser_failure_gives_manual_url_and_still_serves(launch):
    launch.browser_error = RuntimeError("fixture private browser diagnostic")
    result = CliRunner().invoke(cli, ["ui"])
    assert result.exit_code == 0, result.output
    assert "Open the printed URL in your browser" in result.output
    assert "fixture private browser diagnostic" not in result.output
    assert "serving" in launch.events
    assert launch.managers[0].closed and launch.servers[0].closed


def test_ui_unexpected_server_error_still_closes_resources(launch):
    launch.serve_error = RuntimeError("fixture server failure")
    result = CliRunner().invoke(cli, ["ui", "--no-open"])
    assert result.exit_code != 0
    assert launch.managers[0].closed and launch.servers[0].closed
