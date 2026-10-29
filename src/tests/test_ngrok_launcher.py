"""Launcher checks without starting processes, reading secrets or opening tunnels."""

import secrets
from unittest.mock import MagicMock, Mock

import pytest

from scripts import iniciar_ngrok as launcher


@pytest.mark.parametrize("value", ["demo.ngrok-free.app", "https://demo.ngrok-free.app/"])
def test_fixed_domain(value):
    assert launcher.domain_url(value) == "https://demo.ngrok-free.app"


@pytest.mark.parametrize("value", ["", "http://demo.ngrok.app", "https://user@demo.ngrok.app",
                                 "demo.ngrok.app:80", "demo.ngrok.app/path", "--url foo",
                                 "demo.ngrok.app & calc", "demo..app"])
def test_invalid_domain(value):
    with pytest.raises(ValueError):
        launcher.domain_url(value)


def test_waits_for_model():
    api = Mock()
    api.poll.return_value = None
    client = Mock()
    client.get.side_effect = [Mock(status_code=503), Mock(status_code=200, json=lambda: {"status": "ok"})]
    launcher.wait_ready(api, client, timeout=2)
    assert client.get.call_count == 2


def test_dead_api_does_not_wait():
    api = Mock()
    api.poll.return_value = 1
    with pytest.raises(RuntimeError):
        launcher.wait_ready(api, Mock())


@pytest.mark.parametrize("auth_status", [200, 401])
def test_launch_order_and_cleanup(tmp_path, monkeypatch, auth_status):
    (tmp_path / ".env").touch()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "load_dotenv", Mock())
    monkeypatch.setenv("QA_API_TOKEN", secrets.token_urlsafe(32))
    monkeypatch.setenv("QA_TEAM", "fixture")
    monkeypatch.setenv("NGROK_DOMAIN", "fixture.ngrok-free.app")
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "ngrok.exe")
    monkeypatch.setattr(launcher.socket, "socket", MagicMock())
    api, tunnel = Mock(), Mock(returncode=0)
    tunnel.poll.return_value = 0
    spawn = Mock(side_effect=[api, tunnel])
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    ready, stop = Mock(), Mock()
    monkeypatch.setattr(launcher, "wait_ready", ready)
    monkeypatch.setattr(launcher, "stop_owned", stop)
    client_context = MagicMock()
    client_context.__enter__.return_value.get.side_effect = [Mock(status_code=401), Mock(status_code=auth_status)]
    monkeypatch.setattr(launcher.httpx, "Client", lambda **kwargs: client_context)
    result = launcher.main()
    ready.assert_called_once()
    assert spawn.call_count == (2 if auth_status == 200 else 1)
    assert result == (0 if auth_status == 200 else 1)
    assert stop.call_args_list[-1].args == (api,)
    if auth_status == 200:
        command = spawn.call_args_list[1].args[0]
        assert command == ["ngrok.exe", "http", launcher.LOCAL_URL, "--url",
                           "https://fixture.ngrok-free.app", "--inspect=false"]
        assert stop.call_args_list[0].args == (tunnel,)


def test_shutdown_owns_process():
    process = Mock()
    process.poll.return_value = None
    launcher.stop_owned(process)
    process.wait.assert_called_once_with(timeout=10)
