import asyncio
from types import SimpleNamespace

from open_storyline.mcp.local_server_manager import LocalMCPServerManager


def test_local_mcp_manager_detects_when_it_should_manage_streamable_http() -> None:
    manager = LocalMCPServerManager(
        cwd="/tmp/workspace",
        host="127.0.0.1",
        port=8001,
        transport="streamable-http",
    )

    assert manager.should_manage is True


def test_local_mcp_manager_skips_non_local_hosts() -> None:
    manager = LocalMCPServerManager(
        cwd="/tmp/workspace",
        host="192.168.1.20",
        port=8001,
        transport="streamable-http",
    )

    assert manager.should_manage is False


def test_local_mcp_manager_can_spawn_and_report_ready(monkeypatch) -> None:
    states = iter([False, True])

    class _FakeProc:
        def __init__(self):
            self.returncode = None
            self.stdout = None
            self.stderr = None

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    fake_proc = _FakeProc()

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    async def _fake_read_stream(self, stream, label):
        return None

    def _fake_is_port_open(self):
        return next(states, True)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(LocalMCPServerManager, "_read_stream", _fake_read_stream)
    monkeypatch.setattr(LocalMCPServerManager, "_is_port_open", _fake_is_port_open)

    manager = LocalMCPServerManager(
        cwd="/tmp/workspace",
        host="127.0.0.1",
        port=8001,
        transport="streamable-http",
        startup_timeout=1.0,
        poll_interval=0.0,
    )

    ok = asyncio.run(manager.ensure_started())

    assert ok is True
    assert manager.status == "running"
    assert manager.snapshot()["port"] == 8001


def test_local_mcp_status_endpoint_returns_snapshot() -> None:
    import agent_fastapi
    from fastapi.testclient import TestClient

    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.local_mcp_manager = SimpleNamespace(
            snapshot=lambda: {"status": "running", "port": 8001}
        )
        response = client.get("/api/meta/local_mcp_status")

    assert response.status_code == 200
    assert response.json() == {"status": "running", "port": 8001}
