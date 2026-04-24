import asyncio
from types import SimpleNamespace

import httpx

from open_storyline.agent import _build_storyline_tools, _is_mcp_connection_failure


def test_is_mcp_connection_failure_detects_nested_connect_errors() -> None:
    exc = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [httpx.ConnectError("All connection attempts failed")],
    )

    assert _is_mcp_connection_failure(exc) is True


def test_build_storyline_tools_can_fallback_when_local_mcp_is_down(monkeypatch) -> None:
    class _ExplodingMCPClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get_tools(self):
            raise ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [httpx.ConnectError("All connection attempts failed")],
            )

    async def _fake_load_skills(_skill_dir):
        return ["skill-a", "skill-b"]

    monkeypatch.setattr("open_storyline.agent.MultiServerMCPClient", _ExplodingMCPClient)
    monkeypatch.setattr("open_storyline.agent.load_skills", _fake_load_skills)

    cfg = SimpleNamespace(
        local_mcp_server=SimpleNamespace(
            server_name="storyline",
            server_transport="streamable-http",
            url="http://127.0.0.1:8001/mcp",
            timeout=60,
        ),
        skills=SimpleNamespace(skill_dir="./.storyline/skills"),
    )

    async def _run():
        return await _build_storyline_tools(
            cfg,
            "session-1",
            sampling_callback=None,
            tool_interceptors=[],
            allow_mcp_failure=True,
        )

    tools, skills, node_manager = asyncio.run(_run())

    assert tools == []
    assert skills == ["skill-a", "skill-b"]
    assert node_manager.kind_to_node_ids == {}
    assert node_manager.id_to_tool == {}
