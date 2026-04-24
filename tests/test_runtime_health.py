import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

import agent_fastapi
from open_storyline.runtime_health import (
    build_runtime_health_snapshot,
    probe_localai_job_endpoint,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._json_data if self._json_data is not None else {}


def _cfg():
    auth_value = "fixture-auth"
    return SimpleNamespace(
        codex=SimpleNamespace(binary="codex"),
        generate_voiceover=SimpleNamespace(
            providers={
                "localai": {
                    "mode": "gateway",
                    "base_url": "http://127.0.0.1:18080",
                    "api_key": auth_value,
                }
            }
        ),
        select_bgm=SimpleNamespace(
            providers={
                "localai": {
                    "mode": "gateway",
                    "base_url": "http://127.0.0.1:18080",
                    "api_key": auth_value,
                }
            }
        ),
    )


def test_probe_localai_job_endpoint_treats_validation_error_as_ready(monkeypatch) -> None:
    auth_value = "fixture-auth"

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        assert url == "http://127.0.0.1:18080/api/v1/jobs/tts"
        assert headers["Authorization"] == f"Bearer {auth_value}"
        assert json == {}
        return _FakeResponse(422, text='{"detail":"TTS script must include text or ssml"}')

    monkeypatch.setattr("open_storyline.runtime_health.requests.post", _fake_post)

    result = probe_localai_job_endpoint(
        base_url="http://127.0.0.1:18080",
        api_key=auth_value,
        job_type="tts",
    )

    assert result["reachable"] is True
    assert result["ready"] is True
    assert result["status"] == "ready"


def test_probe_localai_job_endpoint_marks_auth_failures_unavailable(monkeypatch) -> None:
    auth_value = "broken-auth"

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        return _FakeResponse(401, text="unauthorized")

    monkeypatch.setattr("open_storyline.runtime_health.requests.post", _fake_post)

    result = probe_localai_job_endpoint(
        base_url="http://127.0.0.1:18080",
        api_key=auth_value,
        job_type="music",
    )

    assert result["reachable"] is True
    assert result["ready"] is False
    assert result["status"] == "auth_error"


def test_build_runtime_health_snapshot_includes_expected_sections() -> None:
    async def _fake_codex_account():
        return {"signed_in": True, "account": {"email": "user@example.com"}}

    snapshot = asyncio.run(
        build_runtime_health_snapshot(
            cfg=_cfg(),
            firered_port=7861,
            local_mcp_manager=SimpleNamespace(
                snapshot=lambda: {
                    "status": "running",
                    "host": "127.0.0.1",
                    "port": 8001,
                    "transport": "streamable-http",
                    "should_manage": True,
                    "last_error": "",
                    "recent_logs": [],
                }
            ),
            socket_checker=lambda host, port, timeout=0.25: True,
            localai_probe=lambda **kwargs: {
                "status": "ready",
                "reachable": True,
                "ready": True,
                "detail": "",
            },
            codex_account_reader=_fake_codex_account,
        )
    )

    assert set(snapshot.keys()) == {
        "firered_web",
        "local_mcp",
        "localai_gateway",
        "localai_tts_ready",
        "localai_music_ready",
        "comfyui_bridge",
        "codex_auth_state",
        "warnings",
    }
    assert snapshot["codex_auth_state"]["signed_in"] is True
    assert snapshot["localai_tts_ready"]["ready"] is True
    assert snapshot["localai_music_ready"]["ready"] is True
    assert snapshot["warnings"] == []


def test_runtime_health_endpoint_returns_snapshot(monkeypatch) -> None:
    async def _fake_health_snapshot(**kwargs):
        return {
            "firered_web": {"status": "running", "ready": True},
            "local_mcp": {"status": "running", "ready": True},
            "localai_gateway": {"status": "running", "ready": True},
            "localai_tts_ready": {"status": "ready", "ready": True},
            "localai_music_ready": {"status": "ready", "ready": True},
            "comfyui_bridge": {"status": "running", "ready": True},
            "codex_auth_state": {"signed_in": False, "status": "signed_out"},
            "warnings": [],
        }

    monkeypatch.setattr(agent_fastapi, "build_runtime_health_snapshot", _fake_health_snapshot)

    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.cfg = _cfg()
        agent_fastapi.app.state.local_mcp_manager = SimpleNamespace(snapshot=lambda: {"status": "running"})
        agent_fastapi.app.state.codex_auth_client = SimpleNamespace(account_read=lambda: None)
        response = client.get("/api/meta/runtime_health")

    assert response.status_code == 200
    body = response.json()
    assert body["firered_web"]["ready"] is True
    assert body["localai_music_ready"]["status"] == "ready"
