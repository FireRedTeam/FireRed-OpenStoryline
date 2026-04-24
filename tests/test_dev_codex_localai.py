import argparse
import asyncio
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dev_codex_localai.py"
SPEC = importlib.util.spec_from_file_location("dev_codex_localai", SCRIPT_PATH)
assert SPEC and SPEC.loader
dev_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dev_runtime)


def test_runtime_health_failures_reports_missing_services() -> None:
    snapshot = {
        "firered_web": {"ready": True, "status": "running"},
        "local_mcp": {"ready": False, "status": "missing", "detail": "not reachable"},
        "localai_gateway": {"ready": True, "status": "running"},
        "localai_tts_ready": {"ready": False, "status": "auth_error", "detail": "unauthorized"},
        "localai_music_ready": {"ready": True, "status": "ready"},
        "comfyui_bridge": {"ready": True, "status": "running"},
        "codex_auth_state": {"signed_in": False, "status": "signed_out", "detail": "login required"},
        "warnings": [],
    }

    failures = dev_runtime._runtime_health_failures(snapshot)

    assert "Local MCP: not reachable" in failures
    assert "LocalAI TTS: unauthorized" in failures
    assert "Codex auth: login required" in failures


def test_runtime_health_brief_only_exposes_brief_status() -> None:
    snapshot = {
        "firered_web": {"ready": True, "status": "running", "detail": "ignore"},
        "local_mcp": {"ready": True, "status": "running", "detail": "ignore"},
        "localai_gateway": {"ready": True, "status": "running", "detail": "ignore"},
        "localai_tts_ready": {"ready": True, "status": "ready", "detail": "ignore"},
        "localai_music_ready": {"ready": True, "status": "ready", "detail": "ignore"},
        "comfyui_bridge": {"ready": True, "status": "running", "detail": "ignore"},
        "codex_auth_state": {"signed_in": True, "status": "signed_in", "detail": "ignore"},
        "warnings": ["warn-a"],
    }

    brief = dev_runtime._runtime_health_brief(snapshot)

    assert brief["warnings"] == ["warn-a"]
    assert brief["codex_auth_state"] == {"signed_in": True, "status": "signed_in"}
    assert brief["localai_tts_ready"] == {"ready": True, "status": "ready"}
    assert "detail" not in brief["localai_tts_ready"]


def test_runtime_env_loads_local_dotenv_without_overriding_shell_env(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_local_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=from-env\nGEMINI_API_KEY='from-env-gemini'\n", encoding="utf-8")
    env_local_file.write_text("OPENAI_API_KEY=from-env-local\nNVIDIA_API_KEY=from-env-local-nvidia\n", encoding="utf-8")
    monkeypatch.setattr(dev_runtime, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    env = dev_runtime._runtime_env()

    assert env["OPENAI_API_KEY"] == "from-shell"
    assert env["GEMINI_API_KEY"] == "from-env-gemini"
    assert env["NVIDIA_API_KEY"] == "from-env-local-nvidia"


def test_cmd_up_returns_failure_when_health_not_ready(monkeypatch, capsys) -> None:
    monkeypatch.setattr(dev_runtime, "_start_mcp", lambda port: {"name": "local_mcp", "port": port})
    monkeypatch.setattr(dev_runtime, "_start_web", lambda port: {"name": "firered_web", "port": port})
    monkeypatch.setattr(dev_runtime, "_start_bridge", lambda port, host, target: {"name": "comfyui_bridge", "port": port})
    monkeypatch.setattr(
        dev_runtime,
        "_wait_for_runtime_health",
        lambda **kwargs: {
            "firered_web": {"ready": True, "status": "running"},
            "local_mcp": {"ready": False, "status": "missing", "detail": "not reachable"},
            "localai_gateway": {"ready": True, "status": "running"},
            "localai_tts_ready": {"ready": True, "status": "ready"},
            "localai_music_ready": {"ready": True, "status": "ready"},
            "comfyui_bridge": {"ready": True, "status": "running"},
            "codex_auth_state": {"signed_in": True, "status": "signed_in"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(dev_runtime.asyncio, "run", lambda coro: coro)

    args = argparse.Namespace(
        web_port=7861,
        mcp_port=8001,
        bridge_port=8188,
        bridge_target_host="172.17.0.1",
        bridge_target_port=8188,
        timeout=60.0,
        smoke=False,
        model="gpt-5.4",
        reasoning="medium",
    )

    rc = dev_runtime.cmd_up(args)
    output = capsys.readouterr().out

    assert rc == 1
    assert "\"ok\": false" in output
    assert "Local MCP: not reachable" in output


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeSessionClient:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    async def get(self, path: str) -> _JsonResponse:
        return _JsonResponse(self.snapshot)


def test_ensure_tool_completed_reports_tool_error_info_instead_of_preview_symptom() -> None:
    snapshot = {
        "history": [
            {
                "role": "tool",
                "name": "select_bgm",
                "state": "complete",
                "summary": {
                    "isError": True,
                    "error_info": "RuntimeError: local-ai-platform music asset download http 500: Internal Server Error",
                },
            }
        ]
    }

    with pytest.raises(dev_runtime.SmokeFailure) as exc_info:
        asyncio.run(
            dev_runtime._ensure_tool_completed(
                _FakeSessionClient(snapshot),
                object(),
                session_id="sess-123",
                tool_name="select_bgm",
                model="gpt-5.4",
                reasoning="medium",
                timeout=0.1,
            )
        )

    assert exc_info.value.step == "select_bgm"
    assert "local-ai-platform music asset download http 500" in exc_info.value.message
    assert "preview url missing" not in exc_info.value.message


def test_cmd_verify_returns_success_with_smoke(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        dev_runtime,
        "_fetch_runtime_health",
        lambda web_port, mcp_port: {
            "firered_web": {"ready": True, "status": "running"},
            "local_mcp": {"ready": True, "status": "running"},
            "localai_gateway": {"ready": True, "status": "running"},
            "localai_tts_ready": {"ready": True, "status": "ready"},
            "localai_music_ready": {"ready": True, "status": "ready"},
            "comfyui_bridge": {"ready": True, "status": "running"},
            "codex_auth_state": {"signed_in": True, "status": "signed_in"},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        dev_runtime,
        "_smoke_flow",
        lambda args: {"session_id": "sess-123", "steps": {"render_video": {"preview": "/tmp/demo.mp4"}}},
    )
    monkeypatch.setattr(dev_runtime.asyncio, "run", lambda coro: coro)

    args = argparse.Namespace(
        web_port=7861,
        mcp_port=8001,
        model="gpt-5.4",
        reasoning="medium",
    )

    rc = dev_runtime.cmd_verify(args)
    output = capsys.readouterr().out

    assert rc == 0
    assert "\"ok\": true" in output
    assert "\"session_id\": \"sess-123\"" in output
