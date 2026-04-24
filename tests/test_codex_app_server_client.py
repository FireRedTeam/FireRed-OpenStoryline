import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import agent_fastapi
from open_storyline.codex.app_server_client import (
    CodexAppServerClient,
    normalize_codex_account_response,
    normalize_codex_login_response,
    normalize_codex_model_list_response,
    normalize_codex_rate_limits_response,
)


class _StubCodexAuthClient:
    async def account_read(self) -> dict:
        return {
            "account": {
                "type": "chatgpt",
                "email": "creator@example.com",
                "planType": "prolite",
            },
            "requiresOpenaiAuth": False,
        }

    async def account_login_start(self, flow: str) -> dict:
        if flow == "device_code":
            return {
                "type": "chatgptDeviceCode",
                "loginId": "login-device",
                "verificationUrl": "https://chatgpt.com/device",
                "userCode": "ABCD-EFGH",
            }
        return {
            "type": "chatgpt",
            "loginId": "login-browser",
            "authUrl": "https://chatgpt.com/auth",
        }

    async def account_rate_limits_read(self) -> dict:
        return {
            "rateLimits": {
                "primary": {
                    "usedPercent": 5,
                    "windowMinutes": 10080,
                    "resetsAt": "2026-04-24T00:00:00Z",
                },
                "secondary": {
                    "usedPercent": 18,
                    "windowMinutes": 1440,
                    "resetsAt": "2026-04-23T20:00:00Z",
                },
            },
            "rateLimitsByLimitId": {
                "codex_bengalfox": {
                    "limitName": "GPT-5.3-Codex-Spark",
                    "primary": {
                        "usedPercent": 7,
                        "windowMinutes": 1440,
                        "resetsAt": "2026-04-23T20:00:00Z",
                    },
                },
            },
        }


class _FailingCodexModelsClient:
    async def model_list(self) -> dict:
        raise RuntimeError("codex app-server unavailable")


def test_normalize_codex_account_response_maps_chatgpt_account() -> None:
    raw = {
        "account": {
            "type": "chatgpt",
            "email": "creator@example.com",
            "planType": "prolite",
        },
        "requiresOpenaiAuth": False,
    }

    normalized = normalize_codex_account_response(raw)

    assert normalized == {
        "signed_in": True,
        "requires_openai_auth": False,
        "account": {
            "type": "chatgpt",
            "email": "creator@example.com",
            "plan_type": "prolite",
        },
    }


def test_normalize_codex_login_response_supports_device_code() -> None:
    raw = {
        "type": "chatgptDeviceCode",
        "loginId": "login-device",
        "verificationUrl": "https://chatgpt.com/device",
        "userCode": "ABCD-EFGH",
    }

    normalized = normalize_codex_login_response(raw)

    assert normalized == {
        "flow": "device_code",
        "login_id": "login-device",
        "verification_url": "https://chatgpt.com/device",
        "user_code": "ABCD-EFGH",
    }


def test_normalize_codex_rate_limits_response_keeps_primary_secondary_and_named_limits() -> None:
    raw = {
        "rateLimits": {
            "primary": {
                "usedPercent": 5,
                "windowMinutes": 10080,
                "resetsAt": "2026-04-24T00:00:00Z",
            },
            "secondary": {
                "usedPercent": 18,
                "windowMinutes": 1440,
                "resetsAt": "2026-04-23T20:00:00Z",
            },
        },
        "rateLimitsByLimitId": {
            "codex_bengalfox": {
                "limitName": "GPT-5.3-Codex-Spark",
                "primary": {
                    "usedPercent": 7,
                    "windowMinutes": 1440,
                    "resetsAt": "2026-04-23T20:00:00Z",
                },
            },
        },
    }

    normalized = normalize_codex_rate_limits_response(raw)

    assert normalized["primary"]["used_percent"] == 5
    assert normalized["secondary"]["used_percent"] == 18
    assert normalized["limits"]["codex_bengalfox"]["limit_name"] == "GPT-5.3-Codex-Spark"


def test_codex_account_endpoint_returns_normalized_account_payload() -> None:
    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.codex_auth_client = _StubCodexAuthClient()
        response = client.get("/api/codex/account")

    assert response.status_code == 200
    assert response.json()["account"]["email"] == "creator@example.com"
    assert response.json()["account"]["plan_type"] == "prolite"


def test_codex_login_start_endpoint_supports_device_code_flow() -> None:
    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.codex_auth_client = _StubCodexAuthClient()
        response = client.post("/api/codex/login/start", json={"flow": "device_code"})

    assert response.status_code == 200
    assert response.json() == {
        "flow": "device_code",
        "login_id": "login-device",
        "verification_url": "https://chatgpt.com/device",
        "user_code": "ABCD-EFGH",
    }


def test_codex_rate_limits_endpoint_returns_normalized_limits() -> None:
    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.codex_auth_client = _StubCodexAuthClient()
        response = client.get("/api/codex/rate_limits")

    assert response.status_code == 200
    assert response.json()["primary"]["used_percent"] == 5
    assert response.json()["limits"]["codex_bengalfox"]["limit_name"] == "GPT-5.3-Codex-Spark"


def test_normalize_codex_model_list_response_keeps_reasoning_defaults() -> None:
    raw = {
        "data": [
            {
                "id": "gpt-5.4",
                "model": "gpt-5.4",
                "displayName": "GPT-5.4",
                "inputModalities": ["text", "image"],
                "defaultReasoningEffort": "medium",
            }
        ]
    }

    normalized = normalize_codex_model_list_response(raw)

    assert normalized["models"] == [
        {
            "id": "gpt-5.4",
            "model": "gpt-5.4",
            "display_name": "GPT-5.4",
            "description": "",
            "is_default": False,
            "input_modalities": ["text", "image"],
            "reasoning_effort_options": [],
            "default_reasoning_effort": "medium",
            "additional_speed_tiers": [],
        }
    ]


def test_codex_models_endpoint_falls_back_to_config_models_with_reasoning_defaults() -> None:
    with TestClient(agent_fastapi.app) as client:
        agent_fastapi.app.state.codex_auth_client = _FailingCodexModelsClient()
        response = client.get("/api/codex/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"]
    assert payload["models"][0]["model"] == agent_fastapi.app.state.cfg.codex.available_models[0]
    assert payload["models"][0]["default_reasoning_effort"] == agent_fastapi.app.state.cfg.codex.default_reasoning_effort
    assert payload["models"][0]["reasoning_effort_options"] == ["none", "minimal", "low", "medium", "high", "xhigh"]


def test_codex_reader_loop_handles_large_json_messages_without_readline_limit() -> None:
    large_text = "x" * 80_000
    payload = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "text": large_text,
                },
            }
        )
        + "\n"
    ).encode("utf-8")

    class _ChunkedStream:
        def __init__(self, data: bytes, *, chunk_size: int = 4096):
            self._data = data
            self._chunk_size = chunk_size
            self._offset = 0

        async def read(self, n: int = -1) -> bytes:
            await asyncio.sleep(0)
            if self._offset >= len(self._data):
                return b""
            size = self._chunk_size if n < 0 else min(n, self._chunk_size)
            chunk = self._data[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    async def _run() -> dict:
        client = CodexAppServerClient()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client._pending[1] = future
        client._proc = SimpleNamespace(stdout=_ChunkedStream(payload))

        reader = asyncio.create_task(client._reader_loop())
        result = await future
        await reader
        return result

    result = asyncio.run(_run())

    assert result["text"] == large_text
