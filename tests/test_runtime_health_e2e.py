from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest

playwright = pytest.importorskip("playwright.sync_api")

BASE_URL = os.getenv("FIRERED_E2E_BASE_URL", "http://127.0.0.1:7861")
CHROME_PATH = os.getenv("GOOGLE_CHROME_BIN", "/usr/bin/google-chrome")

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sZQx1QAAAAASUVORK5CYII="
)


def _require_server() -> None:
    try:
        response = httpx.get(BASE_URL, timeout=5.0)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"FireRed web is not reachable at {BASE_URL}: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"FireRed web returned {response.status_code} at {BASE_URL}")


def _create_real_session_id() -> str:
    response = httpx.post(f"{BASE_URL}/api/sessions", timeout=10.0)
    response.raise_for_status()
    return str(response.json()["session_id"])


def _preview_url(session_id: str, path: Path) -> str:
    return f"/api/sessions/{session_id}/preview?path={quote(str(path), safe='')}"


def _route_json(page, url: str, payload: dict) -> None:
    page.route(url, lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(payload, ensure_ascii=False)))


def _mock_runtime_bootstrap(page, *, session_id: str, snapshot: dict, runtime_health: dict) -> None:
    page.add_init_script(
        f"""
        (() => {{
          localStorage.setItem("openstoryline_session_id", {session_id!r});
          class MockWebSocket {{
            constructor(url) {{
              this.url = url;
              this.readyState = 1;
              setTimeout(() => {{ if (this.onopen) this.onopen(); }}, 0);
            }}
            send(_data) {{}}
            close() {{
              this.readyState = 3;
              setTimeout(() => {{
                if (this.onclose) this.onclose({{ code: 1000, reason: "mock", wasClean: true }});
              }}, 0);
            }}
          }}
          window.WebSocket = MockWebSocket;
        }})();
        """
    )

    _route_json(
        page,
        f"**/api/meta/model_providers",
        {
            "llm": {"providers": [{"provider": "codex", "label": "Codex", "model": "gpt-5.4", "models": ["gpt-5.4"]}]},
            "vlm": {"providers": [{"provider": "codex", "label": "Codex", "model": "gpt-5.4", "models": ["gpt-5.4"]}]},
        },
    )
    _route_json(page, f"**/api/codex/account", {"signed_in": True, "account": {"email": "tester@example.com", "plan_type": "Plus"}})
    _route_json(page, f"**/api/codex/models", {"models": [{"model": "gpt-5.4", "display_name": "GPT-5.4", "reasoning_effort_options": ["medium", "high"], "default_reasoning_effort": "medium"}]})
    _route_json(page, f"**/api/codex/rate_limits", {"primary": {"used_percent": 12}, "secondary": {"used_percent": 2}})
    _route_json(page, f"**/api/meta/bgm", {"default_provider": "localai", "providers": [{"provider": "localai", "label": "LocalAI", "fields": []}]})
    _route_json(page, f"**/api/meta/tts", {"default_provider": "localai", "providers": [{"provider": "localai", "label": "LocalAI", "fields": []}]})
    _route_json(page, f"**/api/meta/ai_transition", {"default_provider": "", "providers": []})
    _route_json(page, f"**/api/meta/runtime_health", runtime_health)
    _route_json(page, f"**/api/sessions/{session_id}", snapshot)


def test_runtime_status_panel_and_media_previews(tmp_path: Path) -> None:
    _require_server()
    session_id = _create_real_session_id()
    image_a = tmp_path / "beach_a.png"
    image_b = tmp_path / "beach_b.png"
    video = tmp_path / "rendered.mp4"
    image_a.write_bytes(_PNG_1X1)
    image_b.write_bytes(_PNG_1X1)
    video.write_bytes(b"00")

    runtime_health = {
        "firered_web": {"status": "running", "ready": True},
        "local_mcp": {"status": "running", "ready": True},
        "localai_gateway": {"status": "running", "ready": True},
        "localai_tts_ready": {"status": "ready", "ready": True},
        "localai_music_ready": {"status": "ready", "ready": True},
        "comfyui_bridge": {"status": "running", "ready": True},
        "codex_auth_state": {"signed_in": True, "status": "signed_in", "detail": ""},
        "warnings": [],
    }
    snapshot = {
        "session_id": session_id,
        "developer_mode": False,
        "pending_media": [
            {
                "id": "media_1",
                "name": "beach-a",
                "kind": "image",
                "thumb_url": _preview_url(session_id, image_a),
                "file_url": _preview_url(session_id, image_a),
            },
            {
                "id": "media_2",
                "name": "beach-b",
                "kind": "image",
                "thumb_url": _preview_url(session_id, image_b),
                "file_url": _preview_url(session_id, image_b),
            },
        ],
        "history": [
            {"id": "u1", "role": "user", "content": "请生成素材并剪片。", "attachments": [], "ts": 1.0},
            {"id": "a1", "role": "assistant", "content": "素材已生成并进入素材库。", "ts": 2.0},
            {
                "id": "tool_img",
                "role": "tool",
                "tool_call_id": "call_img",
                "name": "codex.image_gen",
                "server": "storyline",
                "state": "complete",
                "progress": 1.0,
                "message": "done",
                "summary": {
                    "preview_urls": [str(image_a), str(image_b)],
                    "INFO_USER": "Imported 2 generated images into the session media library.",
                },
                "ts": 3.0,
            },
            {
                "id": "tool_vid",
                "role": "tool",
                "tool_call_id": "call_vid",
                "name": "render_video",
                "server": "storyline",
                "state": "complete",
                "progress": 1.0,
                "message": "done",
                "summary": {
                    "preview_urls": [str(video)],
                    "INFO_USER": "Video generated successfully.",
                },
                "ts": 4.0,
            },
        ],
        "turn_running": False,
        "restore_degraded": False,
        "restore_degraded_reason": "",
        "limits": {"max_media_per_session": 30, "max_pending_media_per_session": 30, "upload_chunk_bytes": 8388608},
        "stats": {"media_count": 2, "pending_count": 2, "inflight_uploads": 0, "sent_media_total": 0},
        "chat_model_key": "codex",
        "chat_models": ["codex"],
        "llm_model_key": "codex",
        "llm_models": ["codex"],
        "vlm_model_key": "codex",
        "vlm_models": ["codex"],
        "lang": "zh",
    }

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROME_PATH)
        page = browser.new_page()
        _mock_runtime_bootstrap(page, session_id=session_id, snapshot=snapshot, runtime_health=runtime_health)
        page.goto(BASE_URL, wait_until="networkidle")

        expect = playwright.expect
        expect(page.locator("#runtimeProfileBadge")).to_have_text("Codex + LocalAI")
        expect(page.locator("#runtimeHealthList")).to_contain_text("FireRed Web")
        expect(page.locator("#runtimeHealthList")).to_contain_text("LocalAI TTS")
        expect(page.locator("#runtimeHealthList")).to_contain_text("Codex Auth")
        expect(page.locator("#pendingRow .media-item")).to_have_count(2)
        expect(page.locator("#chat")).to_contain_text("图片（点击预览）")
        expect(page.locator("#chat")).to_contain_text("成片预览")
        expect(page.locator("#chat video")).to_have_count(1)
        browser.close()


def test_runtime_panel_surfaces_warnings() -> None:
    _require_server()
    session_id = _create_real_session_id()
    runtime_health = {
        "firered_web": {"status": "running", "ready": True},
        "local_mcp": {"status": "missing", "ready": False, "detail": "Local MCP server is not reachable."},
        "localai_gateway": {"status": "configured", "ready": False, "detail": "LocalAI gateway is configured but not reachable."},
        "localai_tts_ready": {"status": "auth_error", "ready": False, "detail": "unauthorized"},
        "localai_music_ready": {"status": "unreachable", "ready": False, "detail": "connection refused"},
        "comfyui_bridge": {"status": "unreachable", "ready": False, "detail": "ComfyUI bridge is not reachable."},
        "codex_auth_state": {"signed_in": False, "status": "signed_out", "detail": ""},
        "warnings": [
            "Local MCP unavailable: Local MCP server is not reachable.",
            "LocalAI gateway unavailable: LocalAI gateway is configured but not reachable.",
            "Codex auth is signed out.",
        ],
    }
    snapshot = {
        "session_id": session_id,
        "developer_mode": False,
        "pending_media": [],
        "history": [],
        "turn_running": False,
        "restore_degraded": False,
        "restore_degraded_reason": "",
        "limits": {"max_media_per_session": 30, "max_pending_media_per_session": 30, "upload_chunk_bytes": 8388608},
        "stats": {"media_count": 0, "pending_count": 0, "inflight_uploads": 0, "sent_media_total": 0},
        "chat_model_key": "codex",
        "chat_models": ["codex"],
        "llm_model_key": "codex",
        "llm_models": ["codex"],
        "vlm_model_key": "codex",
        "vlm_models": ["codex"],
        "lang": "zh",
    }

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=CHROME_PATH)
        page = browser.new_page()
        _mock_runtime_bootstrap(page, session_id=session_id, snapshot=snapshot, runtime_health=runtime_health)
        page.goto(BASE_URL, wait_until="networkidle")

        expect = playwright.expect
        expect(page.locator("#runtimeHealthWarnings")).to_contain_text("Local MCP unavailable")
        expect(page.locator("#runtimeHealthWarnings")).to_contain_text("LocalAI gateway unavailable")
        expect(page.locator("#runtimeHealthWarnings")).to_contain_text("Codex auth is signed out.")
        browser.close()
