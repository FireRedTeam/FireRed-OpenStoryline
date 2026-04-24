from __future__ import annotations

import socket
from types import SimpleNamespace
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import requests

from open_storyline.codex import normalize_codex_account_response
from open_storyline.utils.localai_auth import resolve_localai_shared_token


SocketChecker = Callable[[str, int, float], bool]
LocalAIProbe = Callable[..., dict[str, Any]]
CodexAccountReader = Callable[[], Awaitable[dict[str, Any]]]


def probe_tcp_listener(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout)):
            return True
    except OSError:
        return False


def _norm_url(text: Any) -> str:
    return str(text or "").strip().rstrip("/")


def _parse_host_port(base_url: Any) -> tuple[str, int | None]:
    parsed = urlparse(_norm_url(base_url))
    if not parsed.hostname:
        return ("", None)
    if parsed.port is not None:
        return (parsed.hostname, int(parsed.port))
    if parsed.scheme == "https":
        return (parsed.hostname, 443)
    if parsed.scheme == "http":
        return (parsed.hostname, 80)
    return (parsed.hostname, None)


def _localai_section(cfg: Any, section_name: str) -> dict[str, Any]:
    section = getattr(cfg, section_name, None)
    providers = getattr(section, "providers", {}) if section is not None else {}
    if isinstance(providers, dict):
        provider = providers.get("localai")
        return provider if isinstance(provider, dict) else {}
    return {}


def _resolve_localai_api_key(raw_api_key: Any) -> str:
    return resolve_localai_shared_token(raw_api_key)


def probe_localai_job_endpoint(
    *,
    base_url: str,
    api_key: str,
    job_type: str,
    timeout: float = 1.5,
) -> dict[str, Any]:
    normalized_base = _norm_url(base_url)
    url = f"{normalized_base}/api/v1/jobs/{str(job_type or '').strip()}"
    headers: dict[str, str] = {}
    resolved_api_key = _resolve_localai_api_key(api_key)
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    try:
        response = requests.post(url, headers=headers, json={}, timeout=float(timeout))
    except requests.RequestException as exc:
        return {
            "job_type": str(job_type or "").strip(),
            "base_url": normalized_base,
            "reachable": False,
            "ready": False,
            "status": "unreachable",
            "detail": str(exc),
            "http_status": None,
        }

    status_code = int(response.status_code)
    detail = str(getattr(response, "text", "") or "").strip()
    if 200 <= status_code < 300 or status_code == 422:
        status = "ready"
        ready = True
    elif status_code in {401, 403}:
        status = "auth_error"
        ready = False
    elif status_code == 404:
        status = "missing_endpoint"
        ready = False
    else:
        status = "unavailable"
        ready = False
    return {
        "job_type": str(job_type or "").strip(),
        "base_url": normalized_base,
        "reachable": True,
        "ready": ready,
        "status": status,
        "detail": detail,
        "http_status": status_code,
    }


def _warning(summary: dict[str, Any], failure_text: str) -> str | None:
    if bool(summary.get("ready")):
        return None
    detail = str(summary.get("detail") or "").strip()
    status = str(summary.get("status") or "").strip()
    if detail:
        return f"{failure_text}: {detail}"
    if status:
        return f"{failure_text}: {status}"
    return failure_text


def _normalize_codex_auth_payload(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    if "signed_in" in payload:
        normalized = dict(payload)
        normalized.setdefault("status", "signed_in" if normalized.get("signed_in") else "signed_out")
        normalized.setdefault("detail", "")
        return normalized
    normalized = normalize_codex_account_response(payload)
    normalized["status"] = "signed_in" if normalized.get("signed_in") else "signed_out"
    normalized.setdefault("detail", "")
    return normalized


async def build_runtime_health_snapshot(
    *,
    cfg: Any,
    firered_port: int = 7861,
    local_mcp_manager: Any = None,
    socket_checker: SocketChecker = probe_tcp_listener,
    localai_probe: LocalAIProbe = probe_localai_job_endpoint,
    codex_account_reader: CodexAccountReader | None = None,
) -> dict[str, Any]:
    firered_ready = bool(socket_checker("127.0.0.1", int(firered_port), 0.25))
    firered_web = {
        "host": "127.0.0.1",
        "port": int(firered_port),
        "status": "running" if firered_ready else "unreachable",
        "ready": firered_ready,
        "detail": "" if firered_ready else "FireRed web listener is not reachable.",
    }

    mcp_snapshot = (
        local_mcp_manager.snapshot()
        if local_mcp_manager is not None and hasattr(local_mcp_manager, "snapshot")
        else {"status": "missing"}
    )
    mcp_host = str(mcp_snapshot.get("host") or getattr(cfg.local_mcp_server, "connect_host", "127.0.0.1"))
    mcp_port = int(mcp_snapshot.get("port") or getattr(cfg.local_mcp_server, "port", 0) or 0)
    mcp_reachable = bool(mcp_port and socket_checker(mcp_host, mcp_port, 0.25))
    mcp_status = str(mcp_snapshot.get("status") or "").strip().lower()
    local_mcp = {
        **dict(mcp_snapshot),
        "ready": bool(mcp_status in {"running", "external"} and mcp_reachable),
    }
    if not local_mcp["ready"] and not local_mcp.get("detail"):
        local_mcp["detail"] = str(mcp_snapshot.get("last_error") or "").strip()
    if not local_mcp["ready"] and not local_mcp["detail"]:
        local_mcp["detail"] = "Local MCP server is not reachable."

    localai_tts_cfg = _localai_section(cfg, "generate_voiceover")
    localai_music_cfg = _localai_section(cfg, "select_bgm")
    localai_base_url = _norm_url(localai_tts_cfg.get("base_url") or localai_music_cfg.get("base_url"))
    localai_host, localai_port = _parse_host_port(localai_base_url)
    localai_gateway_ready = bool(localai_host and localai_port and socket_checker(localai_host, localai_port, 0.25))
    localai_gateway = {
        "base_url": localai_base_url,
        "host": localai_host,
        "port": localai_port,
        "configured": bool(localai_base_url),
        "status": "running" if localai_gateway_ready else ("configured" if localai_base_url else "missing"),
        "ready": localai_gateway_ready,
        "detail": "" if localai_gateway_ready else ("LocalAI gateway is configured but not reachable." if localai_base_url else "LocalAI gateway is not configured."),
    }

    localai_tts_ready = localai_probe(
        base_url=localai_base_url,
        api_key=str(localai_tts_cfg.get("api_key") or ""),
        job_type="tts",
    ) if localai_base_url else {
        "job_type": "tts",
        "base_url": "",
        "reachable": False,
        "ready": False,
        "status": "missing",
        "detail": "LocalAI TTS provider is not configured.",
        "http_status": None,
    }

    localai_music_ready = localai_probe(
        base_url=localai_base_url,
        api_key=str(localai_music_cfg.get("api_key") or ""),
        job_type="music",
    ) if localai_base_url else {
        "job_type": "music",
        "base_url": "",
        "reachable": False,
        "ready": False,
        "status": "missing",
        "detail": "LocalAI music provider is not configured.",
        "http_status": None,
    }

    comfyui_ready = bool(socket_checker("127.0.0.1", 8188, 0.25))
    comfyui_bridge = {
        "host": "127.0.0.1",
        "port": 8188,
        "status": "running" if comfyui_ready else "unreachable",
        "ready": comfyui_ready,
        "detail": "" if comfyui_ready else "ComfyUI bridge is not reachable.",
    }

    codex_auth_state: dict[str, Any]
    if codex_account_reader is None:
        codex_auth_state = {
            "signed_in": False,
            "status": "unknown",
            "detail": "Codex auth reader is unavailable.",
        }
    else:
        try:
            codex_auth_state = _normalize_codex_auth_payload(await codex_account_reader())
        except Exception as exc:
            codex_auth_state = {
                "signed_in": False,
                "status": "unavailable",
                "detail": str(exc),
            }

    warnings = [
        warning
        for warning in [
            _warning(local_mcp, "Local MCP unavailable"),
            _warning(localai_gateway, "LocalAI gateway unavailable"),
            _warning(localai_tts_ready, "LocalAI TTS unavailable"),
            _warning(localai_music_ready, "LocalAI music unavailable"),
            _warning(comfyui_bridge, "ComfyUI bridge unavailable"),
            None if codex_auth_state.get("signed_in") else "Codex auth is signed out.",
        ]
        if warning
    ]

    return {
        "firered_web": firered_web,
        "local_mcp": local_mcp,
        "localai_gateway": localai_gateway,
        "localai_tts_ready": localai_tts_ready,
        "localai_music_ready": localai_music_ready,
        "comfyui_bridge": comfyui_bridge,
        "codex_auth_state": codex_auth_state,
        "warnings": warnings,
    }
