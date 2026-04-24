from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from typing import Iterable


_TOKEN_ENV_VARS = (
    "LOCAL_AI_PLATFORM_SHARED_TOKEN",
    "LOCAL_AI_PLATFORM_BEARER_TOKEN",
    "PLATFORM_SHARED_TOKEN",
    "LOCALAI_API_KEY",
)

_DOCKER_CONTAINER_HINTS = (
    "control-plane-gateway",
    "worker-audio-worker-tts",
    "worker-visual-worker-music",
    "control-plane-mcp-server",
)


def _normalize_token(value: object) -> str:
    return str(value or "").strip()


def _extract_env_token(env_items: Iterable[object]) -> str:
    for item in env_items:
        text = str(item or "")
        if "=" not in text:
            continue
        key, raw_value = text.split("=", 1)
        if key.strip() in _TOKEN_ENV_VARS:
            token = _normalize_token(raw_value)
            if token:
                return token
    return ""


def _docker_container_names() -> list[str]:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    names = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        if any(hint in name for hint in _DOCKER_CONTAINER_HINTS):
            names.append(name)
    return names


@lru_cache(maxsize=1)
def _resolve_token_from_docker() -> str:
    candidates = list(dict.fromkeys([*_DOCKER_CONTAINER_HINTS, *_docker_container_names()]))
    for container_name in candidates:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{json .Config.Env}}", container_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue

        raw = _normalize_token(result.stdout)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        token = _extract_env_token(payload)
        if token:
            return token
    return ""


def resolve_localai_shared_token(preferred: object | None = None) -> str:
    token = _normalize_token(preferred)
    if token:
        return token

    for env_name in _TOKEN_ENV_VARS:
        token = _normalize_token(os.getenv(env_name))
        if token:
            return token

    return _resolve_token_from_docker()


def inject_localai_runtime_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    token = resolve_localai_shared_token()
    if not token:
        return env

    for env_name in ("LOCAL_AI_PLATFORM_SHARED_TOKEN", "LOCAL_AI_PLATFORM_BEARER_TOKEN", "PLATFORM_SHARED_TOKEN"):
        env.setdefault(env_name, token)
    return env
