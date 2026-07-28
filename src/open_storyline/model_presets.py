from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple


ATLAS_CLOUD_MODEL_KEY = "Atlas Cloud"

_MODEL_PRESETS = {
    "llm": {
        ATLAS_CLOUD_MODEL_KEY: {
            "model": "deepseek-ai/deepseek-v4-pro",
            "base_url": "https://api.atlascloud.ai/v1",
            "api_key_env": ("ATLASCLOUD_API_KEY", "ATLAS_CLOUD_API_KEY"),
        },
    },
}


def list_model_presets(kind: str) -> list[str]:
    return list(_MODEL_PRESETS.get(kind.strip().lower(), {}))


def resolve_model_preset(
    kind: str,
    key: str,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    kind = kind.strip().lower()
    preset = _MODEL_PRESETS.get(kind, {}).get(key)
    if preset is None:
        return None, f"unknown {kind} model preset: {key}"

    env = os.environ if environ is None else environ
    api_key_env = preset["api_key_env"]
    api_key = next((env.get(name, "").strip() for name in api_key_env if env.get(name, "").strip()), "")
    if not api_key:
        return None, f"{key} requires one of: {', '.join(api_key_env)}"

    return {
        "model": preset["model"],
        "base_url": preset["base_url"],
        "api_key": api_key,
    }, None
