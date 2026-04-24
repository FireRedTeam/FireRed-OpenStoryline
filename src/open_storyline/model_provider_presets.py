from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Tuple


def _norm_url(value: str) -> str:
    text = str(value or "").strip()
    return text.rstrip("/") if text else text


def _cfg_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


_MODEL_PROVIDER_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "llm": (
        {
            "provider": "codex",
            "label": "Codex (ChatGPT)",
            "docs_url": "https://github.com/openai/codex",
            "api_key_env": "",
            "model": "gpt-5.4",
            "models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark", "gpt-5.2"],
            "base_url": "",
            "default_api_key": "",
            "api_key_required": False,
            "auth_kind": "codex",
            "default_reasoning_effort": "medium",
            "reasoning_effort_options": ["none", "minimal", "low", "medium", "high", "xhigh"],
        },
        {
            "provider": "openai",
            "label": "OpenAI",
            "docs_url": "https://platform.openai.com/docs/models",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-5.4",
            "base_url": "https://api.openai.com/v1",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "gemini",
            "label": "Google Gemini",
            "docs_url": "https://ai.google.dev/gemini-api/docs/openai",
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemma-4-26b-a4b-it",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "nvidia",
            "label": "NVIDIA API Catalog",
            "docs_url": "https://docs.api.nvidia.com/nim/reference/llm-apis",
            "api_key_env": "NVIDIA_API_KEY",
            "model": "meta/llama-3.1-70b-instruct",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "ollama",
            "label": "Ollama (Gemma 4)",
            "docs_url": "https://docs.ollama.com/api/openai-compatibility",
            "api_key_env": "OLLAMA_API_KEY",
            "model": "gemma4",
            "base_url": "http://127.0.0.1:11434/v1",
            "default_api_key": "ollama",
            "api_key_required": False,
        },
    ),
    "vlm": (
        {
            "provider": "codex",
            "label": "Codex (ChatGPT)",
            "docs_url": "https://github.com/openai/codex",
            "api_key_env": "",
            "model": "gpt-5.4",
            "models": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark", "gpt-5.2"],
            "base_url": "",
            "default_api_key": "",
            "api_key_required": False,
            "auth_kind": "codex",
            "default_reasoning_effort": "medium",
            "reasoning_effort_options": ["none", "minimal", "low", "medium", "high", "xhigh"],
        },
        {
            "provider": "openai",
            "label": "OpenAI",
            "docs_url": "https://platform.openai.com/docs/guides/images-vision?api-mode=chat",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-5.4",
            "base_url": "https://api.openai.com/v1",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "gemini",
            "label": "Google Gemini",
            "docs_url": "https://ai.google.dev/gemini-api/docs/openai",
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemma-4-26b-a4b-it",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "nvidia",
            "label": "NVIDIA API Catalog",
            "docs_url": "https://docs.api.nvidia.com/nim/reference/multimodal-apis",
            "api_key_env": "NVIDIA_API_KEY",
            "model": "microsoft/phi-4-multimodal-instruct",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "default_api_key": "",
            "api_key_required": True,
        },
        {
            "provider": "ollama",
            "label": "Ollama (Gemma 4)",
            "docs_url": "https://www.ollama.com/library/gemma4",
            "api_key_env": "OLLAMA_API_KEY",
            "model": "gemma4",
            "base_url": "http://127.0.0.1:11434/v1",
            "default_api_key": "ollama",
            "api_key_required": False,
        },
    ),
}


def build_model_provider_ui_schema() -> dict[str, dict[str, list[dict[str, Any]]]]:
    schema: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for kind, presets in _MODEL_PROVIDER_PRESETS.items():
        providers: list[dict[str, Any]] = []
        for preset in presets:
            item = deepcopy(preset)
            item["base_url"] = _norm_url(item.get("base_url", ""))
            providers.append(item)
        schema[kind] = {"providers": providers}
    return schema


def resolve_model_provider_override(cfg: Any, kind: str, provider_name: str, model_name: str) -> Tuple[dict[str, Any] | None, str | None]:
    kind = str(kind or "").strip().lower()
    provider = str(provider_name or "").strip().lower()
    model = str(model_name or "").strip()

    presets = {item["provider"]: item for item in build_model_provider_ui_schema().get(kind, {}).get("providers", [])}
    preset = presets.get(provider)
    if not preset:
        return None, f"unsupported {kind} provider: {provider}"
    if not model:
        return None, f"{kind} provider model is required"
    if str(preset.get("auth_kind") or "").strip().lower() == "codex":
        return {
            "provider": provider,
            "auth_kind": "codex",
            "model": model,
        }, None

    provider_cfg = {}
    try:
        raw = getattr(cfg, "model_providers", None) or {}
        provider_cfg = raw.get(provider) or {}
    except Exception:
        provider_cfg = {}

    provider_base_url = _norm_url(_cfg_value(provider_cfg, "base_url") or "")
    provider_api_key = str(_cfg_value(provider_cfg, "api_key") or "").strip()
    env_api_key = ""
    api_key_env = str(preset.get("api_key_env") or "").strip()
    if api_key_env:
        env_api_key = str(os.getenv(api_key_env) or "").strip()

    base_url = provider_base_url or _norm_url(preset.get("base_url", ""))
    api_key = provider_api_key or env_api_key or str(preset.get("default_api_key", "") or "").strip()

    if not base_url:
        return None, f"{kind} provider {provider} base_url is missing in config"
    if not api_key and bool(preset.get("api_key_required", True)):
        return None, f"{kind} provider {provider} api_key is missing in config.toml [model_providers.{provider}]"

    cfg_block = getattr(cfg, kind, None)
    override: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }
    for key in ("timeout", "temperature", "max_retries", "top_p", "max_tokens"):
        value = getattr(cfg_block, key, None) if cfg_block is not None else None
        if value not in (None, ""):
            override[key] = value
    return override, None
