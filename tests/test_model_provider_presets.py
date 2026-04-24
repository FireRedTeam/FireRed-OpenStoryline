from types import SimpleNamespace

from open_storyline.model_provider_presets import (
    build_model_provider_ui_schema,
    resolve_model_provider_override,
)


def _provider_map(kind: str) -> dict[str, dict]:
    schema = build_model_provider_ui_schema()
    providers = schema[kind]["providers"]
    return {item["provider"]: item for item in providers}


def test_build_model_provider_ui_schema_contains_four_presets_for_llm_and_vlm() -> None:
    schema = build_model_provider_ui_schema()

    assert set(schema.keys()) == {"llm", "vlm"}
    assert {item["provider"] for item in schema["llm"]["providers"]} == {
        "codex",
        "openai",
        "gemini",
        "nvidia",
        "ollama",
    }
    assert {item["provider"] for item in schema["vlm"]["providers"]} == {
        "codex",
        "openai",
        "gemini",
        "nvidia",
        "ollama",
    }


def test_each_provider_exposes_required_frontend_fields() -> None:
    for kind in ("llm", "vlm"):
        providers = _provider_map(kind)
        for provider_name in ("codex", "openai", "gemini", "nvidia", "ollama"):
            provider = providers[provider_name]
            assert provider["provider"] == provider_name
            assert isinstance(provider["label"], str) and provider["label"]
            assert isinstance(provider["docs_url"], str) and provider["docs_url"].startswith("https://")
            assert isinstance(provider["model"], str) and provider["model"]
            if provider_name == "codex":
                assert provider["auth_kind"] == "codex"
                assert provider["base_url"] == ""
                assert provider["api_key_env"] == ""
            else:
                assert isinstance(provider["base_url"], str) and provider["base_url"].startswith(("http://", "https://"))
                assert isinstance(provider["api_key_env"], str) and provider["api_key_env"]


def test_ollama_preset_exposes_local_defaults() -> None:
    llm_ollama = _provider_map("llm")["ollama"]
    vlm_ollama = _provider_map("vlm")["ollama"]

    assert llm_ollama["base_url"] == "http://127.0.0.1:11434/v1"
    assert vlm_ollama["base_url"] == "http://127.0.0.1:11434/v1"
    assert llm_ollama["default_api_key"] == "ollama"
    assert vlm_ollama["default_api_key"] == "ollama"


def test_openai_preset_uses_official_api_defaults() -> None:
    llm_openai = _provider_map("llm")["openai"]
    vlm_openai = _provider_map("vlm")["openai"]

    assert llm_openai["base_url"] == "https://api.openai.com/v1"
    assert vlm_openai["base_url"] == "https://api.openai.com/v1"
    assert llm_openai["model"] == "gpt-5.4"
    assert vlm_openai["model"] == "gpt-5.4"
    assert llm_openai["api_key_env"] == "OPENAI_API_KEY"
    assert vlm_openai["api_key_env"] == "OPENAI_API_KEY"


def test_nvidia_presets_use_tool_capable_llm_and_current_vlm() -> None:
    llm_nvidia = _provider_map("llm")["nvidia"]
    vlm_nvidia = _provider_map("vlm")["nvidia"]

    assert llm_nvidia["model"] == "meta/llama-3.1-70b-instruct"
    assert vlm_nvidia["model"] == "microsoft/phi-4-multimodal-instruct"


def test_gemini_presets_default_to_gemma_4_26b_a4b_it() -> None:
    llm_gemini = _provider_map("llm")["gemini"]
    vlm_gemini = _provider_map("vlm")["gemini"]

    assert llm_gemini["model"] == "gemma-4-26b-a4b-it"
    assert vlm_gemini["model"] == "gemma-4-26b-a4b-it"


def test_codex_presets_are_available_for_llm_and_vlm() -> None:
    llm_codex = _provider_map("llm")["codex"]
    vlm_codex = _provider_map("vlm")["codex"]

    assert llm_codex["model"] == "gpt-5.4"
    assert vlm_codex["model"] == "gpt-5.4"
    assert llm_codex["auth_kind"] == "codex"
    assert vlm_codex["auth_kind"] == "codex"


def test_resolve_model_provider_override_reads_key_from_config() -> None:
    cfg = SimpleNamespace(
        llm=SimpleNamespace(timeout=30.0, temperature=0.1, max_retries=2),
        vlm=SimpleNamespace(timeout=20.0, temperature=0.1, max_retries=2),
        model_providers={
            "openai": SimpleNamespace(api_key="sk-test", base_url="https://api.openai.com/v1"),
        },
    )

    override, err = resolve_model_provider_override(cfg, "llm", "openai", "gpt-4.1-mini")

    assert err is None
    assert override == {
        "model": "gpt-4.1-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "timeout": 30.0,
        "temperature": 0.1,
        "max_retries": 2,
    }


def test_resolve_model_provider_override_falls_back_to_provider_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    cfg = SimpleNamespace(
        llm=SimpleNamespace(timeout=30.0, temperature=0.1, max_retries=2),
        vlm=SimpleNamespace(timeout=20.0, temperature=0.1, max_retries=2),
        model_providers={},
    )

    override, err = resolve_model_provider_override(cfg, "llm", "openai", "gpt-5.4")

    assert err is None
    assert override["api_key"] == "sk-from-env"
    assert override["base_url"] == "https://api.openai.com/v1"


def test_resolve_model_provider_override_uses_ollama_default_key() -> None:
    cfg = SimpleNamespace(
        llm=SimpleNamespace(timeout=30.0, temperature=0.1, max_retries=2),
        vlm=SimpleNamespace(timeout=20.0, temperature=0.1, max_retries=2),
        model_providers={
            "ollama": SimpleNamespace(api_key="", base_url="http://127.0.0.1:11434/v1"),
        },
    )

    override, err = resolve_model_provider_override(cfg, "vlm", "ollama", "gemma4")

    assert err is None
    assert override["api_key"] == "ollama"
    assert override["base_url"] == "http://127.0.0.1:11434/v1"
    assert override["model"] == "gemma4"
