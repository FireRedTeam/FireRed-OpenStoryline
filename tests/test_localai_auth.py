import json
from types import SimpleNamespace

from open_storyline.utils import localai_auth


def test_resolve_localai_shared_token_prefers_environment(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_SHARED_TOKEN", "env-auth")
    monkeypatch.setattr(localai_auth, "_resolve_token_from_docker", lambda: "docker-auth")

    assert localai_auth.resolve_localai_shared_token() == "env-auth"


def test_resolve_localai_shared_token_uses_docker_when_env_missing(monkeypatch) -> None:
    for env_name in (
        "LOCAL_AI_PLATFORM_SHARED_TOKEN",
        "LOCAL_AI_PLATFORM_BEARER_TOKEN",
        "PLATFORM_SHARED_TOKEN",
        "LOCALAI_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)

    localai_auth._resolve_token_from_docker.cache_clear()

    def _fake_run(cmd, check, capture_output, text, timeout):
        assert check is True
        if cmd[:3] == ["docker", "ps", "--format"]:
            return SimpleNamespace(stdout="control-plane-gateway-1\nworker-audio-worker-tts-1\n")
        if cmd[:3] == ["docker", "inspect", "--format"]:
            payload = json.dumps(
                [
                    "PATH=/usr/local/bin",
                    "PLATFORM_SHARED_TOKEN=docker-auth",
                ]
            )
            return SimpleNamespace(stdout=payload)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(localai_auth.subprocess, "run", _fake_run)

    assert localai_auth.resolve_localai_shared_token() == "docker-auth"
    localai_auth._resolve_token_from_docker.cache_clear()


def test_inject_localai_runtime_env_sets_common_aliases(monkeypatch) -> None:
    monkeypatch.setattr(localai_auth, "resolve_localai_shared_token", lambda preferred=None: "runtime-auth")

    env = localai_auth.inject_localai_runtime_env({"PYTHONPATH": "src"})

    assert env["LOCAL_AI_PLATFORM_SHARED_TOKEN"] == "runtime-auth"
    assert env["LOCAL_AI_PLATFORM_BEARER_TOKEN"] == "runtime-auth"
    assert env["PLATFORM_SHARED_TOKEN"] == "runtime-auth"
    assert env["PYTHONPATH"] == "src"
