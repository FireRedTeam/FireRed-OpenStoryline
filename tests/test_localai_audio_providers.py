import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent_fastapi import _build_provider_ui_schema_from_config, _parse_service_config
from open_storyline.nodes.core_nodes.generate_voiceover import GenerateVoiceoverNode
from open_storyline.nodes.core_nodes.select_bgm import SelectBGMNode
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_summary import NodeSummary


class _FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        headers: dict | None = None,
        status_code: int = 200,
        text: str = "",
        json_data=None,
    ):
        self.content = content
        self.headers = headers or {}
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"http {self.status_code}: {self.text}")

    def json(self):
        if self._json_data is None:
            raise RuntimeError("no json payload")
        return self._json_data


def _node_state(*, session_id: str = "test-session", artifact_id: str = "test-artifact") -> NodeState:
    return NodeState(
        session_id=session_id,
        artifact_id=artifact_id,
        lang="zh",
        node_summary=NodeSummary(auto_console=False),
        llm=SimpleNamespace(),
        mcp_ctx=SimpleNamespace(),
    )


def test_parse_service_config_extracts_bgm_provider_block() -> None:
    service_cfg = {
        "tts": {
            "provider": "localai",
            "localai": {"base_url": "http://127.0.0.1:8080", "model": "piper-zh"},
        },
        "bgm": {
            "provider": "localai",
            "localai": {"base_url": "http://127.0.0.1:8080", "model_id": "ace-step-turbo"},
        },
    }

    llm, vlm, tts, ai_transition, bgm, pexels, err = _parse_service_config(service_cfg)

    assert err is None
    assert llm is None
    assert vlm is None
    assert tts == {
        "provider": "localai",
        "localai": {"base_url": "http://127.0.0.1:8080", "model": "piper-zh"},
    }
    assert ai_transition == {}
    assert bgm == {
        "provider": "localai",
        "localai": {"base_url": "http://127.0.0.1:8080", "model_id": "ace-step-turbo"},
    }
    assert pexels == {}


def test_parse_service_config_accepts_flat_provider_fields() -> None:
    service_cfg = {
        "tts": {
            "provider": "localai",
            "mode": "gateway",
            "base_url": "http://127.0.0.1:18080",
            "voice_provider": "voxcpm",
            "voice_id": "default",
        },
        "bgm": {
            "provider": "localai",
            "mode": "gateway",
            "base_url": "http://127.0.0.1:18080",
            "duration_seconds": 12,
            "instrumental": True,
        },
    }

    llm, vlm, tts, ai_transition, bgm, pexels, err = _parse_service_config(service_cfg)

    assert err is None
    assert llm is None
    assert vlm is None
    assert ai_transition == {}
    assert pexels == {}
    assert tts == {
        "provider": "localai",
        "localai": {
            "mode": "gateway",
            "base_url": "http://127.0.0.1:18080",
            "voice_provider": "voxcpm",
            "voice_id": "default",
        },
    }
    assert bgm == {
        "provider": "localai",
        "localai": {
            "mode": "gateway",
            "base_url": "http://127.0.0.1:18080",
            "duration_seconds": 12,
            "instrumental": True,
        },
    }


def test_build_provider_ui_schema_uses_localai_as_default_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[generate_voiceover.providers.localai]
label = "Local-ai-platform"
mode = "gateway"
base_url = "http://127.0.0.1:18080"

[generate_voiceover.providers.minimax]
base_url = ""
api_key = ""

[select_bgm.providers.localai]
label = "Local-ai-platform"
mode = "gateway"
base_url = "http://127.0.0.1:18080"
duration_seconds = 12
        """.strip(),
        encoding="utf-8",
    )

    tts_schema = _build_provider_ui_schema_from_config(str(config_path), "generate_voiceover")
    bgm_schema = _build_provider_ui_schema_from_config(str(config_path), "select_bgm")

    assert tts_schema["default_provider"] == "localai"
    assert bgm_schema["default_provider"] == "localai"


def test_select_bgm_normalizes_list_filters_for_localai_prompt() -> None:
    normalized = SelectBGMNode._normalize_filter_map(["Happy", "Travel", "Happy"])

    assert normalized == {
        "mood": ["Happy"],
        "scene": ["Travel"],
    }
    prompt = SelectBGMNode._build_localai_prompt("", normalized)
    assert "instrumental background music" in prompt
    assert "mood: Happy" in prompt
    assert "scene: Travel" in prompt


def test_generate_voiceover_defaults_to_localai_provider_when_missing(tmp_path: Path, monkeypatch) -> None:
    node = object.__new__(GenerateVoiceoverNode)
    node.server_cfg = SimpleNamespace(
        generate_voiceover=SimpleNamespace(
            providers={
                "localai": {
                    "mode": "gateway",
                    "base_url": "http://127.0.0.1:18080",
                    "api_key": "",
                    "voice_provider": "voxcpm",
                    "voice_id": "default",
                    "format": "wav",
                },
                "minimax": {
                    "base_url": "https://example.invalid",
                    "api_key": "unused",
                },
            }
        )
    )
    node.server_cache_dir = tmp_path
    calls = []

    def _fake_localai(self, *, text, wav_path, secrets, tts_params, provider_cfg):
        calls.append({"text": text, "wav_path": str(wav_path), "secrets": secrets})
        wav_path.write_bytes(b"RIFFtest-wave")

    def _unexpected_minimax(self, **kwargs):
        raise AssertionError("minimax handler should not be used when provider is omitted")

    async def _fake_infer(self, **kwargs):
        return {}

    monkeypatch.setattr(GenerateVoiceoverNode, "_tts_localai_sync", _fake_localai)
    monkeypatch.setattr(GenerateVoiceoverNode, "_tts_minimax_sync", _unexpected_minimax)
    monkeypatch.setattr(GenerateVoiceoverNode, "_infer_tts_params_with_llm", _fake_infer)
    monkeypatch.setattr(GenerateVoiceoverNode, "_load_provider_param_schema", lambda self, provider_name: {})
    monkeypatch.setattr(GenerateVoiceoverNode, "_wav_duration_ms", staticmethod(lambda _path: 1234))

    result = asyncio.run(
        node.process(
            _node_state(session_id="sess-tts", artifact_id="art-tts"),
            {
                "generate_script": {
                    "group_scripts": [
                        {"group_id": "group_0001", "raw_text": "你好，海边的夏天。"}
                    ]
                }
            },
        )
    )

    assert len(calls) == 1
    assert result["voiceover"][0]["duration"] == 1234
    assert result["voiceover"][0]["group_id"] == "group_0001"


def test_select_bgm_defaults_to_localai_provider_when_missing(tmp_path: Path, monkeypatch) -> None:
    node = object.__new__(SelectBGMNode)
    node.server_cfg = SimpleNamespace(
        select_bgm=SimpleNamespace(
            sample_rate=22050,
            hop_length=2048,
            frame_length=2048,
            providers={
                "localai": {
                    "mode": "gateway",
                    "base_url": "http://127.0.0.1:18080",
                    "api_key": "",
                    "duration_seconds": 12,
                    "instrumental": True,
                }
            },
        )
    )
    node.server_cache_dir = tmp_path

    def _fake_localai(prompt, output_dir, provider_cfg):
        path = output_dir / "bgm_localai_test.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ID3fake-mp3")
        return {"bgm_id": "bgm_localai_test", "path": str(path)}

    async def _unexpected_recommend(*args, **kwargs):
        raise AssertionError("recommend() should not be used when provider is omitted")

    monkeypatch.setattr(node, "_generate_localai_bgm_sync", _fake_localai)
    monkeypatch.setattr(node, "recommend", _unexpected_recommend)
    monkeypatch.setattr(
        node,
        "analyze_music_metrics",
        lambda **kwargs: {
            "bgm_id": kwargs["bgm_info"]["bgm_id"],
            "path": kwargs["bgm_info"]["path"],
            "duration": 12000,
            "bpm": 120.0,
            "beats": [],
        },
    )

    result = asyncio.run(
        node.process(
            _node_state(session_id="sess-bgm", artifact_id="art-bgm"),
            {
                "user_request": "夏日海边 vlog 背景音乐",
                "filter_include": {"mood": ["Happy"]},
                "filter_exclude": {},
            },
        )
    )

    assert result["bgm"]["bgm_id"] == "bgm_localai_test"
    assert result["bgm"]["path"].endswith("bgm_localai_test.mp3")


def test_generate_voiceover_preserves_gateway_mode_when_node_mode_is_auto() -> None:
    node = object.__new__(GenerateVoiceoverNode)

    secrets = node._resolve_provider_secrets(
        "localai",
        {
            "mode": "gateway",
            "base_url": "http://127.0.0.1:18080",
            "api_key": "",
            "voice_provider": "voxcpm",
            "voice_id": "default",
        },
        {"mode": "auto"},
        type("_NodeState", (), {"node_summary": type("_Summary", (), {"info_for_llm": staticmethod(lambda *_args, **_kwargs: None)})()})(),
    )

    assert secrets["mode"] == "gateway"
    assert secrets["base_url"] == "http://127.0.0.1:18080"


def test_select_bgm_preserves_gateway_mode_when_node_mode_is_auto() -> None:
    node = object.__new__(SelectBGMNode)
    node.server_cfg = type(
        "_Cfg",
        (),
        {
            "select_bgm": type(
                "_SelectBgmCfg",
                (),
                {
                    "providers": {
                        "localai": {
                            "mode": "gateway",
                            "base_url": "http://127.0.0.1:18080",
                            "api_key": "",
                            "duration_seconds": 12,
                            "instrumental": True,
                        }
                    }
                },
            )()
        },
    )()

    provider_cfg = node._resolve_provider_cfg("localai", {"mode": "auto"})

    assert provider_cfg["mode"] == "gateway"
    assert provider_cfg["base_url"] == "http://127.0.0.1:18080"


def test_generate_voiceover_localai_default_base_url_uses_gateway_port() -> None:
    node = object.__new__(GenerateVoiceoverNode)

    assert node._default_base_url("localai") == "http://127.0.0.1:18080"


def test_select_bgm_accent_beat_normalization_handles_short_sequences(monkeypatch) -> None:
    monkeypatch.setattr("open_storyline.nodes.core_nodes.select_bgm.librosa.effects.percussive", lambda y: y)
    monkeypatch.setattr(
        "open_storyline.nodes.core_nodes.select_bgm.librosa.onset.onset_strength",
        lambda **kwargs: np.array([0.1, 0.3, 0.2, 0.8, 0.4, 0.9, 0.1], dtype=np.float64),
    )
    monkeypatch.setattr(
        "open_storyline.nodes.core_nodes.select_bgm.librosa.frames_to_time",
        lambda frames, sr, hop_length: np.asarray(frames, dtype=np.float64) / 10.0,
    )

    result = SelectBGMNode._compute_accent_beats(
        y=np.zeros(32, dtype=np.float32),
        sr=22050,
        beat_frames=np.array([0, 1, 2, 3, 4, 5], dtype=int),
        hop_length=512,
        local_norm_win=8,
    )

    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


def test_generate_voiceover_localai_sync_writes_audio_file(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return _FakeResponse(b"RIFFtest-wave", headers={"Content-Type": "audio/wav"})

    monkeypatch.setattr("open_storyline.nodes.core_nodes.generate_voiceover.requests.post", _fake_post)

    node = object.__new__(GenerateVoiceoverNode)
    wav_path = tmp_path / "voice.wav"

    node._tts_localai_sync(
        text="你好，世界",
        wav_path=wav_path,
        secrets={"base_url": "http://127.0.0.1:8080", "api_key": "local-key"},
        tts_params={"model": "piper-zh", "backend": "piper"},
        provider_cfg={},
    )

    assert wav_path.read_bytes() == b"RIFFtest-wave"
    assert seen["url"] == "http://127.0.0.1:8080/tts"
    assert seen["headers"]["Authorization"] == "Bearer local-key"
    assert seen["json"]["model"] == "piper-zh"
    assert seen["json"]["backend"] == "piper"
    assert seen["json"]["input"] == "你好，世界"


def test_select_bgm_localai_sync_generates_local_audio_file(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return _FakeResponse(b"ID3fake-mp3", headers={"Content-Type": "audio/mpeg"})

    monkeypatch.setattr("open_storyline.nodes.core_nodes.select_bgm.requests.post", _fake_post)

    node = object.__new__(SelectBGMNode)
    out_dir = tmp_path / "bgm"
    out_dir.mkdir()

    result = node._generate_localai_bgm_sync(
        prompt="欢快的旅行 vlog 背景音乐",
        output_dir=out_dir,
        provider_cfg={
            "base_url": "http://127.0.0.1:8080",
            "api_key": "local-key",
            "model_id": "ace-step-turbo",
            "duration_seconds": 12,
        },
    )

    result_path = Path(result["path"])
    assert result["bgm_id"].startswith("bgm_localai_")
    assert result_path.exists()
    assert result_path.suffix == ".mp3"
    assert result_path.read_bytes() == b"ID3fake-mp3"
    assert seen["url"] == "http://127.0.0.1:8080/v1/sound-generation"
    assert seen["headers"]["Authorization"] == "Bearer local-key"
    assert seen["json"]["model_id"] == "ace-step-turbo"
    assert seen["json"]["text"] == "欢快的旅行 vlog 背景音乐"
    assert seen["json"]["duration_seconds"] == 12
    assert seen["json"]["instrumental"] is True


def test_generate_voiceover_localai_gateway_mode_polls_job_and_downloads_asset(tmp_path: Path, monkeypatch) -> None:
    seen = {"post": [], "get": []}
    job_polls = {"count": 0}

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        seen["post"].append((url, headers, json, timeout))
        return _FakeResponse(
            b"",
            headers={"Content-Type": "application/json"},
            json_data={"data": {"job_id": "job-tts-1"}},
        )

    def _fake_get(url, *, headers=None, timeout=None):
        seen["get"].append((url, headers, timeout))
        if url.endswith("/api/v1/jobs/job-tts-1"):
            job_polls["count"] += 1
            status = "running" if job_polls["count"] == 1 else "succeeded"
            payload = {"data": {"status": status, "result_asset_ids": ["asset-tts-1"] if status == "succeeded" else []}}
            return _FakeResponse(
                b"",
                headers={"Content-Type": "application/json"},
                json_data=payload,
            )
        if url.endswith("/api/v1/assets/asset-tts-1/content"):
            return _FakeResponse(b"RIFFgateway-wave", headers={"Content-Type": "audio/wav"})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("open_storyline.nodes.core_nodes.generate_voiceover.requests.post", _fake_post)
    monkeypatch.setattr("open_storyline.nodes.core_nodes.generate_voiceover.requests.get", _fake_get)
    monkeypatch.setattr("open_storyline.nodes.core_nodes.generate_voiceover.time.sleep", lambda *_args, **_kwargs: None)

    node = object.__new__(GenerateVoiceoverNode)
    wav_path = tmp_path / "voice_gateway.wav"

    node._tts_localai_sync(
        text="你好，世界",
        wav_path=wav_path,
        secrets={
            "base_url": "http://127.0.0.1:18080",
            "api_key": "gateway-token",
            "mode": "gateway",
            "voice_provider": "voxcpm",
            "voice_id": "default",
        },
        tts_params={},
        provider_cfg={"format": "wav", "sample_rate_hz": 24000},
    )

    assert wav_path.read_bytes() == b"RIFFgateway-wave"
    create_url, create_headers, create_json, _ = seen["post"][0]
    assert create_url == "http://127.0.0.1:18080/api/v1/jobs/tts"
    assert create_headers["Authorization"] == "Bearer gateway-token"
    assert create_json["voice"]["provider"] == "voxcpm"
    assert create_json["voice"]["voice_id"] == "default"
    assert create_json["output"]["format"] == "wav"
    assert create_json["output"]["sample_rate_hz"] == 24000
    assert job_polls["count"] == 2


def test_select_bgm_localai_gateway_mode_polls_job_and_downloads_asset(tmp_path: Path, monkeypatch) -> None:
    seen = {"post": [], "get": []}

    def _fake_post(url, *, headers=None, json=None, timeout=None):
        seen["post"].append((url, headers, json, timeout))
        return _FakeResponse(
            b"",
            headers={"Content-Type": "application/json"},
            json_data={"data": {"job_id": "job-music-1"}},
        )

    def _fake_get(url, *, headers=None, timeout=None):
        seen["get"].append((url, headers, timeout))
        if url.endswith("/api/v1/jobs/job-music-1"):
            return _FakeResponse(
                b"",
                headers={"Content-Type": "application/json"},
                json_data={"data": {"status": "succeeded", "result_asset_ids": ["asset-music-1"]}},
            )
        if url.endswith("/api/v1/assets/asset-music-1/content"):
            return _FakeResponse(b"ID3gateway-mp3", headers={"Content-Type": "audio/mpeg"})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("open_storyline.nodes.core_nodes.select_bgm.requests.post", _fake_post)
    monkeypatch.setattr("open_storyline.nodes.core_nodes.select_bgm.requests.get", _fake_get)
    monkeypatch.setattr("open_storyline.nodes.core_nodes.select_bgm.time.sleep", lambda *_args, **_kwargs: None)

    node = object.__new__(SelectBGMNode)
    out_dir = tmp_path / "bgm_gateway"
    out_dir.mkdir()

    result = node._generate_localai_bgm_sync(
        prompt="欢快的旅行 vlog 背景音乐",
        output_dir=out_dir,
        provider_cfg={
            "base_url": "http://127.0.0.1:18080",
            "api_key": "gateway-token",
            "mode": "gateway",
            "duration_seconds": 12,
            "instrumental": True,
        },
    )

    result_path = Path(result["path"])
    assert result_path.exists()
    assert result_path.suffix == ".mp3"
    assert result_path.read_bytes() == b"ID3gateway-mp3"
    create_url, create_headers, create_json, _ = seen["post"][0]
    assert create_url == "http://127.0.0.1:18080/api/v1/jobs/music"
    assert create_headers["Authorization"] == "Bearer gateway-token"
    assert create_json["prompt"] == "欢快的旅行 vlog 背景音乐"
    assert create_json["style"]["duration_seconds"] == 12
    assert create_json["style"]["instrumental"] is True
