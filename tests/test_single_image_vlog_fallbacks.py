import asyncio
from pathlib import Path
from types import SimpleNamespace

from open_storyline.config import load_settings
from open_storyline.nodes.core_nodes.generate_script import GenerateScriptNode
from open_storyline.nodes.core_nodes.group_clips import GroupClipsNode
from open_storyline.nodes.core_nodes.understand_clips import UnderstandClipsNode
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_summary import NodeSummary


class _FailingLLM:
    async def complete(self, **kwargs):
        raise AssertionError("fallback path should not invoke llm.complete()")


class _JSONLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '{"caption":"Sunny beach with clear water","aes_score":0.88}'
        return "Overall summary for the analyzed clips."


class _NoneLLM:
    async def complete(self, **kwargs):
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _test_settings(tmp_path: Path):
    cfg = load_settings(_repo_root() / "config.toml").model_copy(deep=True)
    cfg.project.media_dir = tmp_path / "media"
    cfg.project.outputs_dir = tmp_path / "outputs"
    cfg.project.bgm_dir = tmp_path / "bgm"
    return cfg


def _node_state() -> NodeState:
    return NodeState(
        session_id="test-session",
        artifact_id="test-artifact",
        lang="zh",
        node_summary=NodeSummary(auto_console=False),
        llm=_FailingLLM(),
        mcp_ctx=SimpleNamespace(),
    )


def test_understand_clips_single_image_uses_fallback(tmp_path: Path) -> None:
    node = UnderstandClipsNode(_test_settings(tmp_path))

    inputs = {
        "media": {
            "media_0001": {
                "media_id": "media_0001",
                "path": str(tmp_path / "media" / "media_0001.png"),
            }
        },
        "split_shots": {
            "clips": [
                {
                    "clip_id": "clip_0001",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0001"},
                }
            ]
        },
    }

    result = asyncio.run(node.process(_node_state(), inputs))

    assert result["overall"] == "单张静态图片素材，适合制作轻快的旅行短视频。"
    assert result["clip_captions"][0]["clip_id"] == "clip_0001"
    assert "静态" in result["overall"]


def test_understand_clips_default_process_single_image_uses_fallback(tmp_path: Path) -> None:
    node = UnderstandClipsNode(_test_settings(tmp_path))

    inputs = {
        "media": {
            "media_0001": {
                "media_id": "media_0001",
                "path": str(tmp_path / "media" / "media_0001.png"),
            }
        },
        "split_shots": {
            "clips": [
                {
                    "clip_id": "clip_0001",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0001"},
                }
            ]
        },
    }

    result = asyncio.run(node.default_process(_node_state(), inputs))

    assert result["overall"] == "单张静态图片素材，适合制作轻快的旅行短视频。"
    assert result["clip_captions"][0]["caption"] == "A bright still image suitable for a short travel-style video."


def test_understand_clips_default_process_non_single_image_uses_real_understanding(tmp_path: Path) -> None:
    node = UnderstandClipsNode(_test_settings(tmp_path))
    llm = _JSONLLM()
    node_state = NodeState(
        session_id="test-session",
        artifact_id="test-artifact",
        lang="zh",
        node_summary=NodeSummary(auto_console=False),
        llm=llm,
        mcp_ctx=SimpleNamespace(),
    )
    inputs = {
        "media": {
            "media_0001": {
                "media_id": "media_0001",
                "path": str(tmp_path / "media" / "media_0001.png"),
            },
            "media_0002": {
                "media_id": "media_0002",
                "path": str(tmp_path / "media" / "media_0002.png"),
            },
        },
        "split_shots": {
            "clips": [
                {
                    "clip_id": "clip_0001",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0001"},
                },
                {
                    "clip_id": "clip_0002",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0002"},
                },
            ]
        },
    }

    result = asyncio.run(node.default_process(node_state, inputs))

    assert result["clip_captions"][0]["caption"] == "Sunny beach with clear water"
    assert result["overall"] == "Overall summary for the analyzed clips."


def test_understand_clips_vlm_failure_returns_error_caption_instead_of_crashing(tmp_path: Path) -> None:
    node = UnderstandClipsNode(_test_settings(tmp_path))
    node_state = NodeState(
        session_id="test-session",
        artifact_id="test-artifact",
        lang="zh",
        node_summary=NodeSummary(auto_console=False),
        llm=_NoneLLM(),
        mcp_ctx=SimpleNamespace(),
    )
    inputs = {
        "media": {
            "media_0001": {
                "media_id": "media_0001",
                "path": str(tmp_path / "media" / "media_0001.png"),
            },
            "media_0002": {
                "media_id": "media_0002",
                "path": str(tmp_path / "media" / "media_0002.png"),
            },
        },
        "split_shots": {
            "clips": [
                {
                    "clip_id": "clip_0001",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0001"},
                },
                {
                    "clip_id": "clip_0002",
                    "kind": "image",
                    "source_ref": {"media_id": "media_0002"},
                },
            ]
        },
    }

    result = asyncio.run(node.process(node_state, inputs))

    assert result["clip_captions"][0]["caption"] == "Error: VLM request failed"
    assert result["clip_captions"][0]["aes_score"] == -1.0


def test_group_clips_single_image_uses_fallback(tmp_path: Path) -> None:
    node = GroupClipsNode(_test_settings(tmp_path))

    inputs = {
        "filter_clips": {
            "clip_captions": [
                {"clip_id": "clip_0001", "caption": "A bright still image"}
            ],
            "selected": ["clip_0001"],
        },
        "user_request": "做一个4秒旅行vlog",
    }

    result = asyncio.run(node.process(_node_state(), inputs))

    assert len(result["groups"]) == 1
    assert result["groups"][0]["group_id"] == "group_0001"
    assert result["groups"][0]["clip_ids"] == ["clip_0001"]


def test_generate_script_single_image_uses_fallback(tmp_path: Path) -> None:
    node = GenerateScriptNode(_test_settings(tmp_path))

    inputs = {
        "split_shots": {
            "clips": [
                {
                    "clip_id": "clip_0001",
                    "kind": "image",
                    "source_ref": {"duration": 0},
                }
            ]
        },
        "understand_clips": {
            "clip_captions": [
                {"clip_id": "clip_0001", "caption": "A bright still image suitable for a short travel-style video."}
            ],
            "overall": "单张静态图片素材，适合制作轻快的旅行短视频。",
        },
        "group_clips": {
            "groups": [
                {
                    "group_id": "group_0001",
                    "summary": "single image",
                    "clip_ids": ["clip_0001"],
                }
            ]
        },
        "user_request": "请做一个海滩旅行vlog",
        "custom_script": {},
    }

    result = asyncio.run(node.process(_node_state(), inputs))

    assert result["title"] == "夏日海岸"
    assert len(result["group_scripts"]) == 1
    assert result["group_scripts"][0]["group_id"] == "group_0001"
    assert "假期的快乐" in result["group_scripts"][0]["raw_text"]
    assert result["group_scripts"][0]["subtitle_units"]
