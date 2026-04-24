import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from agent_fastapi import ChatSession
from open_storyline.config import load_settings
from open_storyline.mcp.hooks.node_interceptors import ToolInterceptor
from open_storyline.utils.media_handler import scan_media_dir


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn7n6sAAAAASUVORK5CYII="
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _test_settings(tmp_path: Path):
    cfg = load_settings(_repo_root() / "config.toml").model_copy(deep=True)
    cfg.project.media_dir = tmp_path / "media"
    cfg.project.outputs_dir = tmp_path / "outputs"
    cfg.project.bgm_dir = tmp_path / "bgm"
    return cfg


def test_generated_images_are_imported_into_session_media_library(tmp_path: Path) -> None:
    cfg = _test_settings(tmp_path)
    sess = ChatSession("session-generated-media", cfg)

    generated_path = tmp_path / "codex-generated" / "summer-beach.png"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_bytes(_PNG_1X1)

    summary = {
        "generated_files": [
            {"path": str(generated_path), "kind": "image"},
        ]
    }

    async def run():
        return await sess.import_generated_media_from_summary(summary)

    imported = asyncio.run(run())

    assert len(imported) == 1
    assert scan_media_dir(sess.media_dir)["image number in user's media library"] == 1
    assert generated_path.exists() is True
    assert len(sess.pending_media_ids) == 1
    assert len(sess.public_pending_media()) == 1

    imported_meta = imported[0]
    assert Path(imported_meta.path).exists() is True
    assert Path(imported_meta.path).parent == Path(sess.media_dir)
    assert Path(imported_meta.path).name == "media_0001.png"


def test_load_media_interceptor_reads_imported_generated_images(tmp_path: Path) -> None:
    cfg = _test_settings(tmp_path)
    sess = ChatSession("session-generated-media-load-media", cfg)

    generated_path = tmp_path / "codex-generated" / "travel-cover.png"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_bytes(_PNG_1X1)

    summary = {
        "generated_files": [
            {"path": str(generated_path), "kind": "image"},
        ]
    }

    class FakeRequest:
        def __init__(self, *, name, args, runtime):
            self.name = name
            self.args = args
            self.runtime = runtime

        def override(self, *, args):
            return FakeRequest(name=self.name, args=args, runtime=self.runtime)

    class FakeStore:
        @staticmethod
        def generate_artifact_id(node_id: str) -> str:
            return f"{node_id}-artifact"

        @staticmethod
        def get_latest_meta(node_id: str, session_id: str):
            return None

    async def run():
        await sess.import_generated_media_from_summary(summary)

        node_manager = SimpleNamespace(
            id_to_tool={},
            id_to_default_require_prior_kind={},
            id_to_require_prior_kind={},
            kind_to_node_ids={},
        )
        context = SimpleNamespace(
            session_id=sess.session_id,
            media_dir=sess.media_dir,
            cfg=cfg,
            node_manager=node_manager,
            lang="zh",
        )
        runtime = SimpleNamespace(context=context, store=FakeStore())
        request = FakeRequest(name="load_media", args={}, runtime=runtime)
        seen = {}

        async def handler(modified_request):
            seen["args"] = modified_request.args
            return modified_request.args

        await ToolInterceptor.inject_media_content_before(request, handler)
        return seen["args"]

    args = asyncio.run(run())

    assert "inputs" in args
    assert len(args["inputs"]) == 1
    assert str(args["inputs"][0]["path"]).endswith("media_0001.png")


def test_apply_tool_event_normalizes_python_literal_render_summary() -> None:
    cfg = _test_settings(Path("/tmp"))
    sess = ChatSession("session-render-summary", cfg)

    raw_summary = (
        "{'summary': {'node_summary': {'ERROR': '', 'WARNING': '', "
        "'INFO_USER': 'Video generated successfully.', "
        "'preview_urls': ['/tmp/render/output.mp4'], "
        "'artifact_id': 'render_video_1'}, 'tool_excute_result': {}}, 'isError': False}"
    )

    rec = sess.apply_tool_event(
        {
            "type": "tool_end",
            "tool_call_id": "call_render_video",
            "server": "storyline",
            "name": "render_video",
            "summary": raw_summary,
            "is_error": False,
        }
    )

    assert isinstance(rec, dict)
    assert isinstance(rec["summary"], dict)
    assert rec["summary"]["preview_urls"] == ["/tmp/render/output.mp4"]
    assert rec["summary"]["INFO_USER"] == "Video generated successfully."
    assert rec["summary"]["artifact_id"] == "render_video_1"


def test_load_from_state_normalizes_legacy_tool_summary(tmp_path: Path) -> None:
    cfg = _test_settings(tmp_path)
    session_id = "legacy-render-preview"
    state_path = Path(ChatSession.state_file_path_for(session_id, cfg))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    raw_summary = (
        "{'summary': {'node_summary': {'ERROR': '', 'WARNING': '', "
        "'INFO_USER': 'Video generated successfully.', "
        "'preview_urls': ['/tmp/render/output.mp4'], "
        "'artifact_id': 'render_video_1'}, 'tool_excute_result': {}}, 'isError': False}"
    )

    state = {
        "version": 1,
        "session_id": session_id,
        "lang": "zh",
        "history": [
            {
                "id": "tool_call_render_video",
                "role": "tool",
                "tool_call_id": "call_render_video",
                "server": "storyline",
                "name": "render_video",
                "state": "complete",
                "progress": 1.0,
                "message": "",
                "summary": raw_summary,
                "ts": 1.0,
            }
        ],
        "chat_model_key": "demo-llm",
        "vlm_model_key": "demo-vlm",
        "load_media": {},
        "pending_media_ids": [],
        "lc_messages_serialized": [],
        "pexels_key_mode": "default",
        "sent_media_total": 0,
        "custom_llm_config": None,
        "custom_vlm_config": None,
        "tts_config": {},
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    restored = ChatSession.load_from_state(session_id, cfg)

    assert restored is not None
    assert isinstance(restored.history[0]["summary"], dict)
    assert restored.history[0]["summary"]["preview_urls"] == ["/tmp/render/output.mp4"]
    assert restored.history[0]["summary"]["artifact_id"] == "render_video_1"
