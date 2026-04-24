import asyncio
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_storyline.codex.runtime import (
    CodexRuntime,
    _default_developer_instructions,
    _looks_like_imagegen_request,
    _wait_for_new_generated_images,
    build_codex_input_items,
)


def test_build_codex_input_items_flattens_history_and_appends_local_images() -> None:
    messages = [
        SystemMessage(content="system rules"),
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="describe the image"),
    ]
    attachments = [SimpleNamespace(path="/tmp/frame.png", kind="image")]

    items = build_codex_input_items(messages, attachments)

    assert items == [
        {
            "type": "text",
            "text": (
                "System: system rules\n\n"
                "User: hello\n\n"
                "Assistant: hi there\n\n"
                "User: describe the image"
            ),
        },
        {"type": "localImage", "path": "/tmp/frame.png"},
    ]


def test_build_codex_input_items_can_enable_imagegen_skill() -> None:
    messages = [
        HumanMessage(content="please generate a simple icon"),
    ]

    items = build_codex_input_items(
        messages,
        [],
        imagegen_trigger=True,
        imagegen_skill_path="/tmp/imagegen/SKILL.md",
    )

    assert items == [
        {
            "type": "text",
            "text": "$imagegen\nUser: please generate a simple icon",
        },
        {
            "type": "skill",
            "name": "imagegen",
            "path": "/tmp/imagegen/SKILL.md",
        },
    ]


def test_looks_like_imagegen_request_supports_natural_chinese_create_image_phrasing() -> None:
    messages = [HumanMessage(content="創建一張狗狗圖片")]

    assert _looks_like_imagegen_request(messages, []) is True


def test_default_developer_instructions_hide_internal_bootstrap_details() -> None:
    instructions = _default_developer_instructions().lower()

    assert "superpowers" in instructions
    assert "agents files" in instructions
    assert "redundant confirmation" in instructions
    assert "do not mention internal harness details" in instructions


def test_wait_for_new_generated_images_polls_for_late_files(monkeypatch) -> None:
    snapshots = iter(
        [
            {},
            {},
            {"/tmp/dog.png": (1.0, 123)},
        ]
    )

    monkeypatch.setattr(
        "open_storyline.codex.runtime._snapshot_generated_images",
        lambda root: next(snapshots),
    )

    async def run():
        return await _wait_for_new_generated_images(
            "/tmp/generated_images",
            {},
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        )

    assert asyncio.run(run()) == ["/tmp/dog.png"]


def test_codex_runtime_streams_deltas_and_final_ai_message() -> None:
    class FakeClient:
        def __init__(self):
            self.thread_params = None
            self.turn_params = None
            self.closed = False
            self._notifications = asyncio.Queue()
            for item in (
                {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
                {"method": "item/agentMessage/delta", "params": {"delta": "lo"}},
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "agentMessage", "text": "Hello"}},
                },
                {"method": "turn/completed", "params": {"turnId": "turn-1"}},
            ):
                self._notifications.put_nowait(item)

        async def thread_start(self, **kwargs):
            self.thread_params = kwargs
            return {"thread": {"id": "thread-1"}}

        async def turn_start(self, **kwargs):
            self.turn_params = kwargs
            return {"turnId": "turn-1"}

        async def next_notification(self, timeout=None):
            return await self._notifications.get()

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    runtime = CodexRuntime(
        binary="codex",
        cwd="/tmp/workspace",
        model="gpt-5.4",
        client_factory=lambda: fake_client,
    )
    payload = {
        "messages": [
            SystemMessage(content="system rules"),
            HumanMessage(content="hello"),
        ]
    }
    context = SimpleNamespace(
        codex_turn_attachments=[SimpleNamespace(path="/tmp/frame.png", kind="image")],
        codex_reasoning_effort="high",
    )

    async def collect():
        events = []
        async for event in runtime.astream(payload, context=context, stream_mode=["messages", "updates"]):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert events[0][0] == "messages"
    assert events[0][1][0].content == "Hel"
    assert events[1][0] == "messages"
    assert events[1][1][0].content == "lo"
    assert events[-1][0] == "updates"
    assert events[-1][1]["codex"]["messages"][-1].content == "Hello"
    assert fake_client.thread_params["model"] == "gpt-5.4"
    assert fake_client.turn_params["effort"] == "high"
    assert fake_client.turn_params["input_items"][-1] == {"type": "localImage", "path": "/tmp/frame.png"}
    assert fake_client.closed is True


def test_codex_runtime_bridges_dynamic_tool_calls_into_langchain_messages() -> None:
    seen_runtime = {}

    class FakeTool:
        name = "storyline_fake_node"
        description = "Fake node"
        args_schema = {
            "type": "object",
            "properties": {
                "foo": {"type": "string"},
            },
            "required": ["foo"],
        }

        @staticmethod
        async def coroutine(runtime=None, **kwargs):
            seen_runtime["runtime"] = runtime
            seen_runtime["kwargs"] = kwargs
            return {"summary": {"node_summary": "done"}, "isError": False}

    class FakeClient:
        def __init__(self):
            self.thread_params = None
            self.turn_params = None
            self.responses = []
            self.closed = False
            self._notifications = asyncio.Queue()
            for item in (
                {
                    "method": "item/tool/call",
                    "id": 60,
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "callId": "call_1",
                        "tool": "storyline_fake_node",
                        "arguments": {"foo": "bar"},
                    },
                },
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "agentMessage", "text": "done"}},
                },
                {"method": "turn/completed", "params": {"turnId": "turn-1"}},
            ):
                self._notifications.put_nowait(item)

        async def thread_start(self, **kwargs):
            self.thread_params = kwargs
            return {"thread": {"id": "thread-1"}}

        async def turn_start(self, **kwargs):
            self.turn_params = kwargs
            return {"turnId": "turn-1"}

        async def next_notification(self, timeout=None):
            return await self._notifications.get()

        async def respond(self, request_id, result=None, error=None):
            self.responses.append((request_id, result, error))

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    runtime = CodexRuntime(
        binary="codex",
        cwd="/tmp/workspace",
        model="gpt-5.4",
        langchain_tools=[FakeTool()],
        store="artifact-store",
        client_factory=lambda: fake_client,
    )
    payload = {"messages": [HumanMessage(content="run the fake node")]}
    context = SimpleNamespace(node_manager=None)

    async def collect():
        events = []
        async for event in runtime.astream(payload, context=context, stream_mode=["updates"]):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0][0] == "updates"
    messages = events[0][1]["codex"]["messages"]
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls == [
        {
            "id": "call_1",
            "name": "storyline_fake_node",
            "args": {"foo": "bar"},
            "type": "tool_call",
        }
    ]
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == "call_1"
    assert messages[1].content == '{"summary": {"node_summary": "done"}, "isError": false}'
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "done"
    assert fake_client.thread_params["dynamic_tools"] == [
        {
            "name": "storyline_fake_node",
            "description": "Fake node",
            "inputSchema": FakeTool.args_schema,
            "deferLoading": False,
        }
    ]
    assert fake_client.responses == [
        (
            60,
            {
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": '{"summary": {"node_summary": "done"}, "isError": false}',
                    }
                ],
                "success": True,
            },
            None,
        )
    ]
    assert seen_runtime["runtime"].store == "artifact-store"
    assert seen_runtime["runtime"].tool_call_id == "call_1"
    assert seen_runtime["kwargs"] == {"foo": "bar"}
    assert fake_client.closed is True


def test_codex_runtime_surfaces_built_in_imagegen_outputs(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    generated_root = codex_home / "generated_images"
    outputs_root = tmp_path / "outputs"
    skill_path = tmp_path / "skills" / "imagegen" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text("# imagegen\n", encoding="utf-8")

    class FakeClient:
        def __init__(self):
            self.thread_params = None
            self.turn_params = None
            self.closed = False
            self.codex_home = str(codex_home)
            self._notifications = asyncio.Queue()
            for item in (
                {
                    "method": "item/completed",
                    "params": {"item": {"type": "agentMessage", "text": "done"}},
                },
                {"method": "turn/completed", "params": {"turnId": "turn-1"}},
            ):
                self._notifications.put_nowait(item)

        async def thread_start(self, **kwargs):
            self.thread_params = kwargs
            return {"thread": {"id": "thread-1"}}

        async def turn_start(self, **kwargs):
            self.turn_params = kwargs
            generated_root.mkdir(parents=True, exist_ok=True)
            (generated_root / "blue-square.png").write_bytes(b"pngdata")
            return {"turnId": "turn-1"}

        async def next_notification(self, timeout=None):
            return await self._notifications.get()

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    runtime = CodexRuntime(
        binary="codex",
        cwd=str(tmp_path),
        model="gpt-5.4",
        imagegen_skill_path=str(skill_path),
        client_factory=lambda: fake_client,
    )
    payload = {"messages": [HumanMessage(content="generate a tiny blue icon")]}
    context = SimpleNamespace(
        session_id="session-1",
        outputs_dir=str(outputs_root),
        codex_turn_attachments=[],
    )

    async def collect():
        events = []
        async for event in runtime.astream(payload, context=context, stream_mode=["updates"]):
            events.append(event)
        return events

    events = asyncio.run(collect())

    assert fake_client.turn_params["input_items"][0]["text"].startswith("$imagegen\n")
    assert fake_client.turn_params["input_items"][1] == {
        "type": "skill",
        "name": "imagegen",
        "path": str(skill_path),
    }
    assert len(events) == 1
    messages = events[0][1]["codex"]["messages"]
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["name"] == "image_gen"
    assert isinstance(messages[1], ToolMessage)
    tool_payload = messages[1].content
    assert "preview_urls" in tool_payload
    copied_path = os.path.join(str(outputs_root), "session-1", "codex_image_gen", "blue-square.png")
    assert copied_path in tool_payload
    assert os.path.exists(copied_path)
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "done"
