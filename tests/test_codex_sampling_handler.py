from types import SimpleNamespace

from mcp.types import TextContent

from open_storyline.codex.sampling_handler import (
    CodexSamplingLLMClient,
    make_codex_sampling_callback,
)


def test_codex_sampling_callback_uses_codex_turns_for_text_generation() -> None:
    class FakeClient:
        def __init__(self):
            self.thread_params = None
            self.turn_params = None

        async def thread_start(self, **kwargs):
            self.thread_params = kwargs
            return {"thread": {"id": "thread-1"}}

        async def turn_start(self, **kwargs):
            self.turn_params = kwargs
            return {"turnId": "turn-1"}

        async def next_notification(self, timeout=None):
            return {"method": "turn/completed", "params": {"turnId": "turn-1"}}

        async def close(self):
            return None

    fake_client = FakeClient()
    callback = make_codex_sampling_callback(
        binary="codex",
        cwd="/tmp/workspace",
        model="gpt-5.4",
        reasoning_effort="xhigh",
        client_factory=lambda: fake_client,
    )

    params = SimpleNamespace(
        systemPrompt="system rules",
        messages=[
            SimpleNamespace(role="user", content=TextContent(type="text", text="hello from node")),
        ],
        metadata={},
        temperature=0.3,
        maxTokens=256,
    )

    result = __import__("asyncio").run(callback(None, params))

    assert result.content.text == ""
    assert fake_client.thread_params["ephemeral"] is True
    assert fake_client.turn_params["effort"] == "xhigh"
    assert fake_client.turn_params["input_items"] == [
        {"type": "text", "text": "System: system rules\n\nUser: hello from node"}
    ]


def test_codex_sampling_llm_client_supports_multimodal_completion(monkeypatch) -> None:
    monkeypatch.setattr(
        "open_storyline.codex.sampling_handler._build_media_blocks",
        lambda *args, **kwargs: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}],
    )

    class FakeClient:
        def __init__(self):
            self.thread_params = None
            self.turn_params = None
            self.closed = False
            self._notifications = iter(
                (
                    {"method": "item/agentMessage/delta", "params": {"delta": "海边"}},
                    {
                        "method": "item/completed",
                        "params": {"item": {"type": "agentMessage", "text": "海边有一只可爱的狗狗。"}},
                    },
                    {"method": "turn/completed", "params": {"turnId": "turn-1"}},
                )
            )

        async def thread_start(self, **kwargs):
            self.thread_params = kwargs
            return {"thread": {"id": "thread-1"}}

        async def turn_start(self, **kwargs):
            self.turn_params = kwargs
            return {"turnId": "turn-1"}

        async def next_notification(self, timeout=None):
            del timeout
            return next(self._notifications)

        async def close(self):
            self.closed = True

    fake_client = FakeClient()
    llm = CodexSamplingLLMClient(
        binary="codex",
        cwd="/tmp/workspace",
        model="gpt-5.4",
        reasoning_effort="xhigh",
        client_factory=lambda: fake_client,
    )

    result = __import__("asyncio").run(
        llm.complete(
            system_prompt="请用中文描述图片。",
            user_prompt="描述这张图",
            media=[{"path": "/tmp/frame.png"}],
            max_tokens=256,
        )
    )

    assert result == "海边有一只可爱的狗狗。"
    assert fake_client.thread_params["ephemeral"] is True
    assert fake_client.turn_params["effort"] == "xhigh"
    assert fake_client.turn_params["input_items"] == [
        {"type": "text", "text": "System: 请用中文描述图片。\n\nUser: 描述这张图"},
        {"type": "image", "url": "data:image/png;base64,ZmFrZQ=="},
    ]
    assert fake_client.closed is True
