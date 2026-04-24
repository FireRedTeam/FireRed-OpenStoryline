import asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_storyline.mcp.hooks.chat_middleware import reset_mcp_log_sink, set_mcp_log_sink
from open_storyline.openai_agents_runtime import (
    build_agents_function_tools,
    lc_messages_to_openai_input_items,
)


def test_lc_messages_to_openai_input_items_preserve_tool_protocol() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="hello"),
        AIMessage(
            content="I will call a tool.",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "storyline_load_media",
                    "args": {"mode": "default"},
                }
            ],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="call_1"),
        AIMessage(content="done"),
    ]

    items = lc_messages_to_openai_input_items(messages)

    assert items == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I will call a tool."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "storyline_load_media",
            "arguments": '{"mode": "default"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok": true}',
        },
        {"role": "assistant", "content": "done"},
    ]


def test_lc_messages_to_openai_input_items_reads_tool_calls_from_additional_kwargs() -> None:
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "storyline_search_media",
                            "arguments": '{"query": "cat"}',
                        },
                    }
                ]
            },
        ),
        ToolMessage(content="search complete", tool_call_id="call_2"),
    ]

    items = lc_messages_to_openai_input_items(messages)

    assert items == [
        {"role": "user", "content": "hello"},
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "storyline_search_media",
            "arguments": '{"query": "cat"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "search complete",
        },
    ]


def test_build_agents_function_tools_emits_tool_events_and_passes_runtime() -> None:
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

    tools = build_agents_function_tools([FakeTool()], store="artifact-store")
    assert [tool.name for tool in tools] == ["storyline_fake_node", "fake_node"]
    tool = tools[1]
    events = []

    ctx = type(
        "Ctx",
        (),
        {
            "tool_call_id": "call_3",
            "context": type("ClientContext", (), {"node_manager": None})(),
        },
    )()

    async def run_tool():
        token = set_mcp_log_sink(events.append)
        try:
            return await tool.on_invoke_tool(ctx, '{"foo": "bar"}')
        finally:
            reset_mcp_log_sink(token)

    output = asyncio.run(run_tool())

    assert output == {"summary": {"node_summary": "done"}, "isError": False}
    assert seen_runtime["runtime"].store == "artifact-store"
    assert seen_runtime["runtime"].tool_call_id == "call_3"
    assert seen_runtime["kwargs"] == {"foo": "bar"}
    assert events == [
        {
            "type": "tool_start",
            "tool_call_id": "call_3",
            "server": "storyline",
            "name": "fake_node",
            "args": {"foo": "bar"},
        },
        {
            "type": "tool_end",
            "tool_call_id": "call_3",
            "server": "storyline",
            "name": "fake_node",
            "is_error": False,
            "summary": {"node_summary": "done"},
        },
    ]
