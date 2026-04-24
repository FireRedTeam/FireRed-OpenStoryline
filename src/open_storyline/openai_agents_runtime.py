from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable, Sequence

from agents import Agent as OpenAIAgent
from agents import FunctionTool, RawResponsesStreamEvent, RunItemStreamEvent, Runner
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from open_storyline.mcp.hooks.chat_middleware import (
    get_mcp_log_sink,
    reset_active_tool_call_id,
    set_active_tool_call_id,
)

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {
    "api_key",
    "access_token",
    "authorization",
    "token",
    "password",
    "secret",
    "x-api-key",
    "apikey",
}

_TOOL_EVENT_ARG_EXCLUDE = {
    "inputs",
    "artifacts_dir",
    "artifact_id",
    "blobs_dir",
    "meta_path",
    "media_dir",
    "bgm_dir",
    "outputs_dir",
    "debug_dir",
}


@dataclass
class LangChainToolRuntimeShim:
    context: Any
    store: Any
    tool_call_id: str


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "output_text", "input_text"}:
                    texts.append(str(item.get("text", "")))
        return "".join(texts)

    return ""


def _stringify_tool_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    payload = tool_call if isinstance(tool_call, dict) else None
    if payload is None:
        payload = {
            "id": getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None),
            "name": getattr(tool_call, "name", None),
            "args": getattr(tool_call, "args", None),
        }

    call_id = payload.get("id") or payload.get("tool_call_id")
    if not call_id:
        return None

    if isinstance(payload.get("function"), dict):
        fn = payload["function"]
        name = fn.get("name")
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = arguments
        else:
            args = arguments
    else:
        name = payload.get("name")
        args = payload.get("args")
        if args is None and isinstance(payload.get("arguments"), str):
            try:
                args = json.loads(payload["arguments"])
            except json.JSONDecodeError:
                args = payload["arguments"]

    return {
        "id": str(call_id),
        "name": str(name or ""),
        "args": args if isinstance(args, dict) else args or {},
        "type": "tool_call",
    }


def _extract_ai_tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    raw_calls = list(getattr(message, "tool_calls", None) or [])
    if not raw_calls:
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        raw_calls = list(additional_kwargs.get("tool_calls") or [])

    normalized: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        call = _normalize_tool_call(raw_call)
        if call is not None:
            normalized.append(call)
    return normalized


def lc_messages_to_openai_input_items(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            content = _text_from_content(message.content)
            if content:
                items.append({"role": "system", "content": content})
            continue

        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, list):
                items.append({"role": "user", "content": content})
            else:
                items.append({"role": "user", "content": _text_from_content(content)})
            continue

        if isinstance(message, AIMessage):
            content = _text_from_content(message.content)
            if content:
                items.append({"role": "assistant", "content": content})

            for tool_call in _extract_ai_tool_calls(message):
                args = tool_call.get("args", {})
                arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "arguments": arguments,
                    }
                )
            continue

        if isinstance(message, ToolMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(getattr(message, "tool_call_id", "") or ""),
                    "output": _stringify_tool_output(message.content),
                }
            )

    return items


def _mask_secrets(obj: Any) -> Any:
    try:
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if str(key).lower() in _SENSITIVE_KEYS:
                    out[key] = "***"
                else:
                    out[key] = _mask_secrets(value)
            return out
        if isinstance(obj, list):
            return [_mask_secrets(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_mask_secrets(item) for item in obj)
        return obj
    except Exception:
        return "***"


def _tool_event_name(tool_name: str) -> tuple[str, str]:
    if tool_name.startswith("storyline_"):
        return "storyline", tool_name[len("storyline_") :]
    return "", tool_name


def _sanitize_tool_args(tool_name: str, arguments: dict[str, Any], context: Any) -> dict[str, Any]:
    exclude = set(_TOOL_EVENT_ARG_EXCLUDE)
    node_manager = getattr(context, "node_manager", None)
    if node_manager is not None:
        exclude.update(getattr(node_manager, "kind_to_node_ids", {}).keys())

    cleaned = {
        key: value
        for key, value in (arguments or {}).items()
        if key not in exclude
    }

    # Default-mode invocations route through the real args payload.
    if arguments.get("tool_call_type") != "auto" and isinstance(arguments.get("args"), dict):
        cleaned = {
            key: value
            for key, value in arguments["args"].items()
            if key not in exclude
        }

    return _mask_secrets(cleaned)


def _command_to_tool_message(command: Any) -> ToolMessage | None:
    update = getattr(command, "update", None)
    if not isinstance(update, dict):
        return None

    messages = update.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if isinstance(message, ToolMessage):
            return message
    return None


def _content_blocks_to_tool_output(content: Any) -> Any:
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        if text_parts:
            return "\n".join(part for part in text_parts if part)
    return content


def _coerce_langchain_tool_output(raw_result: Any) -> tuple[Any, bool, Any]:
    result = raw_result[0] if isinstance(raw_result, tuple) and len(raw_result) == 2 else raw_result

    if isinstance(result, ToolMessage):
        tool_output = result.content
        is_error = bool((getattr(result, "additional_kwargs", None) or {}).get("isError"))
        summary = tool_output if is_error else tool_output
        return tool_output, is_error, summary

    tool_message = _command_to_tool_message(result)
    if tool_message is not None:
        tool_output = tool_message.content
        is_error = bool(isinstance(tool_output, dict) and tool_output.get("isError"))
        summary = tool_output.get("summary", tool_output) if isinstance(tool_output, dict) else tool_output
        return tool_output, is_error, summary

    tool_output = _content_blocks_to_tool_output(result)
    if isinstance(tool_output, dict):
        is_error = bool(tool_output.get("isError"))
        summary = tool_output.get("summary", tool_output)
        return tool_output, is_error, summary

    return tool_output, False, tool_output


def _tool_output_to_langchain_message(tool_call_id: str, output: Any) -> ToolMessage:
    return ToolMessage(content=output, tool_call_id=tool_call_id)


def _run_item_text(item: Any) -> str:
    raw_item = getattr(item, "raw_item", None)
    content = getattr(raw_item, "content", None)
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "output_text":
            texts.append(str(getattr(block, "text", "") or ""))
    return "".join(texts)


def _tool_called_to_ai_message(item: Any) -> AIMessage | None:
    raw_item = getattr(item, "raw_item", None)
    call_id = getattr(raw_item, "call_id", None) or getattr(raw_item, "id", None)
    name = getattr(raw_item, "name", None)
    arguments = getattr(raw_item, "arguments", None)

    if not call_id or not name:
        return None

    args: Any
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = arguments
    else:
        args = arguments

    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": str(call_id),
                "name": str(name),
                "args": args if isinstance(args, dict) else args or {},
                "type": "tool_call",
            }
        ],
    )


async def _invoke_langchain_tool(tool: Any, runtime: LangChainToolRuntimeShim, args: dict[str, Any]) -> Any:
    coroutine = getattr(tool, "coroutine", None)
    if callable(coroutine):
        kwargs = dict(args)
        if "runtime" in inspect.signature(coroutine).parameters:
            kwargs["runtime"] = runtime
        return await coroutine(**kwargs)

    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(args)

    if hasattr(tool, "invoke"):
        return await asyncio.to_thread(tool.invoke, args)

    raise TypeError(f"Unsupported tool type for {getattr(tool, 'name', '<unknown>')}")


def build_agents_function_tools(langchain_tools: Sequence[Any], store: Any) -> list[FunctionTool]:
    wrapped_tools: list[FunctionTool] = []
    seen_names: set[str] = set()

    for tool in langchain_tools:
        args_schema = getattr(tool, "args_schema", None)
        params_json_schema: dict[str, Any]
        if isinstance(args_schema, dict):
            params_json_schema = args_schema
        elif hasattr(args_schema, "model_json_schema"):
            params_json_schema = args_schema.model_json_schema()
        else:
            params_json_schema = {
                "type": "object",
                "properties": getattr(tool, "args", {}) or {},
            }

        async def on_invoke_tool(
            ctx: Any,
            raw_input: str,
            *,
            _tool: Any = tool,
        ) -> Any:
            try:
                arguments = json.loads(raw_input or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON tool input for {getattr(_tool, 'name', 'tool')}: {exc}") from exc

            if not isinstance(arguments, dict):
                raise ValueError(f"Tool input for {getattr(_tool, 'name', 'tool')} must be a JSON object")

            tool_name = str(getattr(_tool, "name", "") or "")
            server_name, display_tool_name = _tool_event_name(tool_name)
            safe_args = _sanitize_tool_args(tool_name, arguments, ctx.context)
            sink = get_mcp_log_sink()
            runtime = LangChainToolRuntimeShim(
                context=ctx.context,
                store=store,
                tool_call_id=ctx.tool_call_id,
            )

            if sink:
                sink(
                    {
                        "type": "tool_start",
                        "tool_call_id": ctx.tool_call_id,
                        "server": server_name,
                        "name": display_tool_name,
                        "args": safe_args,
                    }
                )

            active_token = set_active_tool_call_id(ctx.tool_call_id)
            try:
                raw_result = await _invoke_langchain_tool(_tool, runtime, arguments)
                tool_output, is_error, summary = _coerce_langchain_tool_output(raw_result)
            except Exception as exc:
                tool_output = (
                    "Tool call failed\n"
                    f"Tool name: {tool_name}\n"
                    f"Tool params: {safe_args}\n"
                    f"Error messege: {type(exc).__name__}: {exc}\n"
                    "If it is a parameter issue, please correct the parameters and call again. "
                    "If it is due to the lack of a preceding dependency, please call the preceding node first. "
                    "If you think it's an occasional error, please try to call it again; "
                    "If you think it's impossible to continue, please explain the reason to the user."
                )
                is_error = True
                summary = tool_output
            finally:
                reset_active_tool_call_id(active_token)

            if sink:
                sink(
                    {
                        "type": "tool_end",
                        "tool_call_id": ctx.tool_call_id,
                        "server": server_name,
                        "name": display_tool_name,
                        "is_error": bool(is_error),
                        "summary": _mask_secrets(summary),
                    }
                )

            return tool_output

        tool_names = [str(getattr(tool, "name", "") or "").strip()]
        if tool_names[0].startswith("storyline_"):
            alias_name = tool_names[0][len("storyline_") :].strip()
            if alias_name:
                tool_names.append(alias_name)

        for tool_name in tool_names:
            if not tool_name or tool_name in seen_names:
                continue
            wrapped_tools.append(
                FunctionTool(
                    name=tool_name,
                    description=str(getattr(tool, "description", "") or ""),
                    params_json_schema=params_json_schema,
                    on_invoke_tool=on_invoke_tool,
                    strict_json_schema=False,
                )
            )
            seen_names.add(tool_name)

    return wrapped_tools


class OpenAIAgentsRuntime:
    def __init__(self, agent: OpenAIAgent[Any]):
        self.agent = agent

    async def astream(
        self,
        payload: dict[str, Any],
        *,
        context: Any = None,
        stream_mode: Sequence[str] | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        del stream_mode

        messages = payload.get("messages") or []
        input_items = lc_messages_to_openai_input_items(messages)
        result = Runner.run_streamed(self.agent, input_items, context=context)

        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                data = getattr(event, "data", None)
                delta = getattr(data, "delta", None)
                event_type = str(getattr(data, "type", "") or "")
                if isinstance(delta, str) and "text" in event_type and "delta" in event_type:
                    yield "messages", (SimpleNamespace(content=delta), {"langgraph_node": "model"})
                continue

            if not isinstance(event, RunItemStreamEvent):
                continue

            if event.name == "message_output_created":
                text = _run_item_text(event.item)
                if text:
                    yield "updates", {"agents_sdk": {"messages": [AIMessage(content=text)]}}
                continue

            if event.name == "tool_called":
                message = _tool_called_to_ai_message(event.item)
                if message is not None:
                    yield "updates", {"agents_sdk": {"messages": [message]}}
                continue

            if event.name == "tool_output":
                raw_item = getattr(event.item, "raw_item", None)
                call_id = getattr(raw_item, "call_id", None)
                if not call_id and isinstance(raw_item, dict):
                    call_id = raw_item.get("call_id")
                if call_id:
                    message = _tool_output_to_langchain_message(
                        str(call_id),
                        getattr(event.item, "output", ""),
                    )
                    yield "updates", {"agents_sdk": {"messages": [message]}}
