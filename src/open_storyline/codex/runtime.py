from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import re
import shutil
from base64 import b64encode
from types import SimpleNamespace
from uuid import uuid4
from typing import Any, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from open_storyline.codex.app_server_client import CodexAppServerClient, CodexAppServerError
from open_storyline.openai_agents_runtime import (
    LangChainToolRuntimeShim,
    _coerce_langchain_tool_output,
    _invoke_langchain_tool,
    _mask_secrets,
    _sanitize_tool_args,
    _stringify_tool_output,
    _tool_event_name,
)
from open_storyline.mcp.hooks.chat_middleware import (
    get_mcp_log_sink,
    reset_active_tool_call_id,
    set_active_tool_call_id,
)

_IMAGEGEN_SKILL_NAME = "imagegen"
_IMAGEGEN_TRIGGER_PREFIX = "$imagegen"
_IMAGEGEN_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_IMAGEGEN_KEYWORDS = (
    "image",
    "images",
    "picture",
    "illustration",
    "icon",
    "poster",
    "cover art",
    "thumbnail",
    "logo",
    "mockup",
    "generate an image",
    "generate image",
    "create an image",
    "create image",
    "make an image",
    "make image",
    "draw ",
    "edit image",
    "remove background",
    "transparent background",
    "生图",
    "生一张图",
    "生成图片",
    "生成一张图",
    "產生圖片",
    "產生一張圖",
    "畫一張",
    "画一张",
    "插画",
    "插圖",
    "海报",
    "封面",
    "圖示",
    "图示",
    "透明背景",
    "去背",
)
_IMAGEGEN_VERBS = (
    "create",
    "generate",
    "make",
    "draw",
    "edit",
    "remove background",
    "生成",
    "產生",
    "创建",
    "創建",
    "画",
    "畫",
    "修改",
    "調整",
    "去背",
)
_IMAGEGEN_NOUNS = (
    "image",
    "images",
    "picture",
    "illustration",
    "icon",
    "poster",
    "cover",
    "thumbnail",
    "logo",
    "mockup",
    "圖片",
    "图片",
    "圖像",
    "图像",
    "插畫",
    "插画",
    "海报",
    "封面",
    "圖示",
    "图示",
    "背景",
)


def _default_developer_instructions() -> str:
    return (
        "You are the multimodal assistant embedded inside FireRed OpenStoryline. "
        "Help the user complete editing tasks by using the available FireRed tools when useful, "
        "and answer directly in plain text when no tool is needed. "
        "Do not mention internal harness details, repository bootstrap steps, AGENTS files, "
        "superpowers, developer messages, system prompts, sandbox settings, approval policies, "
        "or other hidden implementation details unless the user explicitly asks about them. "
        "Never tell the user that you need to run internal setup before helping. "
        "If the user directly asks you to run a specific FireRed tool or has already confirmed the next step, "
        "execute that tool immediately instead of asking for redundant confirmation. "
        "Keep the conversation focused on the user's creative task and tool results."
    )


def _message_role(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "") or message.__class__.__name__.lower()
    text = str(message_type).lower()
    if "system" in text:
        return "System"
    if "human" in text or "user" in text:
        return "User"
    if "tool" in text:
        return "Tool"
    return "Assistant"


def _looks_like_imagegen_request(messages: Sequence[BaseMessage], attachments: Iterable[Any]) -> bool:
    latest_user = ""
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            latest_user = message.content.strip()
            break
    text = latest_user.lower()
    if not text:
        return False
    if _IMAGEGEN_TRIGGER_PREFIX in text:
        return True
    if any(keyword in text for keyword in _IMAGEGEN_KEYWORDS):
        return True
    if any(verb in text for verb in _IMAGEGEN_VERBS) and any(noun in text for noun in _IMAGEGEN_NOUNS):
        return True
    has_image_attachment = any(
        str(getattr(item, "kind", "") or "").strip().lower() == "image"
        for item in (attachments or [])
    )
    return has_image_attachment and any(token in text for token in ("edit", "modify", "change", "background", "crop", "調整", "修改", "去背", "背景"))


def _default_codex_home() -> str:
    return os.path.expanduser(os.getenv("CODEX_HOME") or "~/.codex")


def _resolve_imagegen_skill_path(explicit_path: str | None = None) -> str:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    codex_home = _default_codex_home()
    candidates.extend(
        [
            os.path.join(codex_home, "skills", ".system", "imagegen", "SKILL.md"),
            os.path.join(codex_home, "skills", "imagegen", "SKILL.md"),
        ]
    )
    for candidate in candidates:
        path = os.path.abspath(os.path.expanduser(str(candidate)))
        if os.path.isfile(path):
            return path
    return ""


def _generated_images_root(codex_home: str | None) -> str:
    home = str(codex_home or "").strip() or _default_codex_home()
    return os.path.join(home, "generated_images")


def _snapshot_generated_images(root: str) -> dict[str, tuple[float, int]]:
    snapshot: dict[str, tuple[float, int]] = {}
    if not root or not os.path.isdir(root):
        return snapshot
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in _IMAGEGEN_FILE_EXTENSIONS:
                continue
            path = os.path.abspath(os.path.join(dirpath, filename))
            try:
                stat = os.stat(path)
            except OSError:
                continue
            snapshot[path] = (float(stat.st_mtime), int(stat.st_size))
    return snapshot


def _new_generated_images(before: dict[str, tuple[float, int]], after: dict[str, tuple[float, int]]) -> list[str]:
    changed = [path for path, sig in after.items() if before.get(path) != sig]
    return sorted(changed, key=lambda path: after[path][0])


async def _wait_for_new_generated_images(
    root: str,
    before: dict[str, tuple[float, int]],
    *,
    timeout_seconds: float = 8.0,
    poll_interval_seconds: float = 0.5,
) -> list[str]:
    deadline = asyncio.get_running_loop().time() + max(timeout_seconds, 0.0)
    while True:
        after = _snapshot_generated_images(root)
        changed = _new_generated_images(before, after)
        if changed:
            return changed
        if asyncio.get_running_loop().time() >= deadline:
            return []
        await asyncio.sleep(max(poll_interval_seconds, 0.05))


def _copy_generated_images_to_session_outputs(image_paths: Sequence[str], outputs_dir: str, session_id: str) -> list[str]:
    if not image_paths or not outputs_dir or not session_id:
        return []
    target_dir = os.path.join(outputs_dir, session_id, "codex_image_gen")
    os.makedirs(target_dir, exist_ok=True)
    copied: list[str] = []
    for index, src in enumerate(image_paths, start=1):
        ext = os.path.splitext(src)[1].lower() or ".png"
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", os.path.splitext(os.path.basename(src))[0]).strip("-._") or f"codex-image-{index}"
        dst = os.path.join(target_dir, f"{stem}{ext}")
        if os.path.exists(dst):
            dst = os.path.join(target_dir, f"{stem}-{uuid4().hex[:8]}{ext}")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def _imagegen_tool_summary(image_paths: Sequence[str]) -> dict[str, Any]:
    return {
        "INFO_USER": f"Codex image generation created {len(image_paths)} image(s).",
        "preview_urls": list(image_paths),
        "generated_files": [{"path": path, "kind": "image"} for path in image_paths],
    }


def build_codex_input_items(
    messages: Sequence[BaseMessage],
    attachments: Iterable[Any],
    *,
    imagegen_trigger: bool = False,
    imagegen_skill_path: str = "",
) -> list[dict[str, Any]]:
    transcript_parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        transcript_parts.append(f"{_message_role(message)}: {content}")

    items: list[dict[str, Any]] = []
    transcript = "\n\n".join(transcript_parts).strip()
    if transcript:
        if imagegen_trigger and _IMAGEGEN_TRIGGER_PREFIX not in transcript:
            transcript = f"{_IMAGEGEN_TRIGGER_PREFIX}\n{transcript}"
        items.append({"type": "text", "text": transcript})

    if imagegen_trigger and imagegen_skill_path:
        items.append({"type": "skill", "name": _IMAGEGEN_SKILL_NAME, "path": imagegen_skill_path})

    for attachment in attachments or []:
        path = str(getattr(attachment, "path", "") or "").strip()
        kind = str(getattr(attachment, "kind", "") or "").strip().lower()
        if not path:
            continue
        if kind == "image":
            items.append({"type": "localImage", "path": path})

    return items


def _notification_delta(notification: dict[str, Any]) -> str:
    params = notification.get("params")
    if not isinstance(params, dict):
        return ""
    delta = params.get("delta")
    if isinstance(delta, str):
        return delta
    item = params.get("item")
    if isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str):
            return text
    return ""


def _notification_completed_text(notification: dict[str, Any]) -> str:
    params = notification.get("params")
    if not isinstance(params, dict):
        return ""
    item = params.get("item")
    if not isinstance(item, dict):
        return ""
    text = item.get("text")
    if isinstance(text, str):
        return text.strip()
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def _tool_params_json_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if isinstance(args_schema, dict):
        return args_schema
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    return {
        "type": "object",
        "properties": getattr(tool, "args", {}) or {},
    }


def _scrub_tool_output(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).lower() == "base64" and isinstance(item, str):
                out[key] = "<base64 omitted>"
            else:
                out[key] = _scrub_tool_output(item)
        return out
    if isinstance(value, list):
        return [_scrub_tool_output(item) for item in value]
    return value


def _path_to_data_url(path: str) -> str | None:
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        encoded = b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _collect_image_urls(value: Any, *, limit: int = 4) -> list[str]:
    urls: list[str] = []

    def walk(item: Any) -> None:
        if len(urls) >= limit:
            return
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str):
                data_url = _path_to_data_url(path)
                if data_url and data_url not in urls:
                    urls.append(data_url)
            raw_url = item.get("url") or item.get("imageUrl")
            if isinstance(raw_url, str) and raw_url.startswith(("http://", "https://", "data:image/")):
                if raw_url not in urls:
                    urls.append(raw_url)
            for nested in item.values():
                walk(nested)
            return
        if isinstance(item, list):
            for nested in item:
                walk(nested)

    walk(value)
    return urls


def _tool_output_to_content_items(tool_output: Any) -> list[dict[str, Any]]:
    scrubbed = _scrub_tool_output(tool_output)
    items: list[dict[str, Any]] = [
        {
            "type": "inputText",
            "text": _stringify_tool_output(scrubbed),
        }
    ]
    for image_url in _collect_image_urls(scrubbed):
        items.append({"type": "inputImage", "imageUrl": image_url})
    return items


def _tool_call_to_ai_message(call_id: str, name: str, arguments: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": str(call_id),
                "name": str(name),
                "args": arguments,
                "type": "tool_call",
            }
        ],
    )


class CodexRuntime:
    def __init__(
        self,
        *,
        binary: str,
        cwd: str,
        model: str,
        sandbox: str = "read-only",
        approval_policy: str = "never",
        reasoning_effort: str | None = None,
        developer_instructions: str | None = None,
        langchain_tools: Sequence[Any] | None = None,
        store: Any = None,
        imagegen_skill_path: str | None = None,
        client_factory=None,
    ):
        self.binary = binary
        self.cwd = cwd
        self.model = model
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.reasoning_effort = str(reasoning_effort or "").strip()
        self.developer_instructions = developer_instructions or _default_developer_instructions()
        self.langchain_tools = list(langchain_tools or [])
        self.store = store
        self.imagegen_skill_path = _resolve_imagegen_skill_path(imagegen_skill_path)
        self.client_factory = client_factory
        self.dynamic_tools: list[dict[str, Any]] = []
        self._tool_invokers: dict[str, Any] = {}
        for tool in self.langchain_tools:
            tool_name = str(getattr(tool, "name", "") or "")
            if not tool_name:
                continue
            self.dynamic_tools.append(
                {
                    "name": tool_name,
                    "description": str(getattr(tool, "description", "") or ""),
                    "inputSchema": _tool_params_json_schema(tool),
                    "deferLoading": False,
                }
            )
            self._tool_invokers[tool_name] = self._make_tool_invoker(tool)

    def _build_client(self) -> CodexAppServerClient:
        if self.client_factory is not None:
            return self.client_factory()
        return CodexAppServerClient(binary=self.binary, cwd=self.cwd)

    def _make_tool_invoker(self, tool: Any):
        async def invoke(call_id: str, arguments: dict[str, Any], context: Any) -> tuple[Any, bool]:
            tool_name = str(getattr(tool, "name", "") or "")
            server_name, display_tool_name = _tool_event_name(tool_name)
            safe_args = _sanitize_tool_args(tool_name, arguments, context)
            sink = get_mcp_log_sink()
            runtime = LangChainToolRuntimeShim(
                context=context,
                store=self.store,
                tool_call_id=call_id,
            )

            if sink:
                sink(
                    {
                        "type": "tool_start",
                        "tool_call_id": call_id,
                        "server": server_name,
                        "name": display_tool_name,
                        "args": safe_args,
                    }
                )

            active_token = set_active_tool_call_id(call_id)
            try:
                raw_result = await _invoke_langchain_tool(tool, runtime, arguments)
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
                        "tool_call_id": call_id,
                        "server": server_name,
                        "name": display_tool_name,
                        "is_error": bool(is_error),
                        "summary": _mask_secrets(summary),
                    }
                )

            return tool_output, bool(is_error)

        return invoke

    async def astream(self, payload: dict[str, Any], *, context: Any = None, stream_mode: Sequence[str] | None = None):
        messages = list(payload.get("messages") or [])
        attachments = list(getattr(context, "codex_turn_attachments", []) or [])
        codex_model = str(getattr(context, "codex_model", "") or "").strip() or self.model
        codex_reasoning_effort = str(getattr(context, "codex_reasoning_effort", "") or "").strip() or self.reasoning_effort
        use_imagegen = bool(self.imagegen_skill_path) and _looks_like_imagegen_request(messages, attachments)
        input_items = build_codex_input_items(
            messages,
            attachments,
            imagegen_trigger=use_imagegen,
            imagegen_skill_path=self.imagegen_skill_path,
        )
        if not input_items:
            return

        stream_mode = tuple(stream_mode or [])
        client = self._build_client()
        final_text = ""
        protocol_messages: list[BaseMessage] = []
        notification_timeout = 300.0 if use_imagegen else 60.0
        generated_before = _snapshot_generated_images(_generated_images_root(getattr(client, "codex_home", None))) if use_imagegen else {}
        try:
            thread = await client.thread_start(
                cwd=self.cwd,
                model=codex_model,
                developer_instructions=self.developer_instructions,
                sandbox=self.sandbox,
                approval_policy=self.approval_policy,
                dynamic_tools=self.dynamic_tools or None,
            )
            thread_info = thread.get("thread") if isinstance(thread, dict) else None
            thread_id = str(
                (thread_info or {}).get("id")
                or thread.get("threadId")
                or thread.get("id")
                or ""
            ).strip()
            if not thread_id:
                raise CodexAppServerError("Codex thread/start did not return a thread id")

            await client.turn_start(
                thread_id=thread_id,
                input_items=input_items,
                model=codex_model,
                cwd=self.cwd,
                approval_policy=self.approval_policy,
                effort=codex_reasoning_effort or None,
            )

            while True:
                notification = await client.next_notification(timeout=notification_timeout)
                method = str(notification.get("method") or "").strip()
                if method == "item/tool/call":
                    params = notification.get("params") if isinstance(notification.get("params"), dict) else {}
                    request_id = notification.get("id")
                    call_id = str(params.get("callId") or "").strip()
                    tool_name = str(params.get("tool") or "").strip()
                    arguments = params.get("arguments")
                    if not isinstance(arguments, dict):
                        arguments = {}

                    protocol_messages.append(_tool_call_to_ai_message(call_id, tool_name, arguments))
                    tool_output: Any
                    is_error = True
                    invoker = self._tool_invokers.get(tool_name)
                    if invoker is None:
                        tool_output = f"Tool call failed\nTool name: {tool_name}\nError messege: unknown tool"
                    else:
                        tool_output, is_error = await invoker(call_id, arguments, context)
                    protocol_messages.append(
                        ToolMessage(
                            content=_stringify_tool_output(tool_output),
                            tool_call_id=call_id,
                        )
                    )

                    if request_id is not None:
                        await client.respond(
                            int(request_id),
                            result={
                                "contentItems": _tool_output_to_content_items(tool_output),
                                "success": not is_error,
                            },
                        )
                    continue

                if method == "item/agentMessage/delta":
                    delta = _notification_delta(notification)
                    if delta:
                        final_text += delta
                        if "messages" in stream_mode:
                            yield "messages", (SimpleNamespace(content=delta), {"langgraph_node": "model"})
                    continue

                if method == "item/completed":
                    completed_text = _notification_completed_text(notification)
                    if completed_text:
                        final_text = completed_text
                    continue

                if method == "turn/completed":
                    break

                if method == "error":
                    raise CodexAppServerError(str(notification.get("params") or "Codex turn failed"))

            if use_imagegen:
                generated_root = _generated_images_root(getattr(client, "codex_home", None))
                new_images = await _wait_for_new_generated_images(
                    generated_root,
                    generated_before,
                )
                copied_images = _copy_generated_images_to_session_outputs(
                    new_images,
                    str(getattr(context, "outputs_dir", "") or ""),
                    str(getattr(context, "session_id", "") or ""),
                )
                if copied_images:
                    call_id = f"codex_image_gen_{uuid4().hex[:12]}"
                    summary = _imagegen_tool_summary(copied_images)
                    sink = get_mcp_log_sink()
                    if sink:
                        sink(
                            {
                                "type": "tool_start",
                                "tool_call_id": call_id,
                                "server": "codex",
                                "name": "image_gen",
                                "args": {"source": "codex_builtin"},
                            }
                        )
                        sink(
                            {
                                "type": "tool_end",
                                "tool_call_id": call_id,
                                "server": "codex",
                                "name": "image_gen",
                                "is_error": False,
                                "summary": summary,
                            }
                        )
                    protocol_messages.append(_tool_call_to_ai_message(call_id, "image_gen", {"source": "codex_builtin"}))
                    protocol_messages.append(
                        ToolMessage(
                            content=json.dumps(summary, ensure_ascii=False),
                            tool_call_id=call_id,
                        )
                    )

            if final_text.strip():
                protocol_messages.append(AIMessage(content=final_text.strip()))

            if "updates" in stream_mode:
                yield "updates", {"codex": {"messages": protocol_messages}}
        finally:
            with contextlib.suppress(Exception):
                await client.close()
