from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent

from open_storyline.codex.app_server_client import CodexAppServerClient, CodexAppServerError
from open_storyline.mcp.sampling_handler import (
    _build_media_blocks,
    _extract_text_from_mcp_content,
)
from open_storyline.mcp.sampling_requester import LLMClient


DEFAULT_RESIZE_EDGE = 1280
DEFAULT_JPEG_QUALITY = 82
DEFAULT_MIN_FRAMES = 3
DEFAULT_MAX_FRAMES = 8
DEFAULT_FRAMES_PER_SEC = 0.5
GLOBAL_MAX_IMAGE_BLOCKS = 12


def _build_codex_prompt_items(
    *,
    system_prompt: str | None,
    user_prompt: str,
    media_inputs: list[Any],
    resize_edge: int,
    jpeg_quality: int,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
    global_max_images: int,
) -> list[dict[str, Any]]:
    message = SimpleNamespace(
        role="user",
        content=TextContent(type="text", text=str(user_prompt or "")),
    )
    return _sampling_messages_to_input_items(
        system_prompt,
        [message],
        media_inputs,
        resize_edge=resize_edge,
        jpeg_quality=jpeg_quality,
        min_frames=min_frames,
        max_frames=max_frames,
        frames_per_sec=frames_per_sec,
        global_max_images=global_max_images,
    )


async def _run_codex_sampling_turn(
    *,
    binary: str,
    cwd: str,
    model: str,
    reasoning_effort: str | None,
    input_items: list[dict[str, Any]],
    sandbox: str,
    approval_policy: str,
    client_factory=None,
) -> str:
    client = client_factory() if client_factory is not None else CodexAppServerClient(binary=binary, cwd=cwd)
    final_text = ""
    try:
        thread = await client.thread_start(
            cwd=cwd,
            model=model,
            developer_instructions="You are a concise assistant helping FireRed tools complete subtasks.",
            sandbox=sandbox,
            approval_policy=approval_policy,
            ephemeral=True,
        )
        thread_info = thread.get("thread") if isinstance(thread, dict) else None
        thread_id = str(
            (thread_info or {}).get("id")
            or thread.get("threadId")
            or thread.get("id")
            or ""
        ).strip()
        if not thread_id:
            raise RuntimeError("Codex thread/start did not return a thread id")

        await client.turn_start(
            thread_id=thread_id,
            input_items=input_items,
            model=model,
            cwd=cwd,
            approval_policy=approval_policy,
            effort=str(reasoning_effort or "").strip() or None,
        )

        while True:
            notification = await client.next_notification(timeout=60.0)
            method = str(notification.get("method") or "").strip()
            if method == "item/agentMessage/delta":
                delta = (notification.get("params") or {}).get("delta")
                if isinstance(delta, str):
                    final_text += delta
                continue
            if method == "item/completed":
                item = (notification.get("params") or {}).get("item")
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    completed_text = item.get("text")
                    if isinstance(completed_text, str):
                        final_text = completed_text.strip()
                continue
            if method == "turn/completed":
                break
            if method == "error":
                raise CodexAppServerError(str(notification.get("params") or "Codex turn failed"))

        return final_text.strip()
    finally:
        try:
            await client.close()
        except Exception:
            pass


class CodexSamplingLLMClient(LLMClient):
    def __init__(
        self,
        *,
        binary: str,
        cwd: str,
        model: str,
        reasoning_effort: str | None = None,
        sandbox: str = "read-only",
        approval_policy: str = "never",
        resize_edge: int = DEFAULT_RESIZE_EDGE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        min_frames: int = DEFAULT_MIN_FRAMES,
        max_frames: int = DEFAULT_MAX_FRAMES,
        frames_per_sec: float = DEFAULT_FRAMES_PER_SEC,
        global_max_images: int = GLOBAL_MAX_IMAGE_BLOCKS,
        client_factory=None,
    ):
        self.binary = binary
        self.cwd = cwd
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.resize_edge = resize_edge
        self.jpeg_quality = jpeg_quality
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_sec = frames_per_sec
        self.global_max_images = global_max_images
        self.client_factory = client_factory

    async def complete(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        media: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        model_preferences: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        del temperature, top_p, max_tokens, model_preferences, metadata, stop_sequences
        input_items = _build_codex_prompt_items(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            media_inputs=list(media or []),
            resize_edge=self.resize_edge,
            jpeg_quality=self.jpeg_quality,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            frames_per_sec=self.frames_per_sec,
            global_max_images=self.global_max_images,
        )
        if not input_items:
            input_items = [{"type": "text", "text": ""}]
        return await _run_codex_sampling_turn(
            binary=self.binary,
            cwd=self.cwd,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            input_items=input_items,
            sandbox=self.sandbox,
            approval_policy=self.approval_policy,
            client_factory=self.client_factory,
        )


def _sampling_messages_to_input_items(
    system_prompt: str | None,
    messages: list[Any],
    media_inputs: list[Any],
    *,
    resize_edge: int,
    jpeg_quality: int,
    min_frames: int,
    max_frames: int,
    frames_per_sec: float,
    global_max_images: int,
) -> list[dict[str, Any]]:
    transcript_parts: list[str] = []
    if system_prompt:
        transcript_parts.append(f"System: {system_prompt}")

    for message in messages or []:
        role = str(getattr(message, "role", "") or "user").strip().lower()
        text = _extract_text_from_mcp_content(getattr(message, "content", None)).strip()
        if not text:
            continue
        if role == "assistant":
            transcript_parts.append(f"Assistant: {text}")
        else:
            transcript_parts.append(f"User: {text}")

    media_blocks = _build_media_blocks(
        media_inputs,
        resize_edge,
        jpeg_quality,
        min_frames,
        max_frames,
        frames_per_sec,
        global_max_images,
    )
    image_items: list[dict[str, Any]] = []
    for block in media_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = str(block.get("text", "") or "").strip()
            if text:
                transcript_parts.append(text)
            continue
        if block_type == "image_url":
            raw_url = ((block.get("image_url") or {}).get("url") if isinstance(block.get("image_url"), dict) else None)
            if isinstance(raw_url, str) and raw_url:
                image_items.append({"type": "image", "url": raw_url})

    items: list[dict[str, Any]] = []
    transcript = "\n\n".join(part for part in transcript_parts if part).strip()
    if transcript:
        items.append({"type": "text", "text": transcript})
    items.extend(image_items)
    return items


def make_codex_sampling_callback(
    *,
    binary: str,
    cwd: str,
    model: str,
    reasoning_effort: str | None = None,
    sandbox: str = "read-only",
    approval_policy: str = "never",
    resize_edge: int = DEFAULT_RESIZE_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frames_per_sec: float = DEFAULT_FRAMES_PER_SEC,
    global_max_images: int = GLOBAL_MAX_IMAGE_BLOCKS,
    client_factory=None,
):
    async def sampling_callback(context, params: CreateMessageRequestParams) -> CreateMessageResult:
        del context
        system_prompt = getattr(params, "systemPrompt", None) or ""
        messages = list(getattr(params, "messages", []) or [])
        metadata = getattr(params, "metadata", None) or {}
        media_inputs = list(metadata.get("media", []) or [])

        input_items = _sampling_messages_to_input_items(
            system_prompt,
            messages,
            media_inputs,
            resize_edge=resize_edge,
            jpeg_quality=jpeg_quality,
            min_frames=min_frames,
            max_frames=max_frames,
            frames_per_sec=frames_per_sec,
            global_max_images=global_max_images,
        )
        if not input_items:
            input_items = [{"type": "text", "text": ""}]

        try:
            final_text = await _run_codex_sampling_turn(
                binary=binary,
                cwd=cwd,
                model=model,
                reasoning_effort=reasoning_effort,
                input_items=input_items,
                sandbox=sandbox,
                approval_policy=approval_policy,
                client_factory=client_factory,
            )
            return CreateMessageResult(
                content=TextContent(type="text", text=final_text.strip()),
                model=str(model),
                role="assistant",
                stopReason="endTurn",
            )
        except Exception as exc:
            return CreateMessageResult(
                content=TextContent(type="text", text=f"{type(exc).__name__}: {exc}"),
                model=str(model),
                role="assistant",
                stopReason="error",
            )

    return sampling_callback
