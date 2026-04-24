from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from itertools import count
from typing import Any, Optional

logger = logging.getLogger(__name__)
_STREAM_CHUNK_SIZE = 64 * 1024


def _norm_window(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    used_percent = raw.get("usedPercent")
    if used_percent is None:
        return None
    return {
        "used_percent": int(used_percent),
        "window_minutes": raw.get("windowDurationMins", raw.get("windowMinutes")),
        "resets_at": raw.get("resetsAt"),
    }


def _norm_rate_limit_snapshot(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {
        "limit_id": raw.get("limitId"),
        "limit_name": raw.get("limitName"),
        "plan_type": raw.get("planType"),
        "rate_limit_reached_type": raw.get("rateLimitReachedType"),
        "primary": _norm_window(raw.get("primary")),
        "secondary": _norm_window(raw.get("secondary")),
    }
    credits = raw.get("credits")
    if isinstance(credits, dict):
        out["credits"] = {
            "balance": credits.get("balance"),
            "has_credits": bool(credits.get("hasCredits")),
            "unlimited": bool(credits.get("unlimited")),
        }
    return out


def normalize_codex_account_response(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    account = payload.get("account")
    account_out = None
    if isinstance(account, dict) and account.get("type"):
        account_out = {
            "type": str(account.get("type") or ""),
            "email": account.get("email"),
            "plan_type": account.get("planType"),
        }
    return {
        "signed_in": account_out is not None,
        "requires_openai_auth": bool(payload.get("requiresOpenaiAuth")),
        "account": account_out,
    }


def normalize_codex_login_response(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    response_type = str(payload.get("type") or "").strip()
    if response_type == "chatgptDeviceCode":
        return {
            "flow": "device_code",
            "login_id": payload.get("loginId"),
            "verification_url": payload.get("verificationUrl"),
            "user_code": payload.get("userCode"),
        }
    if response_type == "chatgpt":
        return {
            "flow": "browser",
            "login_id": payload.get("loginId"),
            "auth_url": payload.get("authUrl"),
        }
    if response_type == "apiKey":
        return {"flow": "api_key"}
    return {"flow": response_type or "unknown"}


def normalize_codex_rate_limits_response(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    snapshot = _norm_rate_limit_snapshot(payload.get("rateLimits"))
    limits: dict[str, Any] = {}
    raw_limits = payload.get("rateLimitsByLimitId")
    if isinstance(raw_limits, dict):
        for limit_id, limit_payload in raw_limits.items():
            limits[str(limit_id)] = _norm_rate_limit_snapshot(limit_payload)
    return {
        "primary": snapshot.get("primary"),
        "secondary": snapshot.get("secondary"),
        "limit_id": snapshot.get("limit_id"),
        "limit_name": snapshot.get("limit_name"),
        "plan_type": snapshot.get("plan_type"),
        "rate_limit_reached_type": snapshot.get("rate_limit_reached_type"),
        "credits": snapshot.get("credits"),
        "limits": limits,
    }


def normalize_codex_model_list_response(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    items = payload.get("data")
    models: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if bool(item.get("hidden")):
                continue
            models.append(
                {
                    "id": item.get("id") or item.get("model"),
                    "model": item.get("model"),
                    "display_name": item.get("displayName") or item.get("model"),
                    "description": item.get("description") or "",
                    "is_default": bool(item.get("isDefault")),
                    "input_modalities": list(item.get("inputModalities") or []),
                    "reasoning_effort_options": list(item.get("reasoningEffortOptions") or item.get("reasoningEffort") or []),
                    "default_reasoning_effort": item.get("defaultReasoningEffort"),
                    "additional_speed_tiers": list(item.get("additionalSpeedTiers") or []),
                }
            )
    return {"models": models, "next_cursor": payload.get("nextCursor")}


class CodexAppServerError(RuntimeError):
    def __init__(self, message: str, *, code: Any = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class CodexAppServerClient:
    def __init__(
        self,
        *,
        binary: str = "codex",
        cwd: Optional[str] = None,
        client_name: str = "FireRed OpenStoryline",
        client_version: str = "1.0.0",
        experimental_api: bool = True,
    ):
        self.binary = binary
        self.cwd = cwd
        self.client_name = client_name
        self.client_version = client_version
        self.experimental_api = experimental_api
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = count(1)
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._initialized = False
        self.codex_home: Optional[str] = None
        self.notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        async with self._start_lock:
            if self._proc and self._proc.returncode is None:
                if not self._initialized:
                    await self._initialize_prestarted()
                return

            self._proc = await asyncio.create_subprocess_exec(
                self.binary,
                "app-server",
                "--listen",
                "stdio://",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
            )
            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())
            self._initialized = False
            await self._initialize_prestarted()

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._initialized = False
        if proc is None:
            return

        if proc.stdin and not proc.stdin.is_closing():
            with contextlib.suppress(Exception):
                proc.stdin.close()

        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            if proc.returncode is None:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()

        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
                with contextlib.suppress(Exception):
                    await task

        self._reader_task = None
        self._stderr_task = None

        for future in self._pending.values():
            if not future.done():
                future.set_exception(CodexAppServerError("Codex app-server closed"))
        self._pending.clear()

    async def _initialize_prestarted(self) -> None:
        if self._initialized:
            return
        result = await self._call_started(
            "initialize",
            {
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
                "capabilities": (
                    {"experimentalApi": True}
                    if self.experimental_api
                    else {}
                ),
            },
        )
        if isinstance(result, dict):
            home = result.get("codexHome")
            if isinstance(home, str) and home.strip():
                self.codex_home = home.strip()
        await self._notify_started("initialized", {})
        self._initialized = True

    async def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        await self.start()
        await self._notify_started(method, params)

    async def _notify_started(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        async with self._write_lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _call(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        await self.start()
        return await self._call_started(method, params)

    async def _call_started(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        request_id = next(self._ids)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        async with self._write_lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()
        return await future

    async def respond(self, request_id: int, *, result: Optional[dict[str, Any]] = None, error: Optional[dict[str, Any]] = None) -> None:
        await self.start()
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": int(request_id),
        }
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result or {}

        async with self._write_lock:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            await self._proc.stdin.drain()

    async def _iter_stream_lines(self, stream: asyncio.StreamReader):
        buffer = bytearray()
        while True:
            chunk = await stream.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break
                raw_line = bytes(buffer[:newline_index])
                del buffer[: newline_index + 1]
                if raw_line.endswith(b"\r"):
                    raw_line = raw_line[:-1]
                if raw_line:
                    yield raw_line
        if buffer:
            raw_line = bytes(buffer)
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if raw_line:
                yield raw_line

    async def _reader_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            async for line in self._iter_stream_lines(self._proc.stdout):
                try:
                    message = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    logger.warning("Failed to decode Codex app-server stdout line: %r", line[:200])
                    continue
                if "id" in message and ("result" in message or "error" in message):
                    future = self._pending.pop(int(message["id"]), None)
                    if not future or future.done():
                        continue
                    if "error" in message:
                        error = message.get("error") or {}
                        future.set_exception(
                            CodexAppServerError(
                                str(error.get("message") or "Codex app-server error"),
                                code=error.get("code"),
                                data=error.get("data"),
                            )
                        )
                    else:
                        future.set_result(message.get("result"))
                    continue
                if "method" in message:
                    await self.notifications.put(message)
        except asyncio.CancelledError:
            raise
        finally:
            error = CodexAppServerError("Codex app-server stdout closed")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()

    async def _stderr_loop(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for line in self._iter_stream_lines(self._proc.stderr):
                logger.info("codex app-server: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise

    async def next_notification(self, timeout: Optional[float] = None) -> dict[str, Any]:
        if timeout is None:
            return await self.notifications.get()
        return await asyncio.wait_for(self.notifications.get(), timeout=timeout)

    async def account_read(self) -> dict[str, Any]:
        result = await self._call("account/read", {})
        return result if isinstance(result, dict) else {}

    async def account_login_start(self, flow: str) -> dict[str, Any]:
        flow_name = str(flow or "").strip().lower()
        if flow_name == "device_code":
            payload = {"type": "chatgptDeviceCode"}
        elif flow_name == "browser":
            payload = {"type": "chatgpt"}
        else:
            raise ValueError(f"unsupported codex login flow: {flow}")
        result = await self._call("account/login/start", payload)
        return result if isinstance(result, dict) else {}

    async def account_login_cancel(self, login_id: str) -> dict[str, Any]:
        result = await self._call("account/login/cancel", {"loginId": str(login_id or "")})
        return result if isinstance(result, dict) else {}

    async def account_logout(self) -> dict[str, Any]:
        result = await self._call("account/logout", {})
        return result if isinstance(result, dict) else {}

    async def account_rate_limits_read(self) -> dict[str, Any]:
        result = await self._call("account/rateLimits/read", {})
        return result if isinstance(result, dict) else {}

    async def model_list(self) -> dict[str, Any]:
        result = await self._call("model/list", {})
        return result if isinstance(result, dict) else {}

    async def thread_start(
        self,
        *,
        cwd: Optional[str],
        model: Optional[str],
        developer_instructions: Optional[str],
        sandbox: str = "read-only",
        approval_policy: str = "never",
        personality: str = "friendly",
        dynamic_tools: Optional[list[dict[str, Any]]] = None,
        ephemeral: Optional[bool] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": cwd,
            "model": model,
            "developerInstructions": developer_instructions,
            "sandbox": sandbox,
            "approvalPolicy": approval_policy,
            "personality": personality,
        }
        if dynamic_tools:
            params["dynamicTools"] = dynamic_tools
        if ephemeral is not None:
            params["ephemeral"] = bool(ephemeral)
        result = await self._call(
            "thread/start",
            params,
        )
        return result if isinstance(result, dict) else {}

    async def turn_start(
        self,
        *,
        thread_id: str,
        input_items: list[dict[str, Any]],
        model: Optional[str],
        cwd: Optional[str],
        approval_policy: str = "never",
        effort: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": input_items,
            "model": model,
            "cwd": cwd,
            "approvalPolicy": approval_policy,
        }
        if effort:
            params["effort"] = effort
        result = await self._call(
            "turn/start",
            params,
        )
        return result if isinstance(result, dict) else {}
