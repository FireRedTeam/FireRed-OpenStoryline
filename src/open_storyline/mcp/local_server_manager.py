from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import sys
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


class LocalMCPServerManager:
    def __init__(
        self,
        *,
        cwd: str,
        host: str,
        port: int,
        transport: str,
        startup_timeout: float = 20.0,
        poll_interval: float = 0.25,
    ) -> None:
        self.cwd = str(cwd)
        self.host = str(host or "").strip() or "127.0.0.1"
        self.port = int(port)
        self.transport = str(transport or "").strip().lower()
        self.startup_timeout = float(startup_timeout)
        self.poll_interval = float(poll_interval)

        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._recent_logs: deque[str] = deque(maxlen=40)
        self._start_lock = asyncio.Lock()
        self.status = "idle"
        self.last_error = ""

    @property
    def should_manage(self) -> bool:
        return self.transport == "streamable-http" and self.host in _LOCAL_HOSTS

    def _remember_log(self, line: str) -> None:
        text = str(line or "").rstrip()
        if not text:
            return
        self._recent_logs.append(text)

    async def _read_stream(self, stream: asyncio.StreamReader | None, label: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                self._remember_log(f"[{label}] {text}")
                logger.info("local mcp server %s: %s", label, text)
        except asyncio.CancelledError:
            raise

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        parts: list[str] = []
        src_dir = str(Path(self.cwd) / "src")
        parts.append(src_dir)
        parts.append(self.cwd)
        existing = str(env.get("PYTHONPATH") or "").strip()
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            try:
                return sock.connect_ex((self.host, self.port)) == 0
            except OSError:
                return False

    async def _wait_until_ready(self) -> bool:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        while asyncio.get_running_loop().time() < deadline:
            if self._is_port_open():
                self.status = "running"
                self.last_error = ""
                return True
            proc = self._proc
            if proc is not None and proc.returncode is not None:
                self.status = "error"
                self.last_error = self.recent_logs_text or f"local MCP server exited with code {proc.returncode}"
                return False
            await asyncio.sleep(self.poll_interval)
        self.status = "error"
        self.last_error = self.recent_logs_text or "local MCP server did not become ready before timeout"
        return False

    async def ensure_started(self) -> bool:
        if not self.should_manage:
            self.status = "disabled"
            self.last_error = ""
            return False

        async with self._start_lock:
            if self._is_port_open():
                self.status = "external"
                self.last_error = ""
                return True

            if self._proc is None or self._proc.returncode is not None:
                self.status = "starting"
                self.last_error = ""
                self._recent_logs.clear()
                self._proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "open_storyline.mcp.server",
                    cwd=self.cwd,
                    env=self._build_env(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._stdout_task = asyncio.create_task(self._read_stream(self._proc.stdout, "stdout"))
                self._stderr_task = asyncio.create_task(self._read_stream(self._proc.stderr, "stderr"))

            return await self._wait_until_ready()

    async def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            if proc.returncode is None:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        for task in (self._stdout_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(Exception):
                    await task
        self._stdout_task = None
        self._stderr_task = None
        if self.status not in {"external", "disabled"}:
            self.status = "stopped"

    @property
    def recent_logs_text(self) -> str:
        return "\n".join(self._recent_logs).strip()

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "host": self.host,
            "port": self.port,
            "transport": self.transport,
            "should_manage": self.should_manage,
            "last_error": self.last_error,
            "recent_logs": list(self._recent_logs),
        }
