# agent_fastapi.py
from __future__ import annotations

import asyncio
import errno
import mimetypes
import os
import sys
import json
import re
import time
import uuid
import math
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketState, WebSocketDisconnect
try:
    import tomllib          # Python 3.11+ # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib # Python <= 3.10
import traceback

try:
    from uvicorn.protocols.utils import ClientDisconnected
except Exception:
    ClientDisconnected = None


logger = logging.getLogger(__name__)

import anyio
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage

# ---- 确保 src 可导入（避免环境差异导致找不到模块）----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from open_storyline.agent import build_agent, ClientContext
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.media_handler import scan_media_dir
from open_storyline.config import load_settings, default_config_path
from open_storyline.config import Settings
from open_storyline.storage.agent_memory import ArtifactStore
from open_storyline.mcp.hooks.node_interceptors import ToolInterceptor
from open_storyline.mcp.hooks.chat_middleware import set_mcp_log_sink, reset_mcp_log_sink

WEB_DIR = os.path.join(ROOT_DIR, "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")
NODE_MAP_HTML = os.path.join(WEB_DIR, "node_map/node_map.html")
NODE_MAP_DIR = os.path.join(WEB_DIR, "node_map")

SERVER_CACHE_DIR = os.path.join(ROOT_DIR, '.storyline' , ".server_cache")

CHUNK_SIZE = 1024 * 1024  # 1MB

# 是否根据session_id隔离用户
USE_SESSION_SUBDIR = True

CUSTOM_MODEL_KEY = "__custom__"
SESSION_STATE_FILENAME = "session_state.json"
SESSION_STATE_MAX_HISTORY = 2000
SESSION_STATE_MAX_LC_MESSAGES = 4000
UPLOAD_STATUS_SYSTEM_PREFIX = "【User media upload status】"
UPLOAD_STATUS_SYSTEM_EMPTY = "【User media upload status】{}"
MASKED_SECRET = "***"

def debug_traceback_print(cfg: Settings):
    if cfg.developer.developer_mode:
        traceback.print_exc()

def _s(x: Any) -> str:
    return str(x or "").strip()

def _norm_url(u: Any) -> str:
    u = _s(u)
    return u.rstrip("/") if u else ""

MODEL_ENV_KEYS = {
    "llm": {
        "model": "OPENSTORYLINE_LLM_MODEL",
        "base_url": "OPENSTORYLINE_LLM_BASE_URL",
        "api_key": "OPENSTORYLINE_LLM_API_KEY",
    },
    "vlm": {
        "model": "OPENSTORYLINE_VLM_MODEL",
        "base_url": "OPENSTORYLINE_VLM_BASE_URL",
        "api_key": "OPENSTORYLINE_VLM_API_KEY",
    },
}

def _read_model_env(kind: str) -> Dict[str, str]:
    keys = MODEL_ENV_KEYS[kind]
    return {
        "model": os.getenv(keys.get("model", "")) or "",
        "base_url": _norm_url(os.getenv(keys.get("base_url", ""))),
        "api_key": os.getenv(keys.get("api_key", "")) or "",
    }

def _resolve_builtin_model_override(kind: str, cfg_block: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Default model parsing rules:
    1. Prioritize using `[llm]/[vlm]` from `config.toml`
    2. If any of `model`, `base_url`, or `api_key` is missing, the entire set will fall back to environment variables.
    3. Runtime parameters such as `timeout`, `temperature`, and `max_retries` are still read from the config block.
    """
    kind = kind.strip().lower()
    if kind not in ("llm", "vlm"):
        return None, f"unknown model kind: {kind}"

    model = _s(getattr(cfg_block, "model", None))
    base_url = _norm_url(getattr(cfg_block, "base_url", None))
    api_key = _s(getattr(cfg_block, "api_key", None))

    config_complete = bool(model and base_url and api_key)

    if not config_complete:
        env_cfg = _read_model_env(kind)
        model = env_cfg["model"]
        base_url = env_cfg["base_url"]
        api_key = env_cfg["api_key"]

    if not (model and base_url and api_key):
        keys = MODEL_ENV_KEYS[kind]
        return None, (
            f"{kind} config incomplete. "
            f"Please either fill [{kind}].model/base_url/api_key in config.toml, "
            f"or set env vars: "
            f"{keys['model']}, {keys['base_url']}, {keys['api_key']}"
            f"\nand then rerun."
        )

    override: Dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }

    for k in ("timeout", "temperature", "max_retries", "top_p", "max_tokens"):
        v = getattr(cfg_block, k, None)
        if v not in (None, ""):
            override[k] = v

    return override, None

def _peek_builtin_model_name(kind: str, cfg: Settings) -> str:
    # Get the model name displayed on the front end
    cfg_block = cfg.llm if kind == "llm" else cfg.vlm
    resolved, _ = _resolve_builtin_model_override(kind, cfg_block)
    if resolved and resolved.get("model"):
        return _s(resolved["model"])

    return "unknown model"

def _stable_dict_key(d: Optional[Dict[str, Any]]) -> str:
    try:
        return json.dumps(d or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(d or {})

def _parse_service_config(service_cfg: Any) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Any],
    Optional[str]]:
    """
    返回 (custom_llm, custom_vlm, tts_cfg, pexels, err)
    - custom_llm/custom_vlm: {"model","base_url","api_key"} 或 None（允许只传 llm 或只传 vlm）
    - tts_cfg: dict（可能为空）
    """
    if not isinstance(service_cfg, dict):
        return None, None, {}, {}, None

    # ---- custom models ----
    custom_llm = None
    custom_vlm = None
    custom_models = service_cfg.get("custom_models")

    if custom_models is not None:
        if not isinstance(custom_models, dict):
            return None, None, {}, {}, "service_config.custom_models 必须是对象"

        def _pick(m: Any, label: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
            if m is None:
                return None, None
            if not isinstance(m, dict):
                return None, f"service_config.custom_models.{label} 必须是对象"

            model = _s(m.get("model"))
            base_url = _norm_url(m.get("base_url"))
            api_key = _s(m.get("api_key"))

            if not (model and base_url and api_key):
                return None, f"自定义 {label.upper()} 配置不完整：请填写 model/base_url/api_key"
            if not (base_url.startswith("http://") or base_url.startswith("https://")):
                return None, f"自定义 {label.upper()} 的 base_url 必须以 http(s) 开头"
            return {"model": model, "base_url": base_url, "api_key": api_key}, None

        custom_llm, err1 = _pick(custom_models.get("llm"), "llm")
        if err1:
            return None, None, {}, {}, err1

        custom_vlm, err2 = _pick(custom_models.get("vlm"), "vlm")
        if err2:
            return None, None, {}, {}, err2

    # ---- tts ----
    tts_cfg: Dict[str, Any] = {}
    tts = service_cfg.get("tts")
    if isinstance(tts, dict):
        provider = (tts.get("provider") or "").strip().lower()
        if provider:
            provider_block = tts.get(provider)
            tts_cfg = {"provider": provider, provider: provider_block}
    
    # ---- pexels ----
    pexels_cfg: Dict[str, Any] = {}
    search_media = service_cfg.get("search_media")
    if isinstance(search_media, dict):
        # 支持两种格式：
        # 1) {search_media:{pexels:{mode, api_key}}}
        # 2) {search_media:{mode, pexel_api_key}}
        p = search_media.get("pexels") or search_media.get("pexels")
        if isinstance(p, dict):
            mode = _s(p.get("mode")).lower()
            if mode not in ("default", "custom"):
                mode = "default"
            api_key = _s(p.get("api_key") or p.get("pexels_api_key") or p.get("pexels_api_key"))
            pexels_cfg = {"mode": mode, "api_key": api_key}
        else:
            mode = _s(search_media.get("mode") or search_media.get("pexels_mode") or search_media.get("pexels_mode")).lower()
            if mode not in ("default", "custom"):
                mode = "default"
            api_key = _s(search_media.get("pexels_api_key") or search_media.get("pexels_api_key"))
            pexels_cfg = {"mode": mode, "api_key": api_key}

    return custom_llm, custom_vlm, tts_cfg, pexels_cfg, None

def is_developer_mode(cfg: Settings) -> bool:
    try:
        return bool(cfg.developer.developer_mode)
    except Exception:
        return False

def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def _format_exc(e: BaseException) -> str:
    excs = getattr(e, "exceptions", None)
    if excs:
        parts = [f"{type(e).__name__}: {e}"]
        for i, sub in enumerate(excs, 1):
            parts.append(f"\n--- sub-exception #{i}: {type(sub).__name__}: {sub} ---\n")
            parts.extend(traceback.format_exception(type(sub), sub, sub.__traceback__))
        return "".join(parts)
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

def resolve_media_dir(cfg_media_dir: str, session_id: str) -> str:
    root = _abs(cfg_media_dir).rstrip("/\\")
    if not USE_SESSION_SUBDIR:
        return root
    project_dir = os.path.dirname(root)
    leaf = os.path.basename(root)
    return os.path.join(project_dir, session_id, leaf)


def sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "")
    name = name.replace("\x00", "")
    return name or "unnamed"


def detect_media_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return "image"
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    return "unknown"


_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9._-]{16,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]{16,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9._-]{16,}\.?[A-Za-z0-9._-]*"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization)\s*[:=]\s*[\"']?[A-Za-z0-9._\-+/=]{16,}[\"']?"),
]


def _mask_secret_string(text: str) -> str:
    s = str(text or "")
    for pat in _SECRET_VALUE_PATTERNS:
        s = pat.sub(MASKED_SECRET, s)
    return s


def _to_json_safe(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_json_safe(x) for k, x in v.items()}
    try:
        json.dumps(v, ensure_ascii=False)
        return v
    except Exception:
        return str(v)


def _mask_secrets_recursive(v: Any) -> Any:
    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for k, val in v.items():
            key = str(k)
            if _is_secret_field_name(key):
                out[key] = MASKED_SECRET
            else:
                out[key] = _mask_secrets_recursive(val)
        return out
    if isinstance(v, list):
        return [_mask_secrets_recursive(x) for x in v]
    if isinstance(v, tuple):
        return [_mask_secrets_recursive(x) for x in v]
    if isinstance(v, str):
        return _mask_secret_string(v)
    return _to_json_safe(v)


def _mask_tool_history_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(rec or {})
    for key in ("args", "summary", "message"):
        if key in r:
            r[key] = _mask_secrets_recursive(r.get(key))
    return r


def _serialize_lc_message(msg: BaseMessage) -> Dict[str, Any]:
    if isinstance(msg, SystemMessage):
        return {"type": "system", "content": _to_json_safe(msg.content)}
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": _to_json_safe(msg.content)}
    if isinstance(msg, ToolMessage):
        payload = {
            "type": "tool",
            "content": _to_json_safe(msg.content),
            "tool_call_id": str(getattr(msg, "tool_call_id", "") or ""),
        }
        ak = getattr(msg, "additional_kwargs", None)
        if isinstance(ak, dict) and ak:
            payload["additional_kwargs"] = _to_json_safe(ak)
        name = getattr(msg, "name", None)
        if name not in (None, ""):
            payload["name"] = str(name)
        return payload
    if isinstance(msg, AIMessage):
        ak = getattr(msg, "additional_kwargs", None) or {}
        return {
            "type": "ai",
            "content": _to_json_safe(msg.content),
            "additional_kwargs": _to_json_safe(ak),
        }
    return {
        "type": "unknown",
        "content": _to_json_safe(getattr(msg, "content", "")),
    }


def _deserialize_lc_message(data: Dict[str, Any]) -> Optional[BaseMessage]:
    if not isinstance(data, dict):
        return None
    t = str(data.get("type") or "").strip().lower()
    content = data.get("content", "")
    if t == "system":
        return SystemMessage(content=content)
    if t == "human":
        return HumanMessage(content=content)
    if t == "tool":
        tcid = str(data.get("tool_call_id") or "").strip()
        if not tcid:
            return None
        kwargs: Dict[str, Any] = {}
        ak = data.get("additional_kwargs")
        if isinstance(ak, dict):
            kwargs["additional_kwargs"] = ak
        name = data.get("name")
        if name not in (None, ""):
            kwargs["name"] = str(name)
        try:
            return ToolMessage(content=content, tool_call_id=tcid, **kwargs)
        except TypeError:
            # Keep backward compatibility with signatures that do not accept `name`.
            kwargs.pop("name", None)
            return ToolMessage(content=content, tool_call_id=tcid, **kwargs)
    if t == "ai":
        ak = data.get("additional_kwargs")
        if not isinstance(ak, dict):
            ak = {}
        # Some providers (and langchain-core versions) require tool calls to be present
        # on the AIMessage object (not only inside additional_kwargs) in order to
        # produce a valid OpenAI-style messages list. Without this, ToolMessage
        # may become an "orphan" and providers will reject the request.
        tool_calls = ak.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        try:
            return AIMessage(content=content, additional_kwargs=ak, tool_calls=tool_calls)  # type: ignore[arg-type]
        except TypeError:
            return AIMessage(content=content, additional_kwargs=ak)
    return None


def _tool_call_ids_from_ai_message_for_state(m: BaseMessage) -> Set[str]:
    ids: Set[str] = set()
    if not isinstance(m, AIMessage):
        return ids

    tc = getattr(m, "tool_calls", None) or []
    for c in tc:
        _id = None
        if isinstance(c, dict):
            _id = c.get("id") or c.get("tool_call_id")
        else:
            _id = getattr(c, "id", None) or getattr(c, "tool_call_id", None)
        if _id:
            ids.add(str(_id))

    ak = getattr(m, "additional_kwargs", None) or {}
    tc2 = ak.get("tool_calls") or []
    for c in tc2:
        if isinstance(c, dict):
            _id = c.get("id") or c.get("tool_call_id")
            if _id:
                ids.add(str(_id))
    return ids


def _is_valid_session_id_hex(name: str) -> bool:
    if len(name) != 32:
        return False
    try:
        val = uuid.UUID(name)
        return val.hex == name and val.version == 4
    except Exception:
        return False

_MEDIA_RE = re.compile(r"^media_(\d+)", re.IGNORECASE)

def make_media_store_filename(seq: int, ext: str) -> str:
    ext = (ext or "").lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return f"{MEDIA_PREFIX}{seq:0{MEDIA_SEQ_WIDTH}d}{ext}"

def parse_media_seq(filename: str) -> Optional[int]:
    m = _MEDIA_RE.match(os.path.basename(filename or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def safe_save_path_no_overwrite(media_dir: str, filename: str) -> str:
    filename = sanitize_filename(filename)
    stem, ext = os.path.splitext(filename)
    path = os.path.join(media_dir, filename)
    if not os.path.exists(path):
        return path
    i = 2
    while True:
        p2 = os.path.join(media_dir, f"{stem} ({i}){ext}")
        if not os.path.exists(p2):
            return p2
        i += 1


def ensure_thumbs_dir(media_dir: str) -> str:
    d = os.path.join(media_dir, ".thumbs")
    os.makedirs(d, exist_ok=True)
    return d

def ensure_uploads_dir(media_dir: str) -> str:
    d = os.path.join(media_dir, ".uploads")
    os.makedirs(d, exist_ok=True)
    return d

def guess_media_type(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def _is_under_dir(path: str, root: str) -> bool:
    try:
        path = os.path.abspath(path)
        root = os.path.abspath(root)
        return os.path.commonpath([path, root]) == root
    except Exception:
        return False


def video_placeholder_svg_bytes() -> bytes:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop stop-color="#f2f2f2" offset="0"/>
      <stop stop-color="#e6e6e6" offset="1"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="320" height="320" fill="url(#g)"/>
  <rect x="22" y="22" width="276" height="276" rx="22" fill="rgba(0,0,0,0.06)"/>
  <polygon points="140,120 140,200 210,160" fill="rgba(0,0,0,0.55)"/>
</svg>"""
    return svg.encode("utf-8")


def make_image_thumbnail_sync(src_path: str, dst_path: str, max_size: Tuple[int, int] = (320, 320)) -> bool:
    try:
        from PIL import Image
        img = Image.open(src_path).convert("RGB")
        img.thumbnail(max_size)
        img.save(dst_path, format="JPEG", quality=85)
        return True
    except Exception:
        return False

async def make_video_thumbnail_async(
    src_video: str,
    dst_path: str,
    *,
    max_size: Tuple[int, int] = (320, 320),
    seek_sec: float = 0.5,
    timeout_sec: float = 20.0,
) -> bool:
    ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found (PATH/FFMPEG_BIN). skip video thumbnail. src=%s", src_video)
        return False

    src_video = os.path.abspath(src_video)
    dst_path = os.path.abspath(dst_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    tmp_path = dst_path + ".tmp.jpg"

    vf = (
        f"scale={max_size[0]}:{max_size[1]}:force_original_aspect_ratio=decrease"
        f",pad={max_size[0]}:{max_size[1]}:(ow-iw)/2:(oh-ih)/2"
    )

    async def _run(args: list[str]) -> tuple[bool, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            await proc.wait()
            return False, f"timeout after {timeout_sec}s"
        err_text = (err or b"").decode("utf-8", "ignore").strip()
        return (proc.returncode == 0), err_text

    # 两种策略：1) -ss 在 -i 前（快，但有些文件/关键帧会失败）
    #          2) -ss 在 -i 后（慢，但更稳定）
    common_tail = [
        "-an",
        "-frames:v", "1",
        "-vf", vf,
        "-vcodec", "mjpeg",
        "-q:v", "3",
        "-f", "image2",
        tmp_path,
    ]

    attempts = [
        # fast seek
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{seek_sec}", "-i", src_video] + common_tail,
        # accurate seek
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src_video, "-ss", f"{seek_sec}"] + common_tail,
        # fallback：如果 seek 太靠前导致失败，再试试 1s
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1.0", "-i", src_video] + common_tail,
    ]

    last_err: Optional[str] = None
    try:
        for args in attempts:
            ok, err = await _run(args)
            if ok and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, dst_path)
                return True
            last_err = err or last_err
            # 清理无效临时文件，避免下次误判
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        logger.warning("ffmpeg thumbnail failed. src=%s dst=%s err=%s", src_video, dst_path, last_err)
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)

def _rpm_to_rps(rpm: float) -> float:
    return float(rpm) / 60.0


# 是否信任反向代理头（X-Forwarded-For / X-Real-IP）
RATE_LIMIT_TRUST_PROXY_HEADERS = os.environ.get("RATE_LIMIT_TRUST_PROXY_HEADERS", "0") == "1"

@dataclass
class _RateBucket:
    tokens: float
    last_ts: float        # monotonic
    last_seen: float      # monotonic (for TTL cleanup)

class TokenBucketRateLimiter:
    """
    内存令牌桶 + 防爆内存：
    - max_buckets: 限制内部桶表最大条目数（防止海量 IP 导致字典膨胀）
    - evict_batch: 超过上限后每次驱逐多少条（按插入顺序驱逐最早创建的桶）
    """
    def __init__(
        self,
        ttl_sec: int = 900,
        cleanup_interval_sec: int = 60,
        *,
        max_buckets: int = 100000,
        evict_batch: int = 2000,
    ):
        self.ttl_sec = int(ttl_sec)
        self.cleanup_interval_sec = int(cleanup_interval_sec)
        self.max_buckets = int(max(1, max_buckets))
        self.evict_batch = int(max(1, evict_batch))

        self._buckets: Dict[str, _RateBucket] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def allow(
        self,
        key: str,
        *,
        capacity: float,
        refill_rate: float,
        cost: float = 1.0,
    ) -> Tuple[bool, float, float]:
        """
        返回: (allowed, retry_after_sec, remaining_tokens)
        """
        now = time.monotonic()
        capacity = float(max(0.0, capacity))
        refill_rate = float(max(0.0, refill_rate))
        cost = float(max(0.0, cost))

        async with self._lock:
            b = self._buckets.get(key)

            if b is None:
                # 先做一次周期清理
                if now - self._last_cleanup > self.cleanup_interval_sec:
                    self._cleanup_locked(now)
                    self._last_cleanup = now

                # 桶表满了：先清 TTL，再做批量驱逐；仍然满 -> 不再创建新桶，直接拒绝
                if len(self._buckets) >= self.max_buckets:
                    self._cleanup_locked(now)

                if len(self._buckets) >= self.max_buckets:
                    self._evict_locked()

                if len(self._buckets) >= self.max_buckets:
                    # 不存任何新 key，避免内存继续涨
                    # retry_after 给一个很短的值即可（客户端会重试）
                    return False, 1.0, 0.0

                b = _RateBucket(tokens=capacity, last_ts=now, last_seen=now)
                self._buckets[key] = b
            else:
                b.last_seen = now

            # refill
            elapsed = max(0.0, now - b.last_ts)
            if refill_rate > 0:
                b.tokens = min(capacity, b.tokens + elapsed * refill_rate)
            else:
                b.tokens = min(capacity, b.tokens)
            b.last_ts = now

            if b.tokens >= cost:
                b.tokens -= cost
                return True, 0.0, float(max(0.0, b.tokens))

            # not enough
            if refill_rate <= 0:
                retry_after = float(self.ttl_sec)
            else:
                need = cost - b.tokens
                retry_after = need / refill_rate
            return False, float(retry_after), float(max(0.0, b.tokens))

    def _cleanup_locked(self, now: float) -> None:
        ttl = float(self.ttl_sec)
        dead = [k for k, b in self._buckets.items() if (now - b.last_seen) > ttl]
        for k in dead:
            self._buckets.pop(k, None)

    def _evict_locked(self) -> None:
        # 按 dict 插入顺序驱逐最早的一批 bucket（不排序，避免在高压下额外 CPU 开销）
        n = min(self.evict_batch, len(self._buckets))
        for _ in range(n):
            try:
                k = next(iter(self._buckets))
            except StopIteration:
                break
            self._buckets.pop(k, None)

def _headers_to_dict(scope_headers: List[Tuple[bytes, bytes]]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for k, v in scope_headers or []:
        try:
            dk = k.decode("latin1").lower()
            dv = v.decode("latin1")
        except Exception:
            continue
        d[dk] = dv
    return d

def _client_ip_from_http_scope(scope: dict, trust_proxy_headers: bool) -> str:
    headers = _headers_to_dict(scope.get("headers") or [])
    if trust_proxy_headers:
        xff = headers.get("x-forwarded-for")
        if xff:
            # "client, proxy1, proxy2" -> client
            return xff.split(",")[0].strip() or "unknown"
        xri = headers.get("x-real-ip")
        if xri:
            return xri.strip() or "unknown"

    client = scope.get("client")
    if client and isinstance(client, (list, tuple)) and len(client) >= 1:
        return str(client[0] or "unknown")
    return "unknown"

def _client_ip_from_ws(ws: WebSocket, trust_proxy_headers: bool) -> str:
    try:
        if trust_proxy_headers:
            xff = ws.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip() or "unknown"
            xri = ws.headers.get("x-real-ip")
            if xri:
                return xri.strip() or "unknown"
    except Exception:
        pass

    try:
        if ws.client:
            return str(ws.client.host or "unknown")
    except Exception:
        pass

    return "unknown"

# 分片上传（绕开网关对单次请求体/单文件的限制）
UPLOAD_RESUMABLE_CHUNK_BYTES = _env_int("UPLOAD_RESUMABLE_CHUNK_BYTES", 8 * 1024 * 1024)

# 未完成的分片上传状态保留多久（超时自动清理临时文件）
RESUMABLE_UPLOAD_TTL_SEC = _env_int("RESUMABLE_UPLOAD_TTL_SEC", 3600)  # 1 hour

MEDIA_SEQ_WIDTH = 4  # media_0001
MEDIA_PREFIX = "media_"


# -------- 注意：在服务器上，所有用户的ip可能是相同的----

# 每个 IP 的总体请求速率（包括 /static、/api、/ 等）
HTTP_GLOBAL_RPM   = _env_int("RATE_LIMIT_HTTP_GLOBAL_RPM", 3000)
HTTP_GLOBAL_BURST = _env_int("RATE_LIMIT_HTTP_GLOBAL_BURST", 600)

# 创建 session：防止刷 session 导致内存爆
HTTP_CREATE_SESSION_RPM   = _env_int("RATE_LIMIT_CREATE_SESSION_RPM", 3000)
HTTP_CREATE_SESSION_BURST = _env_int("RATE_LIMIT_CREATE_SESSION_BURST", 50)

# 上传素材：最容易被滥用（大文件 + 频率）
HTTP_UPLOAD_MEDIA_RPM   = _env_int("RATE_LIMIT_UPLOAD_MEDIA_RPM", 12000)
HTTP_UPLOAD_MEDIA_BURST = _env_int("RATE_LIMIT_UPLOAD_MEDIA_BURST", 300)

# 上传“成本”换算：content-length 每多少字节算 1 个 token（越大越费 token）
UPLOAD_COST_BYTES = _env_int("RATE_LIMIT_UPLOAD_COST_BYTES", 10 * 1024 * 1024)  # 默认 10MB = 1 token

# 素材个数控制：会话内上线+上传上限
MAX_UPLOAD_FILES_PER_REQUEST = _env_int("MAX_UPLOAD_FILES_PER_REQUEST", 30)          # 单次请求最多文件数
MAX_MEDIA_PER_SESSION = _env_int("MAX_MEDIA_PER_SESSION", 30)                    # 每个 session 总素材上限（pending + 已用）
MAX_PENDING_MEDIA_PER_SESSION = _env_int("MAX_PENDING_MEDIA_PER_SESSION", 30)     # 每个 session pending 素材上限（UI 友好）

HTTP_UPLOAD_MEDIA_COUNT_RPM   = _env_int("RATE_LIMIT_UPLOAD_MEDIA_COUNT_RPM", 50000)
HTTP_UPLOAD_MEDIA_COUNT_BURST = _env_int("RATE_LIMIT_UPLOAD_MEDIA_COUNT_BURST", 1000)

# 下载/缩略图：适中限制（防刷资源）
HTTP_MEDIA_GET_RPM   = _env_int("RATE_LIMIT_MEDIA_GET_RPM", 2400)
HTTP_MEDIA_GET_BURST = _env_int("RATE_LIMIT_MEDIA_GET_BURST", 60)

# 清空会话：避免频繁清空扰动
HTTP_CLEAR_RPM   = _env_int("RATE_LIMIT_CLEAR_SESSION_RPM", 3000)
HTTP_CLEAR_BURST = _env_int("RATE_LIMIT_CLEAR_SESSION_BURST", 50)

# 其它 API 默认：比 global 更细一点（可选）
HTTP_API_RPM   = _env_int("RATE_LIMIT_API_RPM", 2400)
HTTP_API_BURST = _env_int("RATE_LIMIT_API_BURST", 120)

# WebSocket：连接创建频率
WS_CONNECT_RPM   = _env_int("RATE_LIMIT_WS_CONNECT_RPM", 600)
WS_CONNECT_BURST = _env_int("RATE_LIMIT_WS_CONNECT_BURST", 50)

# WebSocket：chat.send（真正触发 LLM 成本）
WS_CHAT_SEND_RPM   = _env_int("RATE_LIMIT_WS_CHAT_SEND_RPM", 300)
WS_CHAT_SEND_BURST = _env_int("RATE_LIMIT_WS_CHAT_SEND_BURST", 20)

# ---- 全局（所有 IP 合并）限流：抵御多 IP 同时访问 ----
HTTP_ALL_RPM   = _env_int("RATE_LIMIT_HTTP_ALL_RPM", 1200)   # 全站 HTTP 总量：1200/min ~= 20 rps
HTTP_ALL_BURST = _env_int("RATE_LIMIT_HTTP_ALL_BURST", 200)

CREATE_SESSION_ALL_RPM   = _env_int("RATE_LIMIT_CREATE_SESSION_ALL_RPM", 120)
CREATE_SESSION_ALL_BURST = _env_int("RATE_LIMIT_CREATE_SESSION_ALL_BURST", 20)

UPLOAD_MEDIA_ALL_RPM   = _env_int("RATE_LIMIT_UPLOAD_MEDIA_ALL_RPM", 6000)
UPLOAD_MEDIA_ALL_BURST = _env_int("RATE_LIMIT_UPLOAD_MEDIA_ALL_BURST", 2000)

# “素材个数”限流：默认复用 upload_media 的 rpm/burst
UPLOAD_MEDIA_COUNT_ALL_RPM    = _env_int("RATE_LIMIT_UPLOAD_MEDIA_COUNT_ALL_RPM", UPLOAD_MEDIA_ALL_RPM)
UPLOAD_MEDIA_COUNT_ALL_BURST  = _env_int("RATE_LIMIT_UPLOAD_MEDIA_COUNT_ALL_BURST", UPLOAD_MEDIA_ALL_BURST)

MEDIA_GET_ALL_RPM   = _env_int("RATE_LIMIT_MEDIA_GET_ALL_RPM", 600)
MEDIA_GET_ALL_BURST = _env_int("RATE_LIMIT_MEDIA_GET_ALL_BURST", 120)

WS_CONNECT_ALL_RPM   = _env_int("RATE_LIMIT_WS_CONNECT_ALL_RPM", 60000)
WS_CONNECT_ALL_BURST = _env_int("RATE_LIMIT_WS_CONNECT_ALL_BURST", 2000)

WS_CHAT_SEND_ALL_RPM   = _env_int("RATE_LIMIT_WS_CHAT_SEND_ALL_RPM", 500)
WS_CHAT_SEND_ALL_BURST = _env_int("RATE_LIMIT_WS_CHAT_SEND_ALL_BURST", 30)

# ---- 全局并发上限：抵御“很多 IP 同时连/同时触发 LLM/同时上传” ----
WS_MAX_CONNECTIONS     = _env_int("RATE_LIMIT_WS_MAX_CONNECTIONS", 500)  # 同时在线 WS 连接数上限
CHAT_MAX_CONCURRENCY   = _env_int("RATE_LIMIT_CHAT_MAX_CONCURRENCY", 80)  # 同时跑的 LLM turn 上限
UPLOAD_MAX_CONCURRENCY = _env_int("RATE_LIMIT_UPLOAD_MAX_CONCURRENCY", 100) # 同时处理上传（含缩略图）上限

WS_CONN_SEM   = asyncio.Semaphore(WS_MAX_CONNECTIONS)
CHAT_TURN_SEM = asyncio.Semaphore(CHAT_MAX_CONCURRENCY)
UPLOAD_SEM    = asyncio.Semaphore(UPLOAD_MAX_CONCURRENCY)

def _global_http_rule_limit(rule_name: str) -> Optional[Tuple[int, int]]:
    if rule_name == "create_session":
        return CREATE_SESSION_ALL_BURST, CREATE_SESSION_ALL_RPM
    if rule_name == "upload_media":
        return UPLOAD_MEDIA_ALL_BURST, UPLOAD_MEDIA_ALL_RPM
    if rule_name == "media_get":
        return MEDIA_GET_ALL_BURST, MEDIA_GET_ALL_RPM
    return None


def _get_content_length(scope: dict) -> Optional[int]:
    try:
        headers = _headers_to_dict(scope.get("headers") or [])
        v = headers.get("content-length")
        if v is None:
            return None
        n = int(v)
        if n < 0:
            return None
        return n
    except Exception:
        return None

def _match_http_rule(method: str, path: str) -> Tuple[str, int, int, float]:
    """
    返回 (rule_name, burst, rpm, cost)
    cost 默认为 1；上传接口会按 content-length 动态计算 cost（在 middleware 内处理）。
    """
    method = (method or "").upper()
    path = path or ""

    # 精确接口优先
    if method == "POST" and path == "/api/sessions":
        return ("create_session", HTTP_CREATE_SESSION_BURST, HTTP_CREATE_SESSION_RPM, 1.0)

    # 上传素材（含分片接口）
    if method == "POST" and path.startswith("/api/sessions/"):
        if path.endswith("/media") or path.endswith("/media/init"):
            return ("upload_media", HTTP_UPLOAD_MEDIA_BURST, HTTP_UPLOAD_MEDIA_RPM, 1.0)
        if "/media/" in path and (path.endswith("/chunk") or path.endswith("/complete") or path.endswith("/cancel")):
            return ("upload_media", HTTP_UPLOAD_MEDIA_BURST, HTTP_UPLOAD_MEDIA_RPM, 1.0)

    if method == "GET" and path.startswith("/api/sessions/") and (path.endswith("/thumb") or path.endswith("/file")):
        return ("media_get", HTTP_MEDIA_GET_BURST, HTTP_MEDIA_GET_RPM, 1.0)

    if method == "POST" and path.startswith("/api/sessions/") and path.endswith("/clear"):
        return ("clear_session", HTTP_CLEAR_BURST, HTTP_CLEAR_RPM, 1.0)

    # 其它 API
    if path.startswith("/api/"):
        return ("api_general", HTTP_API_BURST, HTTP_API_RPM, 1.0)

    # 非 /api 的其他请求：只走 global
    return ("", 0, 0, 1.0)

class HttpRateLimitMiddleware:
    """
    ASGI middleware：对 HTTP 请求做限流（WebSocket 不在这里处理）。
    """
    def __init__(self, app: Any, limiter: TokenBucketRateLimiter, trust_proxy_headers: bool = False):
        self.app = app
        self.limiter = limiter
        self.trust_proxy_headers = bool(trust_proxy_headers)

    async def __call__(self, scope: dict, receive: Any, send: Any):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        ip = _client_ip_from_http_scope(scope, self.trust_proxy_headers)

        # 0) 全局总量桶（所有 IP 合并）
        ok, retry_after, _ = await self.limiter.allow(
            key="http:all",
            capacity=float(HTTP_ALL_BURST),
            refill_rate=_rpm_to_rps(float(HTTP_ALL_RPM)),
            cost=1.0,
        )
        if not ok:
            return await self._reject(send, retry_after)

        # 1) 单 IP 全局桶（防单点）
        ok, retry_after, _ = await self.limiter.allow(
            key=f"http:global:{ip}",
            capacity=float(HTTP_GLOBAL_BURST),
            refill_rate=_rpm_to_rps(float(HTTP_GLOBAL_RPM)),
            cost=1.0,
        )
        if not ok:
            return await self._reject(send, retry_after)

        # 2) 规则桶
        rule_name, burst, rpm, cost = _match_http_rule(method, path)

        # 上传接口：按 content-length 动态增加 cost（越大越费 token）
        if rule_name == "upload_media":
            cl = _get_content_length(scope)
            if cl and cl > 0 and UPLOAD_COST_BYTES > 0:
                cost = max(1.0, float(math.ceil(cl / float(UPLOAD_COST_BYTES))))

        if rule_name:
            # 2.1 规则的“全局桶”（跨 IP）
            g = _global_http_rule_limit(rule_name)
            if g:
                g_burst, g_rpm = g
                okg, rag, _ = await self.limiter.allow(
                    key=f"http:{rule_name}:all",
                    capacity=float(g_burst),
                    refill_rate=_rpm_to_rps(float(g_rpm)),
                    cost=float(cost),
                )
                if not okg:
                    return await self._reject(send, rag)

            # 2.2 规则的“单 IP 桶”
            ok2, retry_after2, _ = await self.limiter.allow(
                key=f"http:{rule_name}:{ip}",
                capacity=float(burst),
                refill_rate=_rpm_to_rps(float(rpm)),
                cost=float(cost),
            )
            if not ok2:
                return await self._reject(send, retry_after2)

        return await self.app(scope, receive, send)


    async def _reject(self, send: Any, retry_after: float):
        ra = int(math.ceil(float(retry_after or 0.0)))
        body = json.dumps(
            {"detail": "Too Many Requests", "retry_after": ra},
            ensure_ascii=False
        ).encode("utf-8")

        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"retry-after", str(ra).encode("ascii")),
        ]

        await send({"type": "http.response.start", "status": 429, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

RATE_LIMITER = TokenBucketRateLimiter(
    ttl_sec=_env_int("RATE_LIMIT_TTL_SEC", 900),  # 默认 15 分钟：多 IP 攻击时更快释放桶表
    cleanup_interval_sec=_env_int("RATE_LIMIT_CLEANUP_INTERVAL_SEC", 60),
    max_buckets=_env_int("RATE_LIMIT_MAX_BUCKETS", 100000),
    evict_batch=_env_int("RATE_LIMIT_EVICT_BATCH", 2000),
)


@dataclass
class MediaMeta:
    id: str
    name: str
    kind: str
    path: str
    thumb_path: Optional[str]
    ts: float

@dataclass
class ResumableUpload:
    upload_id: str
    filename: str              # 素材原名（用于UI展示）
    store_filename: str        # 落盘名 media_0001.mp4
    size: int
    chunk_size: int
    total_chunks: int
    tmp_path: str
    kind: str
    created_ts: float
    last_ts: float
    received: Set[int] = field(default_factory=set)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class MediaStore:
    """
    专注文件系统层：
    - 保存上传文件（async chunk）
    - 生成缩略图（图片：线程；视频：异步子进程）
    - 删除文件（只删 media_dir 下的文件）
    """
    def __init__(self, media_dir: str):
        self.media_dir = os.path.abspath(media_dir)
        os.makedirs(self.media_dir, exist_ok=True)
        self.thumbs_dir = ensure_thumbs_dir(self.media_dir)

    async def save_upload(self, uf: UploadFile, *, store_filename: str, display_name: str) -> MediaMeta:
        media_id = uuid.uuid4().hex[:10]

        display_name = sanitize_filename(display_name or uf.filename or "unnamed")
        store_filename = sanitize_filename(store_filename)

        kind = detect_media_kind(display_name)

        save_path = os.path.join(self.media_dir, store_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            raise HTTPException(status_code=409, detail=f"media filename exists: {store_filename}")

        # async chunk 写盘（不一次性读入内存）
        async with await anyio.open_file(save_path, "wb") as out:
            while True:
                chunk = await uf.read(CHUNK_SIZE)
                if not chunk:
                    break
                await out.write(chunk)

        try:
            await uf.close()
        except Exception:
            pass

        thumb_path: Optional[str] = None
        if kind in ("image", "video"):
            thumb_path = os.path.join(self.thumbs_dir, f"{media_id}.jpg")

            if kind == "image":
                ok = await anyio.to_thread.run_sync(make_image_thumbnail_sync, save_path, thumb_path)
            else:
                ok = await make_video_thumbnail_async(save_path, thumb_path)

            if not ok:
                # 图片缩略图失败 -> 用原图；视频失败 -> 置空（thumb endpoint 返回占位 SVG）
                thumb_path = save_path if kind == "image" else None

        return MediaMeta(
            id=media_id,
            name=os.path.basename(display_name),
            kind=kind,
            path=os.path.abspath(save_path),
            thumb_path=os.path.abspath(thumb_path) if thumb_path else None,
            ts=time.time(),
        )
    
    async def save_from_path(
        self,
        src_path: str,
        *,
        store_filename: str,
        display_name: str,
    ) -> MediaMeta:
        """
        将分片上传产生的临时文件移动到 media_dir 下的最终文件。
        - display_name: UI 展示名（原始文件名）
        - store_filename: 落盘名（media_0001.mp4），用于记录顺序
        """
        media_id = uuid.uuid4().hex[:10]

        display_name = sanitize_filename(display_name or "unnamed")
        store_filename = sanitize_filename(store_filename or "unnamed")

        kind = detect_media_kind(display_name)

        src_path = os.path.abspath(src_path)
        if not os.path.exists(src_path):
            raise HTTPException(status_code=400, detail="upload temp file missing")

        save_path = os.path.abspath(os.path.join(self.media_dir, store_filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            raise HTTPException(status_code=409, detail=f"media already exists: {store_filename}")

        # move tmp -> final
        try:
            os.replace(src_path, save_path)
        except OSError as e:
            logger.exception("Failed to finalize resumable media file")
            if e.errno == errno.ENOSPC:
                detail = "Failed to save media: disk full"
            elif e.errno in (errno.EACCES, errno.EPERM):
                detail = "Failed to save media: permission denied"
            else:
                detail = f"Failed to save media file"
            raise HTTPException(status_code=500, detail=detail)

        thumb_path: Optional[str] = None
        if kind in ("image", "video"):
            thumb_path = os.path.join(self.thumbs_dir, f"{media_id}.jpg")

            if kind == "image":
                ok = await anyio.to_thread.run_sync(make_image_thumbnail_sync, save_path, thumb_path)
            else:
                ok = await make_video_thumbnail_async(save_path, thumb_path)

            if not ok:
                thumb_path = save_path if kind == "image" else None

        return MediaMeta(
            id=media_id,
            name=os.path.basename(display_name),  # ★ UI 显示原文件名
            kind=kind,
            path=os.path.abspath(save_path),      # ★ 磁盘文件名 media_0001.ext
            thumb_path=os.path.abspath(thumb_path) if thumb_path else None,
            ts=time.time(),
        )

    async def delete_files(self, meta: MediaMeta) -> None:
        root = self.media_dir
        for p in {meta.path, meta.thumb_path}:
            if not p:
                continue
            ap = os.path.abspath(p)
            if not _is_under_dir(ap, root):
                continue
            if os.path.isdir(ap):
                continue
            if os.path.exists(ap):
                try:
                    os.remove(ap)
                except Exception:
                    pass


class ChatSession:
    """
    一个 session 的全部状态：
    - agent / lc_messages（LangChain上下文）
    - history（给前端回放）
    - load_media / pending_media（staging）
    - tool trace 索引（支持 tool 事件“就地更新”）
    """
    def __init__(self, session_id: str, cfg: Settings):
        self.session_id = session_id
        self.cfg = cfg
        self.lang = "zh"

        default_llm = _peek_builtin_model_name("llm", self.cfg)
        default_vlm = _peek_builtin_model_name("vlm", self.cfg)

        self.chat_models = [default_llm, CUSTOM_MODEL_KEY]
        self.chat_model_key = default_llm

        self.vlm_models = [default_vlm, CUSTOM_MODEL_KEY]
        self.vlm_model_key = default_vlm

        self.developer_mode = is_developer_mode(cfg)

        self.media_dir = resolve_media_dir(cfg.project.media_dir, session_id)
        self.media_store = MediaStore(self.media_dir)
        # 分片上传临时目录 + in-flight 状态
        self.uploads_dir = ensure_uploads_dir(self.media_dir)
        self.resumable_uploads: Dict[str, ResumableUpload] = {}

        # 直传（multipart 多文件）时的“预占位”，避免并发竞争导致超过上限
        self._direct_upload_reservations = 0

        self.agent: Any = None
        self.node_manager = None
        self.client_context = None
        
        # 锁分离：避免“流式输出”阻塞上传/删除 pending
        self.chat_lock = asyncio.Lock()
        self.media_lock = asyncio.Lock()

        self.sent_media_total: int = 0
        self._attach_stats_msg_idx = 1

        self.lc_messages: List[BaseMessage] = [
            SystemMessage(content=get_prompt("instruction.system", lang=self.lang)),
            SystemMessage(content=UPLOAD_STATUS_SYSTEM_EMPTY),
        ]
        self.history: List[Dict[str, Any]] = []

        self.load_media: Dict[str, MediaMeta] = {}
        self.pending_media_ids: List[str] = []

        self._tool_history_index: Dict[str, int] = {}  # tool_call_id -> history index

        self.cancel_event = asyncio.Event() # 打断信号

        # 服务相关配置
        self.custom_llm_config: Optional[Dict[str, Any]] = None
        self.custom_vlm_config: Optional[Dict[str, Any]] = None
        self.tts_config: Dict[str, Any] = {}
        self._agent_build_key: Optional[Tuple[Any, ...]] = None

        self.pexels_key_mode: str = "default"   # "default" | "custom"
        self.pexels_custom_key: str = ""

        self._media_seq_inited = False
        self._media_seq_next = 1

    @classmethod
    def state_file_path_for(cls, session_id: str, cfg: Settings) -> str:
        return os.path.abspath(os.path.join(str(cfg.project.outputs_dir), session_id, SESSION_STATE_FILENAME))

    def state_file_path(self) -> str:
        return self.state_file_path_for(self.session_id, self.cfg)

    def _serialize_load_media(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for mid, meta in (self.load_media or {}).items():
            out[str(mid)] = {
                "id": str(meta.id),
                "name": str(meta.name),
                "kind": str(meta.kind),
                "path": str(meta.path),
                "thumb_path": str(meta.thumb_path) if meta.thumb_path else None,
                "ts": float(meta.ts),
            }
        return out

    @staticmethod
    def _deserialize_load_media(data: Any) -> Dict[str, MediaMeta]:
        out: Dict[str, MediaMeta] = {}
        if not isinstance(data, dict):
            return out
        for _k, v in data.items():
            if not isinstance(v, dict):
                continue
            try:
                meta = MediaMeta(
                    id=str(v.get("id") or ""),
                    name=str(v.get("name") or ""),
                    kind=str(v.get("kind") or "unknown"),
                    path=str(v.get("path") or ""),
                    thumb_path=(str(v.get("thumb_path")) if v.get("thumb_path") else None),
                    ts=float(v.get("ts") or time.time()),
                )
                if meta.id:
                    out[meta.id] = meta
            except Exception:
                continue
        return out

    @staticmethod
    def _sanitize_service_cfg_for_state(cfg_obj: Any) -> Any:
        if cfg_obj is None:
            return None
        return _mask_secrets_recursive(_to_json_safe(cfg_obj))

    @staticmethod
    def _normalized_lc_messages_for_state(messages: List[BaseMessage], lang: str) -> List[BaseMessage]:
        instruction_sys = get_prompt("instruction.system", lang=lang)
        upload_placeholder = UPLOAD_STATUS_SYSTEM_EMPTY
        msgs: List[BaseMessage] = [m for m in (messages or []) if isinstance(m, BaseMessage)]

        if not msgs:
            return [SystemMessage(content=instruction_sys), SystemMessage(content=upload_placeholder)]

        if not isinstance(msgs[0], SystemMessage) or str(getattr(msgs[0], "content", "")).strip() != str(instruction_sys).strip():
            msgs.insert(0, SystemMessage(content=instruction_sys))

        def _is_upload_stats_msg(m: BaseMessage) -> bool:
            if not isinstance(m, SystemMessage):
                return False
            return str(getattr(m, "content", "")).strip().startswith(UPLOAD_STATUS_SYSTEM_PREFIX)

        if len(msgs) == 1:
            msgs.append(SystemMessage(content=upload_placeholder))
        elif not _is_upload_stats_msg(msgs[1]):
            found_idx = None
            for i in range(2, len(msgs)):
                if _is_upload_stats_msg(msgs[i]):
                    found_idx = i
                    break
            if found_idx is not None:
                m = msgs.pop(found_idx)
                msgs.insert(1, m)
            else:
                msgs.insert(1, SystemMessage(content=upload_placeholder))
        return msgs

    def _persist_history(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in (self.history or []):
            if not isinstance(item, dict):
                continue
            rec = _to_json_safe(item)
            if not isinstance(rec, dict):
                continue
            if str(rec.get("role") or "") == "tool":
                rec = _mask_tool_history_record(rec)
            out.append(rec)
        if SESSION_STATE_MAX_HISTORY > 0 and len(out) > SESSION_STATE_MAX_HISTORY:
            out = out[-SESSION_STATE_MAX_HISTORY:]
        return out

    def _serialize_lc_messages(self) -> List[Dict[str, Any]]:
        msgs = self._normalized_lc_messages_for_state(list(self.lc_messages or []), self.lang)
        if SESSION_STATE_MAX_LC_MESSAGES > 0 and len(msgs) > SESSION_STATE_MAX_LC_MESSAGES:
            msgs = msgs[-SESSION_STATE_MAX_LC_MESSAGES:]
            msgs = self._normalized_lc_messages_for_state(msgs, self.lang)
            if len(msgs) > SESSION_STATE_MAX_LC_MESSAGES:
                if SESSION_STATE_MAX_LC_MESSAGES <= 2:
                    msgs = msgs[:SESSION_STATE_MAX_LC_MESSAGES]
                else:
                    # Keep invariant-critical first 2 system messages, trim tail payload.
                    keep_tail = SESSION_STATE_MAX_LC_MESSAGES - 2
                    msgs = msgs[:2] + msgs[-keep_tail:]

        out: List[Dict[str, Any]] = []
        for m in msgs:
            try:
                out.append(_serialize_lc_message(m))
            except Exception:
                continue
        return out

    def dump_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "lang": self.lang,
            "history": self._persist_history(),
            "chat_model_key": self.chat_model_key,
            "vlm_model_key": self.vlm_model_key,
            "load_media": self._serialize_load_media(),
            "pending_media_ids": [str(x) for x in (self.pending_media_ids or [])],
            "lc_messages_serialized": self._serialize_lc_messages(),
            "pexels_key_mode": self.pexels_key_mode,
            "custom_llm_config": self._sanitize_service_cfg_for_state(self.custom_llm_config),
            "custom_vlm_config": self._sanitize_service_cfg_for_state(self.custom_vlm_config),
            "tts_config": self._sanitize_service_cfg_for_state(self.tts_config),
        }

    def save_state_atomic(self) -> None:
        path = self.state_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        data = self.dump_state()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _ensure_lc_messages_invariants(self) -> None:
        instruction_sys = get_prompt("instruction.system", lang=self.lang)
        upload_placeholder = UPLOAD_STATUS_SYSTEM_EMPTY
        msgs: List[BaseMessage] = [m for m in (self.lc_messages or []) if isinstance(m, BaseMessage)]

        if not msgs:
            msgs = [SystemMessage(content=instruction_sys), SystemMessage(content=upload_placeholder)]
            self.lc_messages = msgs
            self._attach_stats_msg_idx = 1
            return

        if not isinstance(msgs[0], SystemMessage) or str(getattr(msgs[0], "content", "")).strip() != str(instruction_sys).strip():
            msgs.insert(0, SystemMessage(content=instruction_sys))

        def _is_upload_stats_msg(m: BaseMessage) -> bool:
            if not isinstance(m, SystemMessage):
                return False
            return str(getattr(m, "content", "")).strip().startswith(UPLOAD_STATUS_SYSTEM_PREFIX)

        if len(msgs) <= 1:
            msgs.append(SystemMessage(content=upload_placeholder))
        elif not _is_upload_stats_msg(msgs[1]):
            found_idx = None
            for i in range(2, len(msgs)):
                if _is_upload_stats_msg(msgs[i]):
                    found_idx = i
                    break
            if found_idx is not None:
                m = msgs.pop(found_idx)
                msgs.insert(1, m)
            else:
                msgs.insert(1, SystemMessage(content=upload_placeholder))

        self.lc_messages = msgs
        self._attach_stats_msg_idx = 1

    def _close_unfinished_tool_calls_in_lc_messages(self) -> None:
        ai_ids: Set[str] = set()
        result_ids: Set[str] = set()
        for m in (self.lc_messages or []):
            if isinstance(m, AIMessage):
                ai_ids |= _tool_call_ids_from_ai_message_for_state(m)
            elif isinstance(m, ToolMessage):
                tcid = getattr(m, "tool_call_id", None)
                if tcid:
                    result_ids.add(str(tcid))

        pending = sorted(ai_ids - result_ids)
        if not pending:
            return

        cancelled_content = json.dumps({"cancelled": True, "cancelled_by_restart": True}, ensure_ascii=False)
        for tcid in pending:
            self.lc_messages.append(ToolMessage(content=cancelled_content, tool_call_id=str(tcid)))

    def _sanitize_tool_protocol_in_lc_messages(self) -> None:
        """
        Ensure lc_messages won't violate OpenAI tool-calling protocol:
        ToolMessage must correspond to a preceding AIMessage tool_call_id.
        When restore/serialization fails to fully reconstruct tool_calls,
        providers may reject requests with 400. We drop orphan ToolMessages here.
        """
        msgs_in: List[BaseMessage] = [m for m in (self.lc_messages or []) if isinstance(m, BaseMessage)]
        pending_ids: Set[str] = set()
        responded_ids: Set[str] = set()
        msgs_out: List[BaseMessage] = []

        for m in msgs_in:
            if isinstance(m, AIMessage):
                pending_ids |= _tool_call_ids_from_ai_message_for_state(m)
                msgs_out.append(m)
                continue
            if isinstance(m, ToolMessage):
                tcid = str(getattr(m, "tool_call_id", "") or "").strip()
                if tcid and (tcid in pending_ids) and (tcid not in responded_ids):
                    responded_ids.add(tcid)
                    msgs_out.append(m)
                # else: drop orphan/duplicate tool message
                continue
            msgs_out.append(m)

        self.lc_messages = msgs_out

    def _normalize_running_tool_records(self) -> None:
        for rec in (self.history or []):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("role") or "") != "tool":
                continue
            if str(rec.get("state") or "").lower() == "running":
                rec["state"] = "error"
                rec["progress"] = 1.0
                rec["message"] = "Cancelled by restart"
                rec["summary"] = {"cancelled": True, "cancelled_by_restart": True}

    def _rebuild_tool_history_index(self) -> None:
        self._tool_history_index = {}
        for idx, rec in enumerate(self.history or []):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("role") or "") != "tool":
                continue
            tcid = rec.get("tool_call_id")
            if tcid:
                self._tool_history_index[str(tcid)] = idx

    @classmethod
    def load_from_state(cls, session_id: str, cfg: Settings) -> Optional["ChatSession"]:
        path = cls.state_file_path_for(session_id, cfg)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            sid_in_file = str(data.get("session_id") or "").strip()
            if sid_in_file and sid_in_file != session_id:
                return None
        except Exception:
            return None

        sess = cls(session_id=session_id, cfg=cfg)
        sess.lang = str(data.get("lang") or "zh").strip().lower()
        if sess.lang not in ("zh", "en"):
            sess.lang = "zh"

        sess.history = list(data.get("history") or [])
        sess.chat_model_key = str(data.get("chat_model_key") or sess.chat_model_key)
        sess.vlm_model_key = str(data.get("vlm_model_key") or sess.vlm_model_key)
        sess.load_media = cls._deserialize_load_media(data.get("load_media"))

        pending_ids = [str(x) for x in (data.get("pending_media_ids") or [])]
        sess.pending_media_ids = [x for x in pending_ids if x in sess.load_media]

        lc_msgs_raw = data.get("lc_messages_serialized") or []
        lc_msgs: List[BaseMessage] = []
        if isinstance(lc_msgs_raw, list):
            for item in lc_msgs_raw:
                try:
                    msg = _deserialize_lc_message(item)
                    if msg is not None:
                        lc_msgs.append(msg)
                except Exception:
                    continue
        sess.lc_messages = lc_msgs
        sess._ensure_lc_messages_invariants()

        sess.pexels_key_mode = str(data.get("pexels_key_mode") or "default").strip().lower()
        if sess.pexels_key_mode not in ("default", "custom"):
            sess.pexels_key_mode = "default"
        sess.pexels_custom_key = ""

        llm_cfg = data.get("custom_llm_config")
        sess.custom_llm_config = llm_cfg if isinstance(llm_cfg, dict) else None
        vlm_cfg = data.get("custom_vlm_config")
        sess.custom_vlm_config = vlm_cfg if isinstance(vlm_cfg, dict) else None
        tts_cfg = data.get("tts_config")
        sess.tts_config = tts_cfg if isinstance(tts_cfg, dict) else {}

        sess._normalize_running_tool_records()
        sess._sanitize_tool_protocol_in_lc_messages()
        sess._close_unfinished_tool_calls_in_lc_messages()
        sess._rebuild_tool_history_index()
        sess._ensure_lc_messages_invariants()
        return sess

    def _missing_secret_config_fields(self) -> List[str]:
        missing: List[str] = []

        def _is_missing(v: Any) -> bool:
            s = str(v or "").strip()
            return (not s) or s == MASKED_SECRET

        if self.chat_model_key == CUSTOM_MODEL_KEY:
            cfg = self.custom_llm_config if isinstance(self.custom_llm_config, dict) else {}
            if _is_missing(cfg.get("model")) or _is_missing(cfg.get("base_url")) or _is_missing(cfg.get("api_key")):
                missing.append("custom_llm")

        if self.vlm_model_key == CUSTOM_MODEL_KEY:
            cfg = self.custom_vlm_config if isinstance(self.custom_vlm_config, dict) else {}
            if _is_missing(cfg.get("model")) or _is_missing(cfg.get("base_url")) or _is_missing(cfg.get("api_key")):
                missing.append("custom_vlm")

        if (self.pexels_key_mode or "").lower() == "custom":
            if _is_missing(self.pexels_custom_key):
                missing.append("pexels_custom")

        tts_cfg = self.tts_config if isinstance(self.tts_config, dict) else {}
        provider = str(tts_cfg.get("provider") or "").strip().lower()
        if provider:
            block = tts_cfg.get(provider)
            if not isinstance(block, dict):
                missing.append(f"tts:{provider}")
            else:
                has_secret_key = False
                miss_secret = False
                for k, v in block.items():
                    if _is_secret_field_name(str(k)):
                        has_secret_key = True
                        if _is_missing(v):
                            miss_secret = True
                if has_secret_key and miss_secret:
                    missing.append(f"tts:{provider}")

        return sorted(set(missing))

    def _ensure_system_prompt(self) -> None:
        # Keep the instruction system prompt and upload-status placeholder stable.
        self._ensure_lc_messages_invariants()
        sys = (get_prompt("instruction.system", lang=self.lang) or "").strip()
        if not sys:
            return

        for m in self.lc_messages:
            if isinstance(m, SystemMessage) and (getattr(m, "content", "") or "").strip() == sys:
                return

        self.lc_messages.insert(0, SystemMessage(content=sys))

    def _init_media_seq_locked(self) -> None:
        """
        初始化 self._media_seq_next：
        - 允许 clear chat 后继续编号，不覆盖旧文件
        """
        if self._media_seq_inited:
            return

        max_seq = 0

        # 1) 已落盘文件
        try:
            for fn in os.listdir(self.media_dir):
                s = parse_media_seq(fn)
                if s is not None:
                    max_seq = max(max_seq, s)
        except Exception:
            pass

        # 2) 内存里已有 load_media（保险）
        for meta in (self.load_media or {}).values():
            s = parse_media_seq(os.path.basename(meta.path or ""))
            if s is not None:
                max_seq = max(max_seq, s)

        # 3) in-flight resumable（保险）
        for u in (self.resumable_uploads or {}).values():
            s = parse_media_seq(getattr(u, "store_filename", "") or "")
            if s is not None:
                max_seq = max(max_seq, s)

        self._media_seq_next = max_seq + 1
        self._media_seq_inited = True


    def _reserve_store_filenames_locked(self, display_filenames: List[str]) -> List[str]:
        """
        按传入顺序生成一组 store 文件名（media_0001.ext ...）
        注意：这里的“顺序”就是你要固化的上传顺序。
        """
        self._init_media_seq_locked()

        out: List[str] = []
        seq = int(self._media_seq_next)

        for disp in display_filenames:
            disp = sanitize_filename(disp or "unnamed")
            ext = os.path.splitext(disp)[1].lower()

            # 不复用旧号；仅在极端情况下跳过已存在文件（防撞）
            while True:
                store = make_media_store_filename(seq, ext)
                if not os.path.exists(os.path.join(self.media_dir, store)):
                    break
                seq += 1

            out.append(store)
            seq += 1

        self._media_seq_next = seq
        return out


    def apply_service_config(self, service_cfg: Any) -> Tuple[bool, Optional[str]]:
        llm, vlm, tts, pexels, err = _parse_service_config(service_cfg)
        if err:
            return False, err

        if llm is not None:
            self.custom_llm_config = llm
        if vlm is not None:
            self.custom_vlm_config = vlm

        # tts 允许为空；非空才覆盖
        if isinstance(tts, dict) and tts:
            self.tts_config = tts

        # ---- pexels ----
        if isinstance(pexels, dict) and pexels:
            mode = _s(pexels.get("mode")).lower()
            if mode == "custom":
                self.pexels_key_mode = "custom"
                self.pexels_custom_key = _s(pexels.get("api_key"))
            else:
                self.pexels_key_mode = "default"
                self.pexels_custom_key = ""

        return True, None

    async def ensure_agent(self) -> None:
        missing = self._missing_secret_config_fields()
        if missing:
            raise RuntimeError(
                "requires_reconfig: custom llm/vlm/tts/pexels secrets missing after restart; "
                f"missing={','.join(missing)}"
            )

        # 1) resolve LLM override
        if self.chat_model_key == CUSTOM_MODEL_KEY:
            if not isinstance(self.custom_llm_config, dict):
                raise RuntimeError("please fill in model/base_url/api_key of custom LLM")
            llm_override = self.custom_llm_config
        else:
            llm_override, err = _resolve_builtin_model_override("llm", self.cfg.llm)
            if err:
                raise RuntimeError(err)

        # 2) resolve VLM override
        if self.vlm_model_key == CUSTOM_MODEL_KEY:
            if not isinstance(self.custom_vlm_config, dict):
                raise RuntimeError("please fill in model/base_url/api_key of custom VLM")
            vlm_override = self.custom_vlm_config
        else:
            vlm_override, err = _resolve_builtin_model_override("vlm", self.cfg.vlm)
            if err:
                raise RuntimeError(err)

        agent_build_key: Tuple[Any, ...] = (
            "models",
            _stable_dict_key(llm_override),
            _stable_dict_key(vlm_override),
        )

        if self.agent is None or self._agent_build_key != agent_build_key:
            artifact_store = ArtifactStore(self.cfg.project.outputs_dir, session_id=self.session_id)
            self.agent, self.node_manager = await build_agent(
                cfg=self.cfg,
                session_id=self.session_id,
                store=artifact_store,
                tool_interceptors=[
                    ToolInterceptor.inject_media_content_before,
                    ToolInterceptor.save_media_content_after,
                    ToolInterceptor.inject_tts_config,
                    ToolInterceptor.inject_pexels_api_key,
                ],
                llm_override=llm_override,
                vlm_override=vlm_override,
            )
            self._agent_build_key = agent_build_key

        if self.client_context is None:
            self.client_context = ClientContext(
                cfg=self.cfg,
                session_id=self.session_id,
                media_dir=self.media_dir,
                bgm_dir=self.cfg.project.bgm_dir,
                outputs_dir=self.cfg.project.outputs_dir,
                node_manager=self.node_manager,
                chat_model_key=self.chat_model_key,
                vlm_model_key=self.vlm_model_key,
                tts_config=(self.tts_config or None),
                pexels_api_key=None,
                lang=self.lang,
            )
        else:
            self.client_context.chat_model_key = self.chat_model_key
            self.client_context.vlm_model_key = self.vlm_model_key
            self.client_context.tts_config = (self.tts_config or None)
            self.client_context.lang = self.lang

        # ---- resolve pexels_api_key for runtime context ----
        pexels_api_key = ""
        if (self.pexels_key_mode or "").lower() == "custom":
            pexels_api_key = _s(self.pexels_custom_key)
        else:
            pexels_api_key = _get_default_pexels_api_key(self.cfg)  # from config.toml

        self.client_context.pexels_api_key = (pexels_api_key or None)

    # ---- DTO / public mapping ----
    def public_media(self, meta: MediaMeta) -> Dict[str, Any]:
        return {
            "id": meta.id,
            "name": meta.name,
            "kind": meta.kind,
            "thumb_url": f"/api/sessions/{self.session_id}/media/{meta.id}/thumb",
            "file_url": f"/api/sessions/{self.session_id}/media/{meta.id}/file",
        }

    def public_pending_media(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for aid in self.pending_media_ids:
            meta = self.load_media.get(aid)
            if meta:
                out.append(self.public_media(meta))
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "developer_mode": self.developer_mode,
            "pending_media": self.public_pending_media(),
            "history": self.history,
            "turn_running": self.chat_lock.locked(),
            "limits": {
                "max_upload_files_per_request": MAX_UPLOAD_FILES_PER_REQUEST,
                "max_media_per_session": MAX_MEDIA_PER_SESSION,
                "max_pending_media_per_session": MAX_PENDING_MEDIA_PER_SESSION,
                "upload_chunk_bytes": UPLOAD_RESUMABLE_CHUNK_BYTES,
            },
            "stats": {
                "media_count": len(self.load_media),
                "pending_count": len(self.pending_media_ids),
                "inflight_uploads": len(self.resumable_uploads),
            },
            "chat_model_key": self.chat_model_key,
            "chat_models": self.chat_models,
            "llm_model_key": self.chat_model_key,
            "llm_models": self.chat_models,
            "vlm_model_key": self.vlm_model_key,
            "vlm_models": self.vlm_models,
            "lang": self.lang,
        }

    # ---- media operations ----
    def _cleanup_stale_uploads_locked(self, now: Optional[float] = None) -> None:
        now = float(now or time.time())
        ttl = float(RESUMABLE_UPLOAD_TTL_SEC)
        dead = [uid for uid, u in self.resumable_uploads.items() if (now - u.last_ts) > ttl]
        for uid in dead:
            u = self.resumable_uploads.pop(uid, None)
            if not u:
                continue
            try:
                if u.tmp_path and os.path.exists(u.tmp_path):
                    os.remove(u.tmp_path)
            except Exception:
                pass

    def _check_media_caps_locked(self, add: int = 0) -> None:
        add = int(max(0, add))
        total = len(self.load_media) + len(self.resumable_uploads) + int(self._direct_upload_reservations)
        pending = len(self.pending_media_ids) + len(self.resumable_uploads) + int(self._direct_upload_reservations)

        if MAX_MEDIA_PER_SESSION > 0 and (total + add) > MAX_MEDIA_PER_SESSION:
            raise HTTPException(
                status_code=400,
                detail=f"会话素材总数已达上限：{total}/{MAX_MEDIA_PER_SESSION}",
            )

        if MAX_PENDING_MEDIA_PER_SESSION > 0 and (pending + add) > MAX_PENDING_MEDIA_PER_SESSION:
            raise HTTPException(
                status_code=400,
                detail=f"待发送素材数量已达上限：{pending}/{MAX_PENDING_MEDIA_PER_SESSION}",
            )

    async def add_uploads(self, files: List[UploadFile], store_filenames: List[str]) -> List[MediaMeta]:
        if len(store_filenames) != len(files):
            raise HTTPException(status_code=500, detail="store_filenames mismatch")

        metas: List[MediaMeta] = []
        for uf, store_fn in zip(files, store_filenames):
            display_name = sanitize_filename(uf.filename or "unnamed")
            metas.append(await self.media_store.save_upload(
                uf,
                store_filename=store_fn,
                display_name=display_name,
            ))

        async with self.media_lock:
            for m in metas:
                self.load_media[m.id] = m
                self.pending_media_ids.append(m.id)

            self.pending_media_ids.sort(
                key=lambda aid: os.path.basename(self.load_media[aid].path or "")
                if aid in self.load_media else ""
            )

        return metas

    async def delete_pending_media(self, media_id: str) -> None:
        async with self.media_lock:
            if media_id not in self.pending_media_ids:
                raise HTTPException(status_code=400, detail="media is not pending (refuse physical delete)")
            self.pending_media_ids = [x for x in self.pending_media_ids if x != media_id]
            meta = self.load_media.pop(media_id, None)

        if meta:
            await self.media_store.delete_files(meta)

    async def take_pending_media_for_message(self, attachment_ids: Optional[List[str]]) -> List[MediaMeta]:
        async with self.media_lock:
            if attachment_ids:
                pick = [aid for aid in attachment_ids if aid in self.pending_media_ids]
            else:
                pick = list(self.pending_media_ids)

            pick_set = set(pick)
            self.pending_media_ids = [aid for aid in self.pending_media_ids if aid not in pick_set]
            metas = [self.load_media[aid] for aid in pick if aid in self.load_media]
            return metas

    # ---- tool trace handling ----
    def _ensure_tool_record(self, tcid: str, server: str, name: str, args: Any) -> Dict[str, Any]:
        idx = self._tool_history_index.get(tcid)
        if idx is None:
            rec = {
                "id": f"tool_{tcid}",
                "role": "tool",
                "tool_call_id": tcid,
                "server": server,
                "name": name,
                "args": args,
                "state": "running",
                "progress": 0.0,
                "message": "",
                "summary": None,
                "ts": time.time(),
            }
            self.history.append(rec)
            self._tool_history_index[tcid] = len(self.history) - 1
            return rec
        return self.history[idx]

    def apply_tool_event(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        et = raw.get("type")
        tcid = raw.get("tool_call_id")
        if et not in ("tool_start", "tool_progress", "tool_end") or not tcid:
            return None

        server = raw.get("server") or ""
        name = raw.get("name") or ""
        args = raw.get("args") or {}

        rec = self._ensure_tool_record(tcid, server, name, args)

        if et == "tool_start":
            rec.update({
                "server": server,
                "name": name,
                "args": args,
                "state": "running",
                "progress": 0.0,
                "message": "Starting...",
                "summary": None,
            })

        elif et == "tool_progress":
            progress = float(raw.get("progress", 0.0))
            total = raw.get("total")
            if total and float(total) > 0:
                p = progress / float(total)
            else:
                p = progress / 100.0 if progress > 1 else progress
            p = max(0.0, min(1.0, p))
            rec.update({
                "state": "running",
                "progress": p,
                "message": raw.get("message") or "",
            })

        elif et == "tool_end":
            is_error = bool(raw.get("is_error"))

            summary = raw.get("summary")
            try:
                json.dumps(summary, ensure_ascii=False)
            except Exception:
                summary = str(summary) if summary is not None else None
            rec.update({
                "state": "error" if is_error else "complete",
                "progress": 1.0,
                "summary": summary,
                "message": raw.get("message") or rec.get("message") or "",
            })

        return rec


class SessionStore:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self._sessions: Dict[str, ChatSession] = {}

    async def create(self) -> ChatSession:
        sid = uuid.uuid4().hex
        sess = ChatSession(sid, self.cfg)
        async with self._lock:
            self._sessions[sid] = sess
        await self.save_session_state(sess)
        return sess

    async def get(self, sid: str) -> Optional[ChatSession]:
        async with self._lock:
            sess = self._sessions.get(sid)
            if sess is not None:
                return sess

        sid = str(sid or "").strip()
        if not _is_valid_session_id_hex(sid):
            return None

        restored = ChatSession.load_from_state(sid, self.cfg)
        if restored is None:
            return None
        try:
            restored.save_state_atomic()
        except Exception:
            pass

        async with self._lock:
            existing = self._sessions.get(sid)
            if existing is not None:
                return existing
            self._sessions[sid] = restored
            return restored

    async def get_or_404(self, sid: str) -> ChatSession:
        sess = await self.get(sid)
        if not sess:
            raise HTTPException(status_code=404, detail="session not found")
        return sess

    async def save_session_state(self, sess: ChatSession) -> None:
        try:
            sess.save_state_atomic()
        except Exception as e:
            logger.warning("failed to persist session state. sid=%s err=%s", getattr(sess, "session_id", "?"), e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_settings(default_config_path())
    app.state.cfg = cfg
    app.state.developer_mode = is_developer_mode(cfg)
    app.state.sessions = SessionStore(cfg)
    yield


app = FastAPI(title="OpenStoryline Web", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    HttpRateLimitMiddleware,
    limiter=RATE_LIMITER,
    trust_proxy_headers=RATE_LIMIT_TRUST_PROXY_HEADERS,
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if os.path.isdir(NODE_MAP_DIR):
    app.mount("/node_map", StaticFiles(directory=NODE_MAP_DIR), name="node_map")

api = APIRouter(prefix="/api")

def _rate_limit_reject_json(retry_after: float) -> JSONResponse:
    ra = int(math.ceil(float(retry_after or 0.0)))
    return JSONResponse(
        {"detail": "Too Many Requests", "retry_after": ra},
        status_code=429,
        headers={"Retry-After": str(ra)},
    )

async def _enforce_upload_media_count_limit(request: Request, cost: float) -> Optional[JSONResponse]:
    ip = _client_ip_from_http_scope(request.scope, RATE_LIMIT_TRUST_PROXY_HEADERS)
    cost = float(max(0.0, cost))

    ok, ra, _ = await RATE_LIMITER.allow(
        key="http:upload_media_count:all",
        capacity=float(UPLOAD_MEDIA_COUNT_ALL_BURST),
        refill_rate=_rpm_to_rps(float(UPLOAD_MEDIA_COUNT_ALL_RPM)),
        cost=cost,
    )
    if not ok:
        return _rate_limit_reject_json(ra)

    ok2, ra2, _ = await RATE_LIMITER.allow(
        key=f"http:upload_media_count:{ip}",
        capacity=float(HTTP_UPLOAD_MEDIA_COUNT_BURST),
        refill_rate=_rpm_to_rps(float(HTTP_UPLOAD_MEDIA_COUNT_RPM)),
        cost=cost,
    )
    if not ok2:
        return _rate_limit_reject_json(ra2)

    return None

_TTS_UI_SECRET_KEYS = {
    "api_key",
    "api-token",
    "api_token",
    "auth_token",
    "access_token",
    "authorization",
    "token",
    "password",
    "secret",
    "client_secret",
    "refresh_token",
    "x-api-key",
    "apikey",
    "access_key",
    "accesskey",
    "pexels_api_key",
}

def _is_secret_field_name(k: str) -> bool:
    if str(k or "").strip().lower() in _TTS_UI_SECRET_KEYS:
        return True
    return False

def _read_config_toml(path: str) -> dict:
    if tomllib is None:
        return {}
    try:
        p = Path(path)
        with p.open("rb") as f:
            return tomllib.load(f) or {}
    except Exception:
        return {}

def _get_default_pexels_api_key(cfg: Settings) -> str:
    # 1) try Settings.search_media.pexels_api_key
    try:
        search_media = getattr(cfg, "search_media", None)
        pexels_api_key = _s(getattr(search_media, "pexels_api_key", None) if search_media else None)
        if pexels_api_key:
            return pexels_api_key
        else: 
            return ""
    except Exception:
        return ""

def _normalize_field_item(item) -> dict | None:
    """
    item 支持：
    - "uid"
    - { key="uid", label="UID", required=true, secret=false, placeholder="..." }
    """
    if isinstance(item, str):
        key = item.strip()
        if not key:
            return None
        return {
            "key": key,
            "secret": _is_secret_field_name(key),
        }
    return None

def _build_provider_schema(provider: str, label: str | None, fields: list[dict]) -> dict:
    seen = set()
    out = []
    for f in fields:
        k = str(f.get("key") or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append({
            "key": k,
            "label": f.get("label") or k,
            "placeholder": f.get("placeholder") or f.get("label") or k,
            "required": bool(f.get("required", False)),
            "secret": bool(f.get("secret", False)),
        })
    return {"provider": provider, "label": label or provider, "fields": out}

def _build_tts_ui_schema_from_config(config_path: str) -> dict:
    """
    返回：
    {
      "providers": [
        {"provider":"bytedance","label":"字节跳动","fields":[{"key":"uid",...}, ...]},
        ...
      ]
    }
    """
    cfg = _read_config_toml(config_path)
    tts = cfg.get("generate_voiceover", {})

    providers_out: list[dict] = []

    # 格式：[tts.providers.<provider>]
    providers = tts.get("providers")
    if isinstance(providers, dict):
        for provider, provider_cfg in providers.items():
            fields: list[dict] = []  
            label = str(provider_cfg.get("label") or provider_cfg.get("name") or provider)
            for key in provider_cfg.keys():
                f = _normalize_field_item(str(key))
                if f:
                    fields.append(f)

            providers_out.append(_build_provider_schema(provider, label, fields))

    return {"providers": providers_out}

@app.get("/")
async def index():
    if not os.path.exists(INDEX_HTML):
        return Response("index.html not found. Put it under ./web/index.html", media_type="text/plain", status_code=404)
    return FileResponse(INDEX_HTML, media_type="text/html")

@app.get("/node-map")
async def node_map():
    if not os.path.exists(NODE_MAP_HTML):
        return Response(
            "node_map.html not found. Put it under ./web/node_map/node_map.html",
            media_type="text/plain",
            status_code=404,
        )
    return FileResponse(NODE_MAP_HTML, media_type="text/html")

@api.get("/meta/tts")
async def get_tts_ui_schema():
    schema = _build_tts_ui_schema_from_config(default_config_path())
    return JSONResponse(schema)

# -------------------------
# Sessions (REST)
# -------------------------
@api.post("/sessions")
async def create_session():
    store: SessionStore = app.state.sessions
    sess = await store.create()
    return JSONResponse(sess.snapshot())


@api.get("/sessions/{session_id}")
async def get_session(session_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)
    return JSONResponse(sess.snapshot())


@api.post("/sessions/{session_id}/clear")
async def clear_session_chat(session_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)
    async with sess.chat_lock:
        sess.sent_media_total = 0
        sess._attach_stats_msg_idx = 1
        sess.lc_messages = [
            SystemMessage(content=get_prompt("instruction.system", lang=sess.lang)),
            SystemMessage(content=UPLOAD_STATUS_SYSTEM_EMPTY),
        ]
        sess._attach_stats_msg_idx = 1

        sess.history = []
        sess._tool_history_index = {}
        await store.save_session_state(sess)
    return JSONResponse({"ok": True})

@api.post("/sessions/{session_id}/cancel")
async def cancel_session_turn(session_id: str):
    """
    打断当前正在进行的 LLM turn（流式回复/工具调用）。
    - 不清空 history / lc_messages
    - 仅设置 cancel_event，由 WS 侧在流式循环中感知并安全收尾
    """
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)
    sess.cancel_event.set()
    return JSONResponse({"ok": True})

# -------------------------
# media (REST, session-scoped)
# -------------------------
@api.post("/sessions/{session_id}/media")
async def upload_media(session_id: str, request: Request, files: List[UploadFile] = File(...)):
    if not isinstance(files, list) or not files:
        raise HTTPException(status_code=400, detail="no files")

    if MAX_UPLOAD_FILES_PER_REQUEST > 0 and len(files) > MAX_UPLOAD_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"单次上传最多 {MAX_UPLOAD_FILES_PER_REQUEST} 个文件")

    # 按素材个数限流（cost = 文件数）
    rej = await _enforce_upload_media_count_limit(request, cost=float(len(files)))
    if rej:
        return rej

    if UPLOAD_SEM.locked():
        raise HTTPException(status_code=429, detail="上传并发过高，请稍后重试")
    await UPLOAD_SEM.acquire()

    n = len(files)
    try:
        store: SessionStore = app.state.sessions
        sess = await store.get_or_404(session_id)

        # session cap 检查 + 预占位（避免并发竞争）
        async with sess.media_lock:
            sess._cleanup_stale_uploads_locked()
            sess._check_media_caps_locked(add=n)
            sess._direct_upload_reservations += n

            display_names = [sanitize_filename(uf.filename or "unnamed") for uf in files]
            store_filenames = sess._reserve_store_filenames_locked(display_names)

        try:
            metas = await sess.add_uploads(files, store_filenames=store_filenames)

        finally:
            async with sess.media_lock:
                sess._direct_upload_reservations = max(0, sess._direct_upload_reservations - n)

        await store.save_session_state(sess)

        return JSONResponse({
            "media": [sess.public_media(m) for m in metas],
            "pending_media": sess.public_pending_media(),
        })
    finally:
        try:
            UPLOAD_SEM.release()
        except Exception:
            pass

@api.post("/sessions/{session_id}/media/init")
async def init_resumable_media_upload(session_id: str, request: Request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    filename = sanitize_filename((data.get("filename") or data.get("name") or "unnamed"))
    size = int(data.get("size") or 0)
    if size <= 0:
        raise HTTPException(status_code=400, detail="invalid size")

    # 按素材个数限流：init 视为“新增 1 个素材”
    rej = await _enforce_upload_media_count_limit(request, cost=1.0)
    if rej:
        return rej

    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)

    async with sess.media_lock:
        sess._cleanup_stale_uploads_locked()
        sess._check_media_caps_locked(add=1)

        store_filename = sess._reserve_store_filenames_locked([filename])[0]

        upload_id = uuid.uuid4().hex
        chunk_size = int(max(1, UPLOAD_RESUMABLE_CHUNK_BYTES))
        total_chunks = int(math.ceil(size / float(chunk_size)))

        tmp_path = os.path.join(sess.uploads_dir, f"{upload_id}.part")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        try:
            with open(tmp_path, "wb"):
                pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"cannot create temp file: {e}")

        u = ResumableUpload(
            upload_id=upload_id,
            filename=filename,
            store_filename=store_filename,
            size=size,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            tmp_path=os.path.abspath(tmp_path),
            kind=detect_media_kind(filename),
            created_ts=time.time(),
            last_ts=time.time(),
        )
        sess.resumable_uploads[upload_id] = u

    return JSONResponse({
        "upload_id": upload_id,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "filename": filename,
    })


@api.post("/sessions/{session_id}/media/{upload_id}/chunk")
async def upload_resumable_media_chunk(
    session_id: str,
    upload_id: str,
    index: int = Form(...),
    chunk: UploadFile = File(...),
):
    if UPLOAD_SEM.locked():
        raise HTTPException(status_code=429, detail="上传并发过高，请稍后重试")
    await UPLOAD_SEM.acquire()
    try:
        store: SessionStore = app.state.sessions
        sess = await store.get_or_404(session_id)

        async with sess.media_lock:
            sess._cleanup_stale_uploads_locked()
            u = sess.resumable_uploads.get(upload_id)

        if not u:
            raise HTTPException(status_code=404, detail="upload_id not found or expired")

        idx = int(index)
        if idx < 0 or idx >= u.total_chunks:
            raise HTTPException(status_code=400, detail="invalid chunk index")

        # 期望长度（最后一片可能小于 chunk_size）
        expected_len = u.size - idx * u.chunk_size
        if expected_len <= 0:
            raise HTTPException(status_code=400, detail="invalid chunk index")
        expected_len = min(u.chunk_size, expected_len)

        written = 0
        async with u.lock:
            if u.closed:
                raise HTTPException(status_code=400, detail="upload already closed")

            async with await anyio.open_file(u.tmp_path, "r+b") as out:
                await out.seek(idx * u.chunk_size)
                while True:
                    buf = await chunk.read(CHUNK_SIZE)
                    if not buf:
                        break
                    written += len(buf)
                    if written > expected_len:
                        raise HTTPException(status_code=400, detail="chunk too large")
                    await out.write(buf)

            try:
                await chunk.close()
            except Exception:
                pass

            if written != expected_len:
                raise HTTPException(status_code=400, detail=f"chunk size mismatch: {written} != {expected_len}")

            u.received.add(idx)
            u.last_ts = time.time()

        return JSONResponse({
            "ok": True,
            "received_chunks": len(u.received),
            "total_chunks": u.total_chunks,
        })
    finally:
        try:
            UPLOAD_SEM.release()
        except Exception:
            pass


@api.post("/sessions/{session_id}/media/{upload_id}/complete")
async def complete_resumable_media_upload(session_id: str, upload_id: str):
    if UPLOAD_SEM.locked():
        raise HTTPException(status_code=429, detail="上传并发过高，请稍后重试")
    await UPLOAD_SEM.acquire()
    try:
        store: SessionStore = app.state.sessions
        sess = await store.get_or_404(session_id)

        async with sess.media_lock:
            sess._cleanup_stale_uploads_locked()
            u = sess.resumable_uploads.get(upload_id)

        if not u:
            raise HTTPException(status_code=404, detail="upload_id not found or expired")

        # 锁住此 upload，防止 chunk 并发写
        async with u.lock:
            u.closed = True
            if len(u.received) != u.total_chunks:
                missing = u.total_chunks - len(u.received)
                raise HTTPException(status_code=400, detail=f"chunks missing: {missing}")

        # 从索引移除（释放会话额度）
        async with sess.media_lock:
            u2 = sess.resumable_uploads.pop(upload_id, None)

        if not u2:
            raise HTTPException(status_code=404, detail="upload_id not found")

        meta = await sess.media_store.save_from_path(
            u2.tmp_path,
            store_filename=u2.store_filename,
            display_name=u2.filename,
        )

        async with sess.media_lock:
            sess.load_media[meta.id] = meta
            sess.pending_media_ids.append(meta.id)

        await store.save_session_state(sess)

        return JSONResponse({
            "media": sess.public_media(meta),
            "pending_media": sess.public_pending_media(),
        })
    finally:
        try:
            UPLOAD_SEM.release()
        except Exception:
            pass


@api.post("/sessions/{session_id}/media/{upload_id}/cancel")
async def cancel_resumable_media_upload(session_id: str, upload_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)

    async with sess.media_lock:
        u = sess.resumable_uploads.pop(upload_id, None)

    if not u:
        return JSONResponse({"ok": True})

    async with u.lock:
        u.closed = True
        try:
            if u.tmp_path and os.path.exists(u.tmp_path):
                os.remove(u.tmp_path)
        except Exception:
            pass

    return JSONResponse({"ok": True})

@api.get("/sessions/{session_id}/media/pending")
async def get_pending_media(session_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)
    return JSONResponse({"pending_media": sess.public_pending_media()})


@api.delete("/sessions/{session_id}/media/pending/{media_id}")
async def delete_pending_media(session_id: str, media_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)
    await sess.delete_pending_media(media_id)
    await store.save_session_state(sess)
    return JSONResponse({"ok": True, "pending_media": sess.public_pending_media()})


@api.get("/sessions/{session_id}/media/{media_id}/thumb")
async def get_media_thumb(session_id: str, media_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)

    meta = sess.load_media.get(media_id)
    if not meta:
        raise HTTPException(status_code=404, detail="media not found")

    # thumb 存在优先
    if meta.thumb_path and os.path.exists(meta.thumb_path):
        return FileResponse(meta.thumb_path, media_type="image/jpeg")

    # video 无 thumb => placeholder
    if meta.kind == "video":
        return Response(content=video_placeholder_svg_bytes(), media_type="image/svg+xml")

    # image thumb 失败 => 用原图
    if meta.path and os.path.exists(meta.path):
        return FileResponse(meta.path, media_type=guess_media_type(meta.path))

    raise HTTPException(status_code=404, detail="thumb not available")


@api.get("/sessions/{session_id}/media/{media_id}/file")
async def get_media_file(session_id: str, media_id: str):
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)

    meta = sess.load_media.get(media_id)
    if not meta:
        raise HTTPException(status_code=404, detail="media not found")
    if not meta.path or (not os.path.exists(meta.path)):
        raise HTTPException(status_code=404, detail="file not found")

    # 安全：只允许 media_dir 下
    if not _is_under_dir(meta.path, sess.media_store.media_dir):
        raise HTTPException(status_code=403, detail="forbidden")

    return FileResponse(
        meta.path,
        media_type=guess_media_type(meta.path),
        filename=meta.name,
    )

@api.get("/sessions/{session_id}/preview")
async def preview_local_file(session_id: str, path: str):
    """
    把 summary.preview_urls 里的“服务器本地路径”安全地转成可访问 URL。
    只允许访问：media_dir / outputs_dir / outputs_dir / bgm_dir / .server_cache 这些根目录下的文件。
    """
    store: SessionStore = app.state.sessions
    sess = await store.get_or_404(session_id)

    p = (path or "").strip()
    if not p:
        raise HTTPException(status_code=400, detail="empty path")
    if "\x00" in p:
        raise HTTPException(status_code=400, detail="bad path")

    # 兼容 file:// 前缀（如果未来有）
    if p.startswith("file://"):
        p = p[len("file://"):]

    # 相对路径：默认相对 ROOT_DIR
    if os.path.isabs(p):
        ap = os.path.abspath(p)
    else:
        ap = os.path.abspath(os.path.join(ROOT_DIR, p))

    allowed_roots = [
        os.path.abspath(sess.media_dir),
        os.path.abspath(app.state.cfg.project.outputs_dir),
        os.path.abspath(app.state.cfg.project.outputs_dir),
        os.path.abspath(app.state.cfg.project.bgm_dir),
        os.path.abspath(SERVER_CACHE_DIR),
    ]

    if not any(_is_under_dir(ap, r) for r in allowed_roots):
        raise HTTPException(status_code=403, detail="forbidden")

    if (not os.path.exists(ap)) or os.path.isdir(ap):
        raise HTTPException(status_code=404, detail="file not found")

    # 对 cache 文件强缓存
    headers = {"Cache-Control": "public, max-age=31536000, immutable"} if _is_under_dir(ap, SERVER_CACHE_DIR) else None

    return FileResponse(
        ap,
        media_type=guess_media_type(ap),
        filename=os.path.basename(ap),
        headers=headers,
    )

app.include_router(api)


# -------------------------
# WebSocket: session-scoped chat stream
# -------------------------
def extract_text_delta(msg_chunk: Any) -> str:
    # 兼容 content_blocks (qwen3 常见)
    blocks = getattr(msg_chunk, "content_blocks", None) or []
    if blocks:
        out = ""
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                out += b.get("text", "")
        return out
    c = getattr(msg_chunk, "content", "")
    return c if isinstance(c, str) else ""


async def ws_send(ws: WebSocket, type_: str, data: Any = None):
    if getattr(ws, "client_state", None) != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_json({"type": type_, "data": data})
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False
    except Exception as e:
        if ClientDisconnected is not None and isinstance(e, ClientDisconnected):
            return False
        logger.exception("ws_send failed: type=%s err=%r", type_, e)
        return False

@asynccontextmanager
async def mcp_sink_context(sink_func):
    token = set_mcp_log_sink(sink_func)
    try:
        yield
    finally:
        reset_mcp_log_sink(token)


@app.websocket("/ws/sessions/{session_id}/chat")
async def ws_chat(ws: WebSocket, session_id: str):
    client_ip = _client_ip_from_ws(ws, RATE_LIMIT_TRUST_PROXY_HEADERS)

    ok, retry_after, _ = await RATE_LIMITER.allow(
        key=f"ws:connect:{client_ip}",
        capacity=float(WS_CONNECT_BURST),
        refill_rate=_rpm_to_rps(float(WS_CONNECT_RPM)),
        cost=1.0,
    )
    if not ok:
        try: 
            await ws.close(code=1013, reason=f"rate_limited, retry after {int(math.ceil(retry_after))}s")
        except Exception:
            debug_traceback_print(app.state.cfg)
            pass
        return
    
    if WS_CONN_SEM.locked():
        try:
            await ws.close(code=1013, reason="Server busy (websocket connections limit)")
        except Exception:
            debug_traceback_print(app.state.cfg)
            pass
        return
    
    await WS_CONN_SEM.acquire()

    try:
        await ws.accept()

        store: SessionStore = app.state.sessions
        sess = await store.get(session_id)
        if not sess:
            await ws.close(code=4404, reason="session not found")
            return
        sess = await store.get_or_404(session_id)

        await ws_send(ws, "session.snapshot", sess.snapshot())

        try:
            while True:
                req = await ws.receive_json()
                if not isinstance(req, dict):
                    continue

                t = req.get("type")
                if t == "ping":
                    await ws_send(ws, "pong", {"ts": time.time()})
                    continue

                if t == "session.set_lang":
                    data = (req.get("data") or {})
                    lang = (data.get("lang") or "").strip().lower()
                    if lang not in ("zh", "en"):
                        lang = "zh"

                    sess.lang = lang
                    if sess.client_context:
                        sess.client_context.lang = lang

                    await store.save_session_state(sess)
                    await ws_send(ws, "session.lang", {"lang": lang})
                    continue

                if t == "chat.clear":
                    async with sess.chat_lock:
                        sess.sent_media_total = 0
                        sess._attach_stats_msg_idx = 1
                        sess.lc_messages = [
                            SystemMessage(content=get_prompt("instruction.system", lang=sess.lang)),
                            SystemMessage(content=UPLOAD_STATUS_SYSTEM_EMPTY),
                        ]
                        sess._attach_stats_msg_idx = 1
                        sess.history = []
                        sess._tool_history_index = {}
                        await store.save_session_state(sess)
                    await ws_send(ws, "chat.cleared", {"ok": True})
                    continue

                if t != "chat.send":
                    await ws_send(ws, "error", {"message": f"unknown type: {t}"})
                    continue

                # ---- WebSocket message rate limit: only limit expensive "chat.send" ----
                if sess.chat_lock.locked():
                    await ws_send(ws, "error", {"message": "上一条消息尚未完成，请稍后再发送"})
                    continue
                
                ok, retry_after, _ = await RATE_LIMITER.allow(
                    key="ws:chat_send:all",
                    capacity=float(WS_CHAT_SEND_ALL_BURST),
                    refill_rate=_rpm_to_rps(float(WS_CHAT_SEND_ALL_RPM)),
                    cost=1.0,
                )
                if not ok:
                    await ws_send(ws, "error", {
                        "message": f"触发全局限流：请 {int(math.ceil(retry_after))} 秒后再试",
                        "retry_after": int(math.ceil(retry_after)),
                    })
                    continue

                ok, retry_after, _ = await RATE_LIMITER.allow(
                    key=f"ws:chat_send:{client_ip}",
                    capacity=float(WS_CHAT_SEND_BURST),
                    refill_rate=_rpm_to_rps(float(WS_CHAT_SEND_RPM)),
                    cost=1.0,
                )
                if not ok:
                    await ws_send(ws, "error", {
                        "message": f"触发限流：请 {int(math.ceil(retry_after))} 秒后再试",
                        "retry_after": int(math.ceil(retry_after)),
                    })
                    continue

                if CHAT_TURN_SEM.locked():
                    await ws_send(ws, "error", {"message": "服务器繁忙（模型并发已满），请稍后再试"})
                    continue

                await CHAT_TURN_SEM.acquire()
                try:
                    # 再次确认（期间有 await，锁状态可能变化）
                    if sess.chat_lock.locked():
                        await ws_send(ws, "error", {"message": "上一条消息尚未完成，请稍后再发送"})
                        continue

                    data = (req.get("data", {}) or {})

                    prompt = data.get("text", "")
                    prompt = (prompt or "").strip()
                    if not prompt:
                        continue

                    requested_llm = data.get("llm_model")
                    requested_vlm = data.get("vlm_model")

                    attachment_ids = data.get("attachment_ids")
                    if not isinstance(attachment_ids, list):
                        attachment_ids = None

                    async with sess.chat_lock:
                        # 新 turn 开始：清掉上一次残留的 cancel 信号
                        sess.cancel_event.clear()
                        # 0.0) 应用 service_config（自定义模型 / TTS）
                        ok_cfg, err_cfg = sess.apply_service_config(data.get("service_config"))
                        if not ok_cfg:
                            await ws_send(ws, "error", {"message": err_cfg or "service_config invalid"})
                            continue

                        # 0) 如果前端传了 model，则更新会话当前对话模型
                        if isinstance(requested_llm, str):
                            m = requested_llm.strip()
                            if m:
                                sess.chat_model_key = m
                                if sess.client_context:
                                    sess.client_context.chat_model_key = m

                        if isinstance(requested_vlm, str):
                            m2 = requested_vlm.strip()
                            if m2:
                                sess.vlm_model_key = m2
                                if sess.client_context:
                                    sess.client_context.vlm_model_key = m2

                        requested_lang = data.get("lang")
                        if isinstance(requested_lang, str):
                            lang = requested_lang.strip().lower()
                            if lang in ("zh", "en"):
                                sess.lang = lang
                        # 0.1) 可能需要重建 agent（比如切换到 __custom__ 或者自定义配置变化）
                        try:
                            await sess.ensure_agent()
                        except Exception as e:
                            await ws_send(ws, "error", {"message": f"{type(e).__name__}: {e}"})
                            logger.error(_format_exc(e))
                            continue

                        sess._ensure_system_prompt()

                        if sess.client_context:
                            sess.client_context.lang = sess.lang

                        # 1) 从 pending 里拿本次要发送的附件
                        attachments = await sess.take_pending_media_for_message(attachment_ids)
                        attachments_public = [sess.public_media(m) for m in attachments]

                        # 统计本轮和累计发送了几个素材
                        turn_attached_count = len(attachments)
                        sess.sent_media_total = int(getattr(sess, "sent_media_total", 0)) + turn_attached_count

                        stats = {
                            "Number of media carried in this message sent by the user": turn_attached_count,
                            "Total number of media sent by the user in all conversations": sess.sent_media_total,
                            "Total number of media in user's media library": scan_media_dir(resolve_media_dir(app.state.cfg.project.media_dir, session_id=session_id)),
                        }

                        idx = int(getattr(sess, "_attach_stats_msg_idx", 1))
                        if len(sess.lc_messages) <= idx:
                            while len(sess.lc_messages) <= idx:
                                sess.lc_messages.append(SystemMessage(content=""))

                        sess.lc_messages[idx] = SystemMessage(
                            content="【User media upload status】The following fields are used to determine the nature of the media provided by the user: \n"
                                    + json.dumps(stats, ensure_ascii=False)
                        )


                        # 2.1 写入 history + lc context
                        user_msg = {
                            "id": uuid.uuid4().hex[:12],
                            "role": "user",
                            "content": prompt,
                            "attachments": attachments_public,
                            "ts": time.time(),
                        }
                        sess.history.append(user_msg)
                        sess.lc_messages.append(HumanMessage(content=prompt))
                        await store.save_session_state(sess)

                        if app.state.cfg.developer.print_context:
                            print("[LLM_CTX]", session_id, sess.lc_messages, flush=True)

                        client_attached = True

                        # 2.2 ack：让前端更新 pending + 插入 user 消息（前端也可本地先插入）
                        if not await ws_send(ws, "chat.user", {
                            "text": prompt,
                            "attachments": attachments_public,
                            "pending_media": sess.public_pending_media(),
                            "llm_model_key": sess.chat_model_key,
                            "vlm_model_key": sess.vlm_model_key,
                        }):
                            client_attached = False

                        # 2.3 建立“单通道事件队列”，确保 ws.send_json 不会并发冲突
                        loop = asyncio.get_running_loop()
                        out_q: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()

                        def sink(ev: Any):
                            # MCP interceptor 可能 emit 非 dict；这里只收 dict
                            if isinstance(ev, dict):
                                loop.call_soon_threadsafe(out_q.put_nowait, ("mcp", ev))

                        new_messages: List[BaseMessage] = []

                        async def pump_agent():
                            nonlocal new_messages
                            try:
                                # Guard: ensure the messages list won't violate tool protocol after restore.
                                # This prevents provider-side 400 like: "Messages with role 'tool' must be a response..."
                                sess._sanitize_tool_protocol_in_lc_messages()
                                stream = sess.agent.astream(
                                    {"messages": sess.lc_messages},
                                    context=sess.client_context,
                                    stream_mode=["messages", "updates"],
                                )
                                async for mode, chunk in stream:
                                    if mode == "messages":
                                        msg_chunk, meta = chunk
                                        if meta.get("langgraph_node") == "model":
                                            delta = extract_text_delta(msg_chunk)
                                            if delta:
                                                await out_q.put(("assistant.delta", delta))

                                    elif mode == "updates":
                                        if isinstance(chunk, dict):
                                            for _step, data in chunk.items():
                                                msgs = (data or {}).get("messages") or []
                                                new_messages.extend(msgs)

                                await out_q.put(("agent.done", None))
                            except asyncio.CancelledError:
                                # 被用户打断 / 连接关闭导致的取消，不属于“真正异常”
                                # 不要发 agent.error；给主循环一个 cancelled 信号即可
                                try:
                                    out_q.put_nowait(("agent.cancelled", None))
                                except Exception:
                                    debug_traceback_print(app.state.cfg)
                                    pass
                                raise  # 让任务保持 cancelled 状态，finally 里 await 时会抛 CancelledError

                            except Exception as e:
                                # 关键：异常也要让主循环“可结束”，否则 UI 卡死
                                await out_q.put(("agent.error", f"{type(e).__name__}: {e}"))
                                logger.error(_format_exc(e))


                        async def emit_turn_event(type_: str, data: Any = None) -> bool:
                            nonlocal client_attached
                            if not client_attached:
                                return False

                            # 前端断开后只停止推送，不中断当前 turn 的执行与状态提交。
                            ok = await ws_send(ws, type_, data)
                            if not ok:
                                client_attached = False
                            return ok

                        # turn 开始（前端可禁用发送按钮/显示占位）
                        await emit_turn_event("assistant.start", {})

                        # 当前 assistant 分段缓冲：用于在 tool_start 到来前“封口”
                        seg_text = ""
                        seg_ts: Optional[float] = None

                        async def flush_segment(send_flush_event: bool):
                            """
                            - send_flush_event=True：告诉前端立刻结束当前 assistant 气泡（不结束整个 turn）
                            - 若 seg_text 有内容：写入 history（用于刷新/回放）
                            """
                            nonlocal seg_text, seg_ts

                            if send_flush_event:
                                await emit_turn_event("assistant.flush", {})

                            text = (seg_text or "").strip()
                            if text:
                                sess.history.append({
                                    "id": uuid.uuid4().hex[:12],
                                    "role": "assistant",
                                    "content": text,
                                    "ts": seg_ts or time.time(),
                                })

                            seg_text = ""
                            seg_ts = None

                        pump_task: Optional[asyncio.Task] = None

                        # helper: 从 AIMessage 提取 tool_call_id（兼容不同 provider 的结构）
                        def _tool_call_ids_from_ai_message(m: BaseMessage) -> set[str]:
                            ids: set[str] = set()

                            tc = getattr(m, "tool_calls", None) or []
                            for c in tc:
                                _id = None
                                if isinstance(c, dict):
                                    _id = c.get("id") or c.get("tool_call_id")
                                else:
                                    _id = getattr(c, "id", None) or getattr(c, "tool_call_id", None)
                                if _id:
                                    ids.add(str(_id))

                            ak = getattr(m, "additional_kwargs", None) or {}
                            tc2 = ak.get("tool_calls") or []
                            for c in tc2:
                                if isinstance(c, dict):
                                    _id = c.get("id") or c.get("tool_call_id")
                                    if _id:
                                        ids.add(str(_id))

                            return ids

                        # helper: new_messages 里有哪些 tool_call_id
                        def _tool_call_ids_in_msgs(msgs: List[BaseMessage]) -> set[str]:
                            ids: set[str] = set()
                            for m in msgs:
                                if isinstance(m, AIMessage):
                                    ids |= _tool_call_ids_from_ai_message(m)
                            return ids

                        # helper: new_messages 里哪些 tool_call_id 已经有 ToolMessage 结果了
                        def _tool_result_ids_in_msgs(msgs: List[BaseMessage]) -> set[str]:
                            ids: set[str] = set()
                            for m in msgs:
                                if isinstance(m, ToolMessage):
                                    tcid = getattr(m, "tool_call_id", None)
                                    if tcid:
                                        ids.add(str(tcid))
                            return ids

                        # helper: 把“已存在的 ToolMessage”强制替换成 cancelled（避免工具其实返回了但用户打断没看到，导致上下文和 UI 不一致）
                        def _force_cancelled_tool_results(msgs: List[BaseMessage], cancel_ids: set[str]) -> List[BaseMessage]:
                            if not cancel_ids:
                                return msgs
                            cancelled_content = json.dumps({"cancelled": True}, ensure_ascii=False)
                            out: List[BaseMessage] = []
                            for m in msgs:
                                if isinstance(m, ToolMessage):
                                    tcid = getattr(m, "tool_call_id", None)
                                    if tcid and str(tcid) in cancel_ids:
                                        out.append(ToolMessage(content=cancelled_content, tool_call_id=str(tcid)))
                                        continue
                                out.append(m)
                            return out

                        def _inject_cancelled_tool_messages(msgs: List[BaseMessage], tool_call_ids: List[str]) -> List[BaseMessage]:
                            if not tool_call_ids:
                                return msgs

                            out = list(msgs)

                            existing = set()
                            for m in out:
                                if isinstance(m, ToolMessage):
                                    tcid = getattr(m, "tool_call_id", None)
                                    if tcid:
                                        existing.add(str(tcid))

                            cancelled_content = json.dumps({"cancelled": True}, ensure_ascii=False)

                            for tcid in tool_call_ids:
                                tcid = str(tcid)
                                if tcid in existing:
                                    continue

                                insert_at = None
                                for i in range(len(out) - 1, -1, -1):
                                    m = out[i]
                                    if isinstance(m, AIMessage) and (tcid in _tool_call_ids_from_ai_message(m)):
                                        insert_at = i + 1
                                        break

                                if insert_at is None:
                                    continue

                                out.insert(insert_at, ToolMessage(content=cancelled_content, tool_call_id=tcid))
                                existing.add(tcid)

                            return out

                        def _sanitize_new_messages_on_cancel(
                            new_messages: List[BaseMessage],
                            *,
                            interrupted_text: str,
                            cancelled_tool_ids_from_ui: List[str],
                        ) -> List[BaseMessage]:
                            """
                            返回：应该写回 sess.lc_messages 的消息序列（只包含“用户可见/认可”的那部分）
                            - 工具：对未返回的 tool_call 补 ToolMessage({"cancelled": true})
                            - 回复：用 interrupted_text 替换末尾 final AIMessage，避免把完整回复泄漏进上下文
                            """
                            msgs = list(new_messages or [])
                            interrupted_text = (interrupted_text or "").strip()

                            # 1) 工具：找出“AI 发起了 tool_call 但没有 ToolMessage 结果”的那些 id
                            ai_tool_ids = _tool_call_ids_in_msgs(msgs)
                            tool_result_ids = _tool_result_ids_in_msgs(msgs)
                            pending_tool_ids = ai_tool_ids - tool_result_ids

                            # UI 认为被取消的 tool（running -> cancelled）
                            ui_cancel_ids = {str(x) for x in (cancelled_tool_ids_from_ui or [])}

                            # 统一要取消的集合：
                            # - UI 侧 running 的（用户按下打断时看见的）
                            # - 以及 messages 里缺结果的（防止漏标）
                            cancel_ids = set(ui_cancel_ids) | set(pending_tool_ids)

                            # 2) 如果 new_messages 里已经有 ToolMessage(真实结果) 但用户打断了，
                            #    为了“UI/上下文一致”，强制替换成 cancelled
                            msgs = _force_cancelled_tool_results(msgs, cancel_ids)

                            # 3) 注入缺失的 ToolMessage(cancelled)
                            msgs = _inject_cancelled_tool_messages(msgs, list(cancel_ids))

                            # 4) 处理 assistant 最终文本（避免把完整 answer 写回）
                            #    - 如果 interrupted_text 非空：用它替换最后一个“非 tool_call 的 AIMessage”
                            #    - 如果 interrupted_text 为空：只在“末尾存在一个 non-toolcall AIMessage（且它后面没有 tool_call）”时移除它
                            def _is_toolcall_ai(m: BaseMessage) -> bool:
                                return isinstance(m, AIMessage) and bool(_tool_call_ids_from_ai_message(m))

                            def _is_text_ai(m: BaseMessage) -> bool:
                                if not isinstance(m, AIMessage):
                                    return False
                                if _tool_call_ids_from_ai_message(m):
                                    return False
                                c = getattr(m, "content", None)
                                return isinstance(c, str) and bool(c.strip())

                            # 找最后一个“文本 AIMessage（非 tool_call）”
                            last_text_ai_idx = None
                            for i in range(len(msgs) - 1, -1, -1):
                                if _is_text_ai(msgs[i]):
                                    last_text_ai_idx = i
                                    break

                            if interrupted_text:
                                if last_text_ai_idx is None:
                                    msgs.append(AIMessage(content=interrupted_text))
                                else:
                                    # 用用户看见的部分替换，且丢弃后面所有消息（防止泄漏）
                                    msgs = msgs[:last_text_ai_idx] + [AIMessage(content=interrupted_text)]
                                return msgs

                            # interrupted_text 为空：用户没看见任何本段 token
                            # 只移除“末尾的 final answer AIMessage”，避免把 unseen answer 写进上下文；
                            # 但如果该 AIMessage 后面还有 tool_call（说明它是 pre-tool 文本），就不要删
                            if last_text_ai_idx is not None:
                                has_toolcall_after = any(_is_toolcall_ai(m) for m in msgs[last_text_ai_idx + 1 :])
                                if not has_toolcall_after:
                                    msgs = msgs[:last_text_ai_idx]

                            return msgs

                        pump_task: Optional[asyncio.Task] = None
                        cancel_wait_task: Optional[asyncio.Task] = None

                        was_interrupted = False  # 本 turn 是否已经走了“打断收尾”

                        try:
                            async with mcp_sink_context(sink):
                                pump_task = asyncio.create_task(pump_agent())
                                cancel_wait_task = asyncio.create_task(sess.cancel_event.wait())

                                while True:
                                    # 同时等：queue 出事件 或 cancel_event
                                    get_task = asyncio.create_task(out_q.get())
                                    done, _ = await asyncio.wait(
                                        {get_task, cancel_wait_task},
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )

                                    # 优先处理队列事件（避免 done/flush 已经在队列里时被 cancel 抢占）
                                    if get_task in done:
                                        kind, payload = get_task.result()
                                    else:
                                        # cancel_event 触发：不再等 queue
                                        try:
                                            get_task.cancel()
                                            await get_task
                                        except asyncio.CancelledError:
                                            debug_traceback_print(app.state.cfg)
                                            pass
                                        except Exception:
                                            debug_traceback_print(app.state.cfg)
                                            pass

                                        kind, payload = ("agent.cancelled", None)

                                    # ------------------------
                                    # 1) 处理打断
                                    # ------------------------
                                    if kind == "agent.cancelled":
                                        # 防止重复触发（cancel_event + pump_agent cancelled 都可能来一次）
                                        if was_interrupted:
                                            break
                                        was_interrupted = True
                                        # 1.1 cancel agent 流（停止继续产出 token/工具）
                                        if pump_task and (not pump_task.done()):
                                            pump_task.cancel()

                                        # 1.2 将所有 running 的工具卡片标记为 error
                                        cancelled_tool_recs: List[Dict[str, Any]] = []
                                        for tcid, idx in list(sess._tool_history_index.items()):
                                            rec = sess.history[idx]
                                            if rec.get("role") == "tool" and rec.get("state") == "running":
                                                rec.update({
                                                    "state": "error",
                                                    "progress": 1.0,
                                                    "message": "Cancelled by user",
                                                    "summary": {"cancelled": True},
                                                })
                                                cancelled_tool_recs.append(rec)

                                        # 推送 tool.end，确保前端停止 spinner
                                        for rec in cancelled_tool_recs:
                                            await emit_turn_event("tool.end", {
                                                "tool_call_id": rec["tool_call_id"],
                                                "server": rec["server"],
                                                "name": rec["name"],
                                                "is_error": True,
                                                "summary": rec.get("summary"),
                                            })
                                        # 1.3 把已输出的 seg_text 写入 history（UI 看到的内容）
                                        interrupted_text = (seg_text or "").strip()
                                        if interrupted_text:
                                            sess.history.append({
                                                "id": uuid.uuid4().hex[:12],
                                                "role": "assistant",
                                                "content": interrupted_text,
                                                "ts": seg_ts or time.time(),
                                            })

                                        # 1.4 上下文：只写回“用户真实看到/认可”的消息序列
                                        cancelled_tool_ids = [rec["tool_call_id"] for rec in cancelled_tool_recs]

                                        commit_msgs = _sanitize_new_messages_on_cancel(
                                            new_messages,
                                            interrupted_text=interrupted_text,
                                            cancelled_tool_ids_from_ui=cancelled_tool_ids,
                                        )

                                        if commit_msgs:
                                            sess.lc_messages.extend(commit_msgs)
                                        elif interrupted_text:
                                            # 极端情况：updates 没来得及给任何消息，但用户已看到 token
                                            sess.lc_messages.append(AIMessage(content=interrupted_text))


                                        # ★打断：只发 assistant.end，带 interrupted=true
                                        await emit_turn_event("assistant.end", {"text": interrupted_text, "interrupted": True})

                                        await store.save_session_state(sess)
                                        sess.cancel_event.clear()
                                        break

                                    # ------------------------
                                    # 2) 事件处理
                                    # ------------------------
                                    if kind == "assistant.delta":
                                        delta = payload or ""
                                        if delta:
                                            if seg_ts is None:
                                                seg_ts = time.time()
                                            seg_text += delta
                                            await emit_turn_event("assistant.delta", {"delta": delta})
                                        continue

                                    if kind == "mcp":
                                        raw = payload

                                        if raw.get("type") == "tool_start":
                                            await flush_segment(send_flush_event=True)

                                        rec = sess.apply_tool_event(raw)
                                        if rec:
                                            if raw["type"] == "tool_start":
                                                await emit_turn_event("tool.start", {
                                                    "tool_call_id": rec["tool_call_id"],
                                                    "server": rec["server"],
                                                    "name": rec["name"],
                                                    "args": rec["args"],
                                                })
                                            elif raw["type"] == "tool_progress":
                                                await emit_turn_event("tool.progress", {
                                                    "tool_call_id": rec["tool_call_id"],
                                                    "server": rec["server"],
                                                    "name": rec["name"],
                                                    "progress": rec["progress"],
                                                    "message": rec["message"],
                                                })
                                            elif raw["type"] == "tool_end":
                                                await emit_turn_event("tool.end", {
                                                    "tool_call_id": rec["tool_call_id"],
                                                    "server": rec["server"],
                                                    "name": rec["name"],
                                                    "is_error": rec["state"] == "error",
                                                    "summary": rec["summary"],
                                                })
                                        continue

                                    if kind == "agent.done":
                                        final_text = (seg_text or "").strip()

                                        if final_text:
                                            sess.history.append({
                                                "id": uuid.uuid4().hex[:12],
                                                "role": "assistant",
                                                "content": final_text,
                                                "ts": seg_ts or time.time(),
                                            })

                                        if new_messages:
                                            sess.lc_messages.extend(new_messages)

                                        await emit_turn_event("assistant.end", {"text": final_text})
                                        await store.save_session_state(sess)
                                        break

                                    if kind == "agent.error":
                                        err_text = str(payload or "unknown error")
                                        partial = (seg_text or "").strip()

                                        # 把已输出部分落盘/落上下文（避免丢上下文）
                                        if partial:
                                            sess.history.append({
                                                "id": uuid.uuid4().hex[:12],
                                                "role": "assistant",
                                                "content": partial,
                                                "ts": seg_ts or time.time(),
                                            })
                                            sess.lc_messages.append(AIMessage(content=partial))

                                        if new_messages:
                                            sess.lc_messages.extend(new_messages)

                                        # ★ 真异常：只发 error（并带 partial_text 让前端结束当前气泡）
                                        await emit_turn_event("error", {"message": err_text, "partial_text": partial})
                                        await store.save_session_state(sess)
                                        break
                        
                        except WebSocketDisconnect:
                            return
                        except asyncio.CancelledError:
                            # 连接关闭/任务取消：不当作 error
                            return
                        except Exception as e:
                            # 如果已经走了打断收尾，别再发 error（避免“打断=报错”）
                            if was_interrupted:
                                return
                            await emit_turn_event("error", {"message": f"{type(e).__name__}: {e}", "partial_text": (seg_text or "").strip()})
                            logger.error(_format_exc(e))
                            return
                        finally:
                            # 结束 cancel_wait_task
                            if cancel_wait_task and (not cancel_wait_task.done()):
                                cancel_wait_task.cancel()

                            # pump_task 取消/收尾：避免 await 卡死，加一个短超时保护
                            if pump_task and (not pump_task.done()):
                                pump_task.cancel()
                            if pump_task:
                                try:
                                    await asyncio.wait_for(pump_task, timeout=2.0)
                                except asyncio.TimeoutError:
                                    debug_traceback_print(app.state.cfg)
                                    pass
                                except asyncio.CancelledError:
                                    pass
                                except Exception:
                                    debug_traceback_print(app.state.cfg)
                                    pass

                        # 当前 ws 已断开：turn 已经完整收尾，结束旧 handler，等待前端重连。
                        if not client_attached:
                            return
                finally:
                    try:
                        CHAT_TURN_SEM.release()
                    except Exception:
                        debug_traceback_print(app.state.cfg)
                        pass

        except WebSocketDisconnect:
            return
    finally:
        try:
            WS_CONN_SEM.release()
        except:
            pass
