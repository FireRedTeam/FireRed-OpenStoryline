from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Any
import logging

import httpx
from openai import AsyncOpenAI

from langchain_openai import ChatOpenAI

from agents import Agent as OpenAIAgent
from agents.model_settings import ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient

from open_storyline.config import Settings
from open_storyline.storage.agent_memory import ArtifactStore
from open_storyline.nodes.node_manager import NodeManager
from open_storyline.mcp.hooks.chat_middleware import on_progress
from open_storyline.mcp.sampling_handler import make_sampling_callback
from open_storyline.openai_agents_runtime import (
    OpenAIAgentsRuntime,
    build_agents_function_tools,
)
from open_storyline.codex.runtime import CodexRuntime
from open_storyline.codex.sampling_handler import make_codex_sampling_callback
from open_storyline.skills.skills_io import load_skills

logger = logging.getLogger(__name__)


def _walk_exception_tree(exc: BaseException):
    yield exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _walk_exception_tree(sub)
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        yield from _walk_exception_tree(cause)
    context = getattr(exc, "__context__", None)
    if context is not None and context is not exc:
        yield from _walk_exception_tree(context)


def _is_mcp_connection_failure(exc: BaseException) -> bool:
    for item in _walk_exception_tree(exc):
        if isinstance(item, (httpx.ConnectError, ConnectionError, TimeoutError)):
            return True
        message = str(item or "")
        if "All connection attempts failed" in message or "Couldn't connect to server" in message:
            return True
    return False

async def validate_api_key(base_url: str, api_key: str, model: str, provider: str = "LLM", timeout: float = 10.0) -> bool:
    """
    Validate API key by sending a direct HTTP request to the OpenAI-compatible API.
    
    Uses httpx to avoid LangChain parsing errors and provides clear error messages
    based on HTTP status codes.
    """
    # Ensure base_url doesn't end with trailing slash
    base_url = base_url.rstrip("/")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Minimal request body for validation (using max_tokens=1 for minimal cost)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception:
                    raise ValueError(f"{provider} returned non-JSON response. Check base_url/gateway.")
                choices = data.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    print(f"{model} validation successful")
                    return True
                raise ValueError(
                    f"{provider} returned a non-OpenAI-compatible response for chat.completions. "
                    f"Please check gateway behavior, base_url routing, or auth configuration."
                )
            
            # Handle specific HTTP status codes
            if response.status_code in (401, 403):
                logger.error(f"{provider} API key validation failed: {response.status_code} {response.reason_phrase}")
                raise ValueError(
                    f"{provider} API key is invalid or unauthorized. Please check your API key in config.toml or environment variables.\n"
                    f"Model: {model}\n"
                    f"Base URL: {base_url}\n"
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
            elif response.status_code == 404:
                logger.error(f"{provider} API endpoint not found: {response.status_code} {response.reason_phrase}")
                raise ValueError(
                    f"{provider} API endpoint not found. Please check your base_url or model name.\n"
                    f"Model: {model}\n"
                    f"Base URL: {base_url}\n"
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
            elif response.status_code == 429:
                logger.warning(f"{provider} API rate limited: {response.status_code} {response.reason_phrase}")
                raise ConnectionError(
                    f"{provider} API rate limited. Please try again later.\n"
                    f"Model: {model}\n"
                    f"Base URL: {base_url}\n"
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
            elif response.status_code >= 500:
                logger.error(f"{provider} API server error: {response.status_code} {response.reason_phrase}")
                raise ConnectionError(
                    f"{provider} API server error. The service may be temporarily unavailable.\n"
                    f"Model: {model}\n"
                    f"Base URL: {base_url}\n"
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
            else:
                logger.error(f"{provider} API validation failed: {response.status_code} {response.reason_phrase}")
                raise ValueError(
                    f"{provider} API validation failed.\n"
                    f"Model: {model}\n"
                    f"Base URL: {base_url}\n"
                    f"HTTP {response.status_code}: {response.reason_phrase}"
                )
                
    except httpx.TimeoutException as e:
        logger.warning(f"{provider} API connection timeout: {e}")
        raise ConnectionError(
            f"{provider} API connection timeout. Please check your network or base_url.\n"
            f"Model: {model}\n"
            f"Base URL: {base_url}\n"
            f"Error: Connection timed out after {timeout:g} seconds"
        )
    except httpx.ConnectError as e:
        logger.warning(f"{provider} API connection failed: {e}")
        raise ConnectionError(
            f"{provider} API connection failed. Please check your network or base_url.\n"
            f"Model: {model}\n"
            f"Base URL: {base_url}\n"
            f"Error: Unable to connect to the API endpoint"
        )
    except Exception as e:
        logger.error(f"{provider} API validation failed with unexpected error: {e}")
        raise

@dataclass
class ClientContext:
    cfg: Settings
    session_id: str
    media_dir: str
    bgm_dir: str
    outputs_dir: str
    node_manager: NodeManager
    chat_model_key: str  # Chat model key
    vlm_model_key: str = ""  # VLM model key
    pexels_api_key: Optional[str] = None
    tts_config: Optional[dict] = None  # TTS config at runtime
    bgm_config: Optional[dict] = None  # BGM provider config at runtime
    ai_transition_config: Optional[dict] = None # AI transition config at runtime
    llm_pool: dict[tuple[str, bool], ChatOpenAI] = field(default_factory=dict)
    lang: str = "zh" # Default language: Chinese
    codex_model: str = ""
    codex_reasoning_effort: str = ""
    codex_turn_attachments: list[Any] = field(default_factory=list)


async def _build_storyline_tools(
    cfg: Settings,
    session_id: str,
    *,
    sampling_callback,
    tool_interceptors=None,
    allow_mcp_failure: bool = False,
):
    connections = {
        cfg.local_mcp_server.server_name: {
            "transport": cfg.local_mcp_server.server_transport,
            "url": cfg.local_mcp_server.url,
            "timeout": timedelta(seconds=cfg.local_mcp_server.timeout),
            "sse_read_timeout": timedelta(minutes=30),
            "headers": {"X-Storyline-Session-Id": session_id},
            "session_kwargs": {"sampling_callback": sampling_callback},
        },
    }

    client = MultiServerMCPClient(
        connections=connections,
        tool_interceptors=tool_interceptors,
        callbacks=Callbacks(on_progress=on_progress),
        tool_name_prefix=True,
    )

    skills = await load_skills(cfg.skills.skill_dir)
    try:
        tools = await client.get_tools()
    except Exception as exc:
        if not allow_mcp_failure or not _is_mcp_connection_failure(exc):
            raise
        logger.warning(
            "Local MCP server is unavailable for session %s; continuing without node tools.",
            session_id,
            exc_info=exc,
        )
        return [], skills, NodeManager()
    node_manager = NodeManager(tools)
    return tools, skills, node_manager


async def build_agent(
    cfg: Settings,
    session_id: str,
    store: ArtifactStore,
    tool_interceptors=None,
    *,
    llm_override: Optional[dict] = None,
    vlm_override: Optional[dict] = None,
):
    def _get(override: Optional[dict], key: str, default: Any) -> Any:
        return (override.get(key) if isinstance(override, dict) and key in override else default)

    def _norm_url(u: str) -> str:
        u = (u or "").strip()
        return u.rstrip("/") if u else u
    
    # 1) LLM: use user input from form first, fall back to config.toml
    llm_model = _get(llm_override, "model", cfg.llm.model)
    llm_base_url = _norm_url(_get(llm_override, "base_url", cfg.llm.base_url))
    llm_api_key = _get(llm_override, "api_key", cfg.llm.api_key)
    llm_timeout = _get(llm_override, "timeout", cfg.llm.timeout)
    llm_temperature = _get(llm_override, "temperature", cfg.llm.temperature)
    llm_max_retries = _get(llm_override, "max_retries", cfg.llm.max_retries)

    # Validate LLM API key before creating the model
    await validate_api_key(llm_base_url, llm_api_key, llm_model, "LLM", llm_timeout)

    llm = ChatOpenAI(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        default_headers={
            "api-key": llm_api_key,
            "Content-Type": "application/json",
        },
        timeout=llm_timeout,
        temperature=llm_temperature,
        streaming=True,
        max_retries=llm_max_retries,
    )

    # 2) VLM: same priority as above
    vlm_model = _get(vlm_override, "model", cfg.vlm.model)
    vlm_base_url = _norm_url(_get(vlm_override, "base_url", cfg.vlm.base_url))
    vlm_api_key = _get(vlm_override, "api_key", cfg.vlm.api_key)
    vlm_timeout = _get(vlm_override, "timeout", cfg.vlm.timeout)
    vlm_temperature = _get(vlm_override, "temperature", cfg.vlm.temperature)
    vlm_max_retries = _get(vlm_override, "max_retries", cfg.vlm.max_retries)

    # Validate VLM API key before creating the model
    await validate_api_key(vlm_base_url, vlm_api_key, vlm_model, "VLM", vlm_timeout)

    vlm = ChatOpenAI(
        model=vlm_model,
        base_url=vlm_base_url,
        api_key=vlm_api_key,
        default_headers={
            "api-key": vlm_api_key,
            "Content-Type": "application/json",
        },
        timeout=vlm_timeout,
        temperature=vlm_temperature,
        max_retries=vlm_max_retries,
    )

    sampling_callback = make_sampling_callback(llm, vlm)

    tools, skills, node_manager = await _build_storyline_tools(
        cfg,
        session_id,
        sampling_callback=sampling_callback,
        tool_interceptors=tool_interceptors,
        allow_mcp_failure=True,
    )

    model = OpenAIChatCompletionsModel(
        model=llm_model,
        openai_client=AsyncOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            default_headers={
                "api-key": llm_api_key,
                "Content-Type": "application/json",
            },
        ),
    )

    runtime_tools = build_agents_function_tools([*tools, *skills], store)
    agent = OpenAIAgentsRuntime(
        OpenAIAgent(
            name="open_storyline",
            instructions=(
                "Follow the provided conversation history and use the available tools when needed. "
                "Treat the provided system messages as the source of truth for language and behavior."
            ),
            model=model,
            tools=runtime_tools,
            model_settings=ModelSettings(
                temperature=llm_temperature,
                parallel_tool_calls=False,
            ),
        )
    )
    return agent, node_manager


async def build_codex_agent(
    cfg: Settings,
    session_id: str,
    store: ArtifactStore,
    tool_interceptors=None,
    *,
    model: str,
    reasoning_effort: str | None = None,
    binary: str = "codex",
    cwd: str = ".",
    sandbox: str = "danger-full-access",
    approval_policy: str = "never",
):
    sampling_callback = make_codex_sampling_callback(
        binary=binary,
        cwd=cwd,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox=sandbox,
        approval_policy=approval_policy,
    )
    tools, skills, node_manager = await _build_storyline_tools(
        cfg,
        session_id,
        sampling_callback=sampling_callback,
        tool_interceptors=tool_interceptors,
        allow_mcp_failure=True,
    )
    agent = CodexRuntime(
        binary=binary,
        cwd=cwd,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox=sandbox,
        approval_policy=approval_policy,
        langchain_tools=[*tools, *skills],
        store=store,
    )
    return agent, node_manager
