from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Any
import logging

import httpx

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.callbacks import Callbacks
from langchain_mcp_adapters.client import MultiServerMCPClient

from open_storyline.config import Settings
from open_storyline.storage.agent_memory import ArtifactStore
from open_storyline.nodes.node_manager import NodeManager
from open_storyline.mcp.hooks.chat_middleware import handle_tool_errors, on_progress, log_tool_request
from open_storyline.mcp.sampling_handler import make_sampling_callback
from open_storyline.skills.skills_io import load_skills

logger = logging.getLogger(__name__)

async def validate_api_key(base_url: str, api_key: str, model: str, provider: str = "LLM") -> bool:
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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            
            if response.status_code == 200:
                return True
            
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
            f"Error: Connection timed out after 10 seconds"
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
    llm_pool: dict[tuple[str, bool], ChatOpenAI] = field(default_factory=dict)
    lang: str = "zh" # Default language: Chinese


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
    await validate_api_key(llm_base_url, llm_api_key, llm_model, "LLM")

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
    await validate_api_key(vlm_base_url, vlm_api_key, vlm_model, "VLM")

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

    tools = await client.get_tools()
    skills = await load_skills(cfg.skills.skill_dir) # Load skills
    node_manager = NodeManager(tools)

    # 4) Use LangChain's agent runtime to handle the multi-turn tool calling loop
    agent = create_agent(
        model=llm,
        tools=tools+skills,
        middleware=[log_tool_request, handle_tool_errors],
        store=store,
        context_schema=ClientContext,
    )
    return agent, node_manager