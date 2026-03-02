from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional, Any
import logging


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
    try:
        test_llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=10,
            max_retries=0,
        )
        await test_llm.ainvoke("test")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "unauthorized" in error_msg or "invalid" in error_msg or "api key" in error_msg:
            logger.error(f"{provider} API key validation failed: {e}")
            raise ValueError(
                f"{provider} API key is invalid. Please check your API key in config.toml or environment variables.\n"
                f"Model: {model}\n"
                f"Base URL: {base_url}\n"
                f"Error: {e}"
            )
        elif "timeout" in error_msg or "connection" in error_msg:
            logger.warning(f"{provider} API connection failed (network issue): {e}")
            raise ConnectionError(
                f"{provider} API connection failed. Please check your network or base_url.\n"
                f"Model: {model}\n"
                f"Base URL: {base_url}\n"
                f"Error: {e}"
            )
        else:
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