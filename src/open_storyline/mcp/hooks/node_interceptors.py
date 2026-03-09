from collections import defaultdict
from typing import List, Any, Dict
import os
from pathlib import Path
import json
import traceback


from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.types import Command
from langchain_core.messages import ToolMessage, ToolCall
from langchain_core.tools import ToolException
from mcp.types import CallToolResult

from open_storyline.nodes.node_manager import NodeManager

from open_storyline.storage.file import FileCompressor
from open_storyline.utils.logging import get_logger

logger = get_logger(__name__)


# Hosts that indicate Agent and MCP server are on the same machine (path-only, no base64). 0.0.0.0 for Docker.
_LOCAL_CONNECT_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def should_inline_media_as_base64(server_cfg=None) -> bool:
    """
    Whether to inline media as base64 in MCP requests.
    - inline_media "always" -> True (base64); "never" -> False (path-only); "auto" -> by connect_host.
    - In "auto": connect_host in 127.0.0.1/localhost/::1/0.0.0.0 -> False (path-only), else True (base64).
    """
    if server_cfg is None:
        return False
    try:
        mcp = getattr(server_cfg, "local_mcp_server", None)
        if mcp is None:
            return False
        mode = getattr(mcp, "inline_media", "auto")
        if mode == "always":
            return True
        if mode == "never":
            return False
        # auto
        host = (getattr(mcp, "connect_host", None) or "").strip().lower()
        if not host:
            return False
        return host not in _LOCAL_CONNECT_HOSTS
    except Exception:
        return False


def compress_payload_to_base64(payload: Dict[str, List[Any]], server_cfg=None):
    """Convert path-only items to base64 when in remote MCP mode. No-op for local mode."""
    if not isinstance(payload, dict):
        return payload
    if not should_inline_media_as_base64(server_cfg):
        return payload
    for key, value in payload.items():
        if isinstance(value, list) and all([isinstance(item, dict) for item in value]):
            for item in value:
                if 'path' in item.keys():
                    path = item['path']
                    compress_data = FileCompressor.compress_and_encode(path)
                    item.update({
                        "path": path,
                        "base64": compress_data.base64,
                        "md5": compress_data.md5
                    })
        elif isinstance(value, dict):
            compress_payload_to_base64(value, server_cfg)

class ToolInterceptor:
    
    @staticmethod
    async def inject_media_content_before(
        request: MCPToolCallRequest,
        handler,
    ):
        try:
            tool_call_type = request.args.get('tool_call_type', 'auto')
            # for default tool call 
            if tool_call_type!= 'auto':
                request.args = request.args.get('args', {})

            runtime = request.runtime
            context = runtime.context
            store = runtime.store
            session_id = context.session_id
            node_id = request.name
            lang = context.lang
            artifact_id = store.generate_artifact_id(node_id)
            meta_collector: NodeManager = context.node_manager
            input_data = defaultdict(list)

            server_cfg = getattr(context, "cfg", None) or getattr(context, "server_cfg", None)
            inline_base64 = should_inline_media_as_base64(server_cfg)

            def load_collected_data(collected_node, input_data, store):
                """Load collected node data"""
                for collect_kind, artifact_meta in collected_node.items():
                    _, prior_node_output = store.load_result(artifact_meta.artifact_id)
                    compress_payload_to_base64(prior_node_output['payload'], server_cfg)
                    input_data[collect_kind] = prior_node_output['payload']

            if node_id == 'load_media':
                input_data['inputs'] = []
                seen_paths: set = set()
                media_dir = Path(context.media_dir)
                try:
                    project_media_root = Path(server_cfg.project.media_dir).resolve()
                except Exception:
                    project_media_root = None
                for file_name in os.listdir(media_dir):
                    path = media_dir / file_name
                    if path.is_dir():
                        continue
                    if inline_base64:
                        rel_path = str(path.relative_to(os.getcwd()))
                        compress_data = FileCompressor.compress_and_encode(path)
                        input_data['inputs'].append({
                            "path": rel_path,
                            "base64": compress_data.base64,
                            "md5": compress_data.md5,
                        })
                    else:
                        # Path-only (local):
                        # - Prefer a path relative to project.media_dir (server contract) when possible.
                        # - Otherwise, fall back to absolute path (e.g. FastAPI session subdir is outside project.media_dir).
                        abs_path = path.resolve()
                        rel_or_abs: str
                        if project_media_root is not None:
                            try:
                                rel_or_abs = str(abs_path.relative_to(project_media_root))
                            except ValueError:
                                rel_or_abs = str(abs_path)
                        else:
                            rel_or_abs = str(abs_path)

                        if rel_or_abs not in seen_paths:
                            seen_paths.add(rel_or_abs)
                            input_data['inputs'].append({
                                "path": rel_or_abs,
                                "orig_path": rel_or_abs,
                                "orig_md5": None,
                            })
                # Path-only mode: include auto-searched media from .storyline/.server_cache so they are readable
                if not inline_base64:
                    latest_search = store.get_latest_meta(node_id='search_media', session_id=session_id)
                    if latest_search:
                        _, data = store.load_result(latest_search.artifact_id)
                        if isinstance(data, dict):
                            paths = data.get('payload', {}).get('search_media') or []
                            for p in paths:
                                # search_media returns a list of {"path": "..."} dicts (and may
                                # also return list[str] in older versions); support both.
                                if isinstance(p, dict):
                                    p = p.get("path")
                                if not p or not isinstance(p, str):
                                    continue
                                norm = str(Path(p).resolve()) if not os.path.isabs(p) else p
                                if norm in seen_paths:
                                    continue
                                seen_paths.add(norm)
                                input_data['inputs'].append({
                                    "path": p,
                                    "orig_path": p,
                                    "orig_md5": None,
                                })
            elif node_id in list(meta_collector.id_to_tool.keys()):
                # 1. Determine execution mode and dependency requirements
                is_skip_mode = request.args.get('mode', 'auto') != 'auto'
                require_kind = (
                    meta_collector.id_to_default_require_prior_kind[node_id] 
                    if is_skip_mode 
                    else meta_collector.id_to_require_prior_kind[node_id]
                )
                
                # 2. Check if node is executable
                collect_result = meta_collector.check_excutable(session_id, store, require_kind)
                load_collected_data(collect_result['collected_node'], input_data, store)
                
                # 3. Handle missing dependencies
                if not collect_result['excutable']:
                    missing_kinds = collect_result['missing_kind']
                    node_ids_missing = [
                        meta_collector.kind_to_node_ids[kind][0] 
                        for kind in missing_kinds
                    ]
                    
                    logger.info(
                        f"`{node_id}` require kind missing `{missing_kinds}`, "
                        f"need to execute prerequisite nodes: {node_ids_missing}"
                    )
                    
                    # 4. Recursively execute missing predecessor nodes
                    async def execute_missing_dependencies(
                        missing_kinds: List[str], 
                        for_node_id: str,
                        depth: int = 0
                    ):
                        """
                        Recursively execute missing dependency nodes
                        
                        Args:
                            missing_kinds: List of missing dependency types
                            for_node_id: ID of the node currently resolving dependencies
                            depth: Recursion depth (used for log indentation)
                        """

                        if not missing_kinds:
                            return
                        
                        indent = "  " * depth
                        logger.info(f"{indent}├─ Resolving dependencies for `{for_node_id}`: {missing_kinds}")
                        
                        for kind in missing_kinds:
                            success = False
                            candidates = meta_collector.kind_to_node_ids[kind]
                            
                            for miss_id in candidates:
                                try:
                                    await execute_node_with_default_mode(
                                        miss_id, 
                                        for_node_id=for_node_id,
                                        depth=depth
                                    )
                                    logger.info(
                                        f"{indent}│  ✓ `{miss_id}` executed successfully for kind `{kind}`"
                                    )
                                    success = True
                                    break
                                except ToolException as e:
                                    logger.warning(
                                        f"{indent}│  ✗ `{miss_id}` failed: {str(e)}"
                                    )
                                    continue
                            
                            if not success:
                                raise ToolException(
                                    f"Cannot satisfy dependency `{kind}` required by `{for_node_id}`. "
                                    f"All candidates failed: {candidates}"
                                )
                    
                    async def execute_node_with_default_mode(
                        miss_id: str, 
                        for_node_id: str,
                        depth: int = 0
                    ):
                        """
                        Execute specified node in default mode
                        
                        Args:
                            miss_id: ID of the node to execute
                            for_node_id: ID of the parent node requesting this execution
                            depth: Recursion depth
                        """
                        indent = "  " * depth
                        logger.info(
                            f"{indent}├─ [Default Mode] Executing `{miss_id}` "
                            f"(required by `{for_node_id}`)"
                        )
                        
                        # Prepare tool invocation arguments
                        tool = meta_collector.get_tool(miss_id)
                        tool_call_input = {
                            'artifact_id': store.generate_artifact_id(miss_id),
                            'mode': 'default'
                        }
                        
                        # Verify dependencies for this node
                        default_require = meta_collector.id_to_default_require_prior_kind[miss_id]
                        default_collect_result = meta_collector.check_excutable(
                            session_id, store, default_require
                        )
                        default_collect_result = meta_collector.check_excutable(session_id, store, default_require)
                        
                        # Recursively process dependencies
                        if default_collect_result['excutable']:
                            load_collected_data(
                                default_collect_result['collected_node'], 
                                tool_call_input, 
                                store
                            )
                            logger.debug(f"{indent}│  Dependencies satisfied for `{miss_id}`")
                        else:
                            logger.info(
                                f"{indent}│  `{miss_id}` has missing dependencies: "
                                f"{default_collect_result['missing_kind']}"
                            )
                            await execute_missing_dependencies(
                                default_collect_result['missing_kind'],
                                for_node_id=miss_id,  # Pass miss node_id here
                                depth=depth + 1  # Increment recursion depth
                            )
                        
                        # Invoke the tool
                        try:
                            output = await tool.arun(
                                ToolCall(
                                    args=tool_call_input, 
                                    tool_call_type='default', 
                                    runtime=runtime
                                )
                            )
                            logger.info(f"{indent}└─ ✓ `{miss_id}` completed successfully")
                            return output
                        except Exception as e:
                            logger.error(f"{indent}└─ ✗ `{miss_id}` execution failed: {str(e)}")
                            raise ToolException(f"Failed to execute `{miss_id}`: {str(e)}")
                    
                    # Start executing missing dependencies
                    await execute_missing_dependencies(missing_kinds, for_node_id=node_id)
                    
                    # Collect dependencies again
                    collect_result = meta_collector.check_excutable(session_id, store, require_kind)
                    load_collected_data(collect_result['collected_node'], input_data, store)
            else:
                input_data['artifacts_dir'] = store.artifacts_dir

            new_req_args = {
                'artifact_id': artifact_id,
                'lang': lang,
            }
            new_req_args.update(request.args)
            new_req_args.update(input_data)

            modified_request = request.override(
                args=new_req_args
            )
            return await handler(modified_request)
        except Exception as e:
            logger.error("[ToolInterceptor]"+ "".join(traceback.format_exception(e)))
            raise

    @staticmethod
    async def save_media_content_after(
        request: MCPToolCallRequest,
        handler,
    ):
        result = ""
        """End agent run when task is marked complete."""
        try:
            tool_call_result: CallToolResult = await handler(request)
            client_ctx = request.runtime.context

            
            result = tool_call_result.model_dump()
            tool_result = json.loads(result['content'][0]['text'])
            node_id = request.name
            
            artifact_id = tool_result['artifact_id']
            session_id = client_ctx.session_id

            store = request.runtime.store

            if not tool_result['isError']:
                if node_id == 'search_media':
                    store.save_result(
                        session_id,
                        node_id,
                        tool_result,
                        Path(client_ctx.media_dir),
                    )
                else:
                    store.save_result(
                        session_id,
                        node_id,
                        tool_result,
                    )
            tool_call_id = request.runtime.tool_call_id
            
            if node_id == 'read_node_history':
                tool_excute_result = tool_result['tool_excute_result']
            else:
                tool_excute_result = {}

            return Command(
                update={
                    "messages": [
                        ToolMessage(content={
                            'summary': {
                                'node_summary': tool_result['summary'],
                                'tool_excute_result': tool_excute_result
                            },
                            'isError': tool_result['isError']
                        }, tool_call_id=tool_call_id)
                    ],
                    "status": "done"
                },
            )
        except Exception as e:
            logger.error("[ToolInterceptor]"+ "".join(traceback.format_exception(e)))
            logger.error(f"Tool Call result: {result}")
            raise

    @staticmethod
    async def inject_tts_config(request: MCPToolCallRequest, handler):
        """
        Interceptor: Injects runtime.context.tts_config parameters into request.args before invoking voiceover/TTS tools.
        - tts_config: {"provider": "bytedance", "bytedance": {...}, "azure": {...}, ...}
        """
        try:
            tool_name = str(getattr(request, "name", "") or "")
            args = getattr(request, "args", None)

            if "voiceover" not in tool_name or not isinstance(args, dict):
                return await handler(request)

            runtime = getattr(request, "runtime", None)
            ctx = getattr(runtime, "context", None) if runtime else None
            tts_cfg = getattr(ctx, "tts_config", None) if ctx else None
            if not isinstance(tts_cfg, dict):
                return await handler(request)

            provider = str(tts_cfg.get("provider") or "").strip().lower()

            if not provider:
                args.setdefault("provider", "302")
                return await handler(request)

            args.setdefault("provider", provider)

            provider_cfg = tts_cfg.get(provider)
            if isinstance(provider_cfg, dict):
                for key, value in provider_cfg.items():
                    if value is None:                        
                        continue
                    args.setdefault(key, str(value).strip())
        except Exception as e:
            logger.warning(f"Failed to inject TTS config: {e}")
        return await handler(request)
    
    @staticmethod
    async def inject_pexels_api_key(request: MCPToolCallRequest, handler):
        """
        Interceptor: Injects runtime.context.pexels_api_key into request.args before invoking media search tools.
        - If pexels_api_key is empty/None: do nothing (tool will fall back to env var internally).
        """
        try:
            tool_name = str(getattr(request, "name", "") or "")
            args = getattr(request, "args", None)

            if not isinstance(args, dict):
                return await handler(request)

            if "search_media" not in tool_name:
                return await handler(request)

            runtime = getattr(request, "runtime", None)
            ctx = getattr(runtime, "context", None) if runtime else None
            key = getattr(ctx, "pexels_api_key", None) if ctx else None
            key = str(key or "").strip()

            if not key:
                return await handler(request)

            args["pexels_api_key"] = key

        except Exception as e:
            logger.warning(f"Failed to inject pexels API key: {e}")
        return await handler(request)
