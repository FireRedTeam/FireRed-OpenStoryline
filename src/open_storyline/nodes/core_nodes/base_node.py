from abc import ABC, abstractmethod
import os
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, List, Optional, Union, ClassVar
import json
import traceback
import re

from open_storyline.config import Settings
from open_storyline.nodes.node_state import NodeState

from open_storyline.storage.file import FileCompressor
from open_storyline.utils.logging import get_logger
from open_storyline.mcp.sampling_requester import LLMClient
from open_storyline.mcp.hooks.node_interceptors import should_inline_media_as_base64

logger = get_logger(__name__)

@dataclass
class NodeMeta:
    """
    Node Metadata Class
    
    Defines metadata information for nodes in a workflow or flowchart, including node 
    identification, type, dependencies, and downstream node routing configurations.
    
    Attributes:
        name: Tool name
        description: Tool functionality description
        node_id: Unique node identifier for uniquely locating the node within the entire process
        node_kind: Node type/category, such as "start", "process", "end", etc.
        require_prior_kind: List of prerequisite node types that must be completed 
                           before the current node runs
                           Example: ["validation", "authentication"]
        default_require_prior_kind: Default list of prerequisite node types, serving as 
                                   the default configuration or fallback for require_prior_kind
        next_available_node: List of downstream node IDs that the current node can transition to 
                            after execution completes
                            Used to define possible branch paths in the workflow
        priority: Execution priority among nodes with the same functionality
    """

    name: str
    description: str
    node_id: str
    node_kind: str
    require_prior_kind: List[str] = field(default_factory=list)
    default_require_prior_kind: List[str] = field(default_factory=list)
    next_available_node: List[str] = field(default_factory=list)
    priority: int = 5



class BaseNode(ABC):
    meta: NodeMeta
    input_schema: ClassVar[type[BaseModel] | None] = None
    # output_schema: ClassVar[type[BaseModel] | None] = None

    def __init__(self, server_cfg: Settings) -> None:
        self.server_cfg = server_cfg
        self.server_cache_dir  = Path(os.getcwd()) / self.server_cfg.local_mcp_server.server_cache_dir

        if not hasattr(self, "meta"):
            raise ValueError("Subclass must define the 'meta' attribute")

    
    def _load_user_info(self, node_state: NodeState, params: Dict[str,Any]) -> Dict[str,Any]:
        return {
            "session_id": node_state.session_id,
            "artifact_id": node_state.artifact_id
        }

    def _load_item(self, node_state: NodeState, user_info: Dict[str,str], item: Dict[str,Any]):
        new_item: Dict[str, Any] = {}
        item_base64 = item.pop("base64", None)
        item_md5 = item.pop("md5", None)
        item_path = item.pop("path", None)
        new_item.update(item)

        if item_base64 and item_path:
            item_save_path = self.server_cache_dir / user_info['session_id'] / user_info['artifact_id'] / os.path.basename(item_path)
            FileCompressor.decompress_from_string(item_base64, item_save_path)
            new_item['path'] = str(item_save_path.relative_to(os.getcwd()))
            new_item['orig_path'] = str(item_path)
            new_item['orig_md5'] = item_md5
        elif item_path:
            # Path-only mode (local MCP):
            # - if item_path is absolute, accept it directly;
            # - if item_path is relative, resolve it under media_dir and ensure it does not escape that directory.
            if os.path.isabs(item_path):
                full_path = Path(item_path)
            else:
                media_root = Path(self.server_cfg.project.media_dir).resolve()
                full_path = (media_root / item_path).resolve()
                try:
                    full_path.relative_to(media_root)
                except ValueError:
                    raise ValueError(
                        f"relative path-only input must be under media_dir: {item_path!r}"
                    )
            new_item['path'] = str(full_path)
            new_item['orig_path'] = str(item_path)
            new_item['orig_md5'] = item_md5
        return new_item

    _FILE_EXT_RE = re.compile(
        r"\.(mp4|mov|avi|mkv|webm|mp3|wav|m4a|aac|flac|ogg|png|jpg|jpeg|gif|bmp|webp|json)$",
        re.IGNORECASE,
    )

    @classmethod
    def _looks_like_file_path(cls, p: Any) -> bool:
        if not isinstance(p, str):
            return False
        s = p.strip()
        if not s:
            return False
        if "://" in s:
            return False
        if cls._FILE_EXT_RE.search(s):
            return True
        if "\\" in s or "/" in s:
            return True
        if len(s) >= 2 and s[1] == ":":
            return True
        return False

    def _pack_item(self, node_state: NodeState, item: Dict[str, Any]):
        orig_path = item.pop('orig_path', None)
        orig_md5 = item.pop('orig_md5', None)
        server_save_path = item.pop('path', None)
        if not server_save_path:
            return item

        # Decide response transport policy (path-only vs base64) using the same policy
        # function as the client side.
        inline_base64 = should_inline_media_as_base64(self.server_cfg)

        # Prefer a media_dir-relative path for responses when possible, with absolute
        # path as a fallback.
        #
        # Important: in path-only mode, returning a relative `orig_path` that is NOT
        # under media_dir can break downstream nodes, because the server resolves
        # relative path-only inputs under project.media_dir. For such cases (e.g. BGM
        # assets under resource/bgms), fall back to an absolute path.
        client_path = orig_path or server_save_path
        try:
            media_root = Path(self.server_cfg.project.media_dir).resolve()
        except Exception:
            media_root = None
        if media_root is not None:
            try:
                client_rel = Path(server_save_path).resolve().relative_to(media_root)
                client_path = str(client_rel)
            except Exception:
                if orig_path and os.path.isabs(str(orig_path)):
                    client_path = str(orig_path)
                else:
                    client_path = str(Path(server_save_path).resolve())
        else:
            if orig_path and os.path.isabs(str(orig_path)):
                client_path = str(orig_path)
            else:
                client_path = str(Path(server_save_path).resolve())

        if not inline_base64:
            node_state.node_summary.debug_for_dev(
                f"[node] node_id: {self.meta.node_id} return `path` only (local mode)"
            )
            item['path'] = client_path
            return item

        # Remote / base64 mode: keep existing md5-based optimisation when possible.
        compress_data = FileCompressor.compress_and_encode(server_save_path)
        if orig_path and orig_md5 and compress_data.md5 == orig_md5:
            node_state.node_summary.debug_for_dev(
                f"[node] node_id: {self.meta.node_id} change `path` change to {orig_path}"
            )
            item['path'] = orig_path
        else:
            node_state.node_summary.debug_for_dev(
                f"[node] node_id: {self.meta.node_id} return `base64` to client"
            )
            item['base64'] = compress_data.base64
            item['path'] = compress_data.filename
            item['md5'] = compress_data.md5
        return item


    def load_inputs_from_client(self, node_state: NodeState, params: Dict[str,Any], user_info: Optional[Dict[str,str]] = None, save: bool=True) -> Dict[str, Any]:
        """
        Read data from client's request and save the transmitted base64 data on the Server.
        """

        if user_info is None:
            user_info = self._load_user_info(node_state, params)

        payload_key = params.keys()
        loaded_input = {}
        kwargs = {}
        for k in payload_key:
            payload_input = params[k]
            if isinstance(payload_input, list) and all([isinstance(item, dict) for item in payload_input]):
                # List: load base64 data and save to server cache
                loaded_input[k] = [self._load_item(node_state, user_info, item) for item in payload_input]
            elif isinstance(payload_input, dict):
                # Dict: recursively process nested data (without saving)
                loaded_input[k] = self.load_inputs_from_client(node_state, payload_input, user_info, save=False)
            elif isinstance(payload_input, LLMClient):
                kwargs[k] = payload_input
            else:
                # Handle primitive types: directly copy the value (e.g., str, int, float, bool)
                loaded_input[k] = params[k]

        # save loaded_input to self.cache_dir in json format
        if save:
            artifact_save_path = (self.server_cache_dir / user_info['session_id'] / user_info['artifact_id']).with_suffix(".json")
            artifact_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(artifact_save_path, 'w') as f:
                json.dump(loaded_input, f, indent=2, ensure_ascii=False)
        loaded_input.update(kwargs)
        return loaded_input


    def pack_outputs_to_client(self, node_state: NodeState, outputs: Union[Dict[str,Any],List[str]]) -> Union[Dict[str,Any],List[str]]:
        """
        Pack the output and return it to the client.
        """
        if not isinstance(outputs, dict):
            return outputs
        payload_key = outputs.keys()
        packed_output = {}
        for k in payload_key:
            payload_output = outputs[k]
            if isinstance(payload_output, list) and all(isinstance(item, dict) for item in payload_output):
                packed_output[k] = [self._pack_item(node_state, item) for item in payload_output]
            elif isinstance(payload_output, dict):
                packed_output[k] = self.pack_outputs_to_client(node_state, payload_output)
            else:
                packed_output[k] = outputs[k]
        return packed_output

    @abstractmethod
    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        Default processing method that must be implemented. Called when the node needs to be skipped.
        """
        ...
    
    @abstractmethod
    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        Main processing method that must be implemented. Executed when the node is invoked normally.
        """
        ...

    def _parse_input(self, node_state: NodeState, inputs: Dict[str, Any]):
        return inputs

    def _combine_tool_outputs(self, node_state: NodeState, outputs: Dict[str, Any]):
        return outputs
    
    def _validate_schema(
        self, 
        params: dict[str,Any], 
        schema_name: Union[str, List[str]], 
        update_params: bool = False
    ) -> Optional[Dict[str,Any]]:
        schema_names = [schema_name] if isinstance(schema_name, str) else schema_name
        
        validated_params = params.copy() if update_params else None
        
        for name in schema_names:
            schema = getattr(self, name, None)
            
            if schema is None:
                logger.warning(
                    f"Schema '{name}' does not exist, skipping validation",
                )
                continue
            
            try:
                validated = schema(**params)
                if update_params:
                    validated_params.update(validated.dict())
            except ValidationError as e:
                logger.error(f"{name} validation failed: {e}")
        return validated_params if update_params else None

    def _prepare_output_directory(self, node_state: NodeState) -> Path:
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        output_directory = self.server_cache_dir / session_id / artifact_id
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    async def __call__(self, node_state: NodeState, **params) -> Dict[str, Any]:
        try:
            mode = params.get("mode", "auto")

            inputs = self.load_inputs_from_client(node_state, params.copy())

            parsed_inputs = self._parse_input(node_state, inputs)

            if mode != 'auto':
                outputs = await self.default_process(node_state, parsed_inputs)
            else:
                outputs = await self.process(node_state, parsed_inputs)

            processed_outputs = self._combine_tool_outputs(node_state, outputs)

            packed_output = self.pack_outputs_to_client(node_state, processed_outputs)
            
            # self._validate_schema(packed_output, 'output_schema')

            return {
                'artifact_id': node_state.artifact_id,
                'summary': node_state.node_summary.get_summary(node_state.artifact_id),
                'tool_excute_result': packed_output,
                'isError': False
            }
        except Exception as e:
            node_state.node_summary.add_error(str(e), artifact_id=node_state.artifact_id)
            if self.server_cfg.developer.developer_mode:
                traceback_info = ''.join(traceback.format_exception(e))
                summary = {
                    "error_info": f"[artifact_id {node_state.artifact_id}] \n {traceback_info}"
                }
                logger.error(traceback_info)
            else:
                summary = node_state.node_summary.get_summary(node_state.artifact_id)
            return {
                'artifact_id': node_state.artifact_id,
                'summary': summary,
                'tool_excute_result': {},
                'isError': True
            }
