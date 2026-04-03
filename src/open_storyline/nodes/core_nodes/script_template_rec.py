from typing import Any, Dict
from pathlib import Path

import numpy as np
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.recall import StorylineRecall
from open_storyline.utils.element_filter import ElementFilter
from open_storyline.nodes.node_schema import RecommendScriptTemplateInput
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class ScriptTemplateRecomendation(BaseNode):

    meta = NodeMeta(
        name="script_template_rec",
        description="Select an script template (script style) for generation",
        node_id="script_template_rec",
        node_kind="script_template_rec",
        require_prior_kind=[],
        default_require_prior_kind=[],
        next_available_node=["generate_script"],
    )

    input_schema = RecommendScriptTemplateInput

    def __init__(self, server_cfg):
        super().__init__(server_cfg)
        self.element_filter = ElementFilter(json_path=self.server_cfg.script_template.script_template_info_path)
        self.vectorstore = StorylineRecall.build_vectorstore(self.element_filter.library)
        self._top_n = 3

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]):
        return {}
    
    async def process(self, node_state: NodeState, inputs: Dict[str, Any]):
        
        user_request = inputs.get("user_request", "")
        filter_include = inputs.get("filter_include", {})
        filter_exclude = inputs.get("filter_exclude", {})

        # Step1: Check resources
        script_template_dir: Path = self.server_cfg.script_template.script_template_dir.expanduser().resolve()
        if not script_template_dir.exists():
            raise FileNotFoundError(f"`script_template_dir` not found: {script_template_dir}")
        if not script_template_dir.is_dir():
            raise NotADirectoryError(f"`script_template_dir` is not a directory: {script_template_dir}")
        
        # Step2: Full Recall
        candidates = StorylineRecall.query_top_n(self.vectorstore, query=user_request)

        # Step3: Filter tags
        candidates = self.element_filter.filter(candidates, filter_include, filter_exclude)

        if not candidates:
            node_state.node_summary.add_error("")
        
        return {"candidates": candidates[:min(self._top_n, len(candidates))]}