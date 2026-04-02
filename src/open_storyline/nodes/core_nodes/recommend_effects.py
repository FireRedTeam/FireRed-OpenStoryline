from typing import Any, Dict, List
from pathlib import Path

import numpy as np

from open_storyline.utils.element_filter import ElementFilter
from open_storyline.utils.prompts import get_prompt
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import RecommendTransitionInput, RecommendTextInput
from open_storyline.utils.parse_json import parse_json_dict
from src.open_storyline.config import Settings
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class RecommendTransitionNode(BaseNode):
    meta = NodeMeta(
        name="elementrec_transition",
        description="Recommend transition effects according to user needs and segment count, ensuring transition list length equals group count",
        node_id="elementrec_transition",
        node_kind="transition_rec",
        require_prior_kind=['group_clips'],
        default_require_prior_kind=[],
        next_available_node=["plan_timeline_pro"],
    )

    input_schema = RecommendTransitionInput

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        node_state.node_summary.info_for_user(f"[{self.meta.node_id}] Transition effect not used")
        return []


    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        duration = inputs.get('duration', 1000) # default 1000ms
        node_state.node_summary.info_for_user(
            f"[{self.meta.node_id}] Adding fade transitions: {duration}ms fade-in at start, {duration}ms fade-out at end"
        )
        return [
            {
                'type': 'fade_in',
                'position': 'opening',
                'duration': duration
            },
            {
                'type': 'fade_out',
                'position': 'ending',
                'duration': duration
            }
        ]


@NODE_REGISTRY.register()
class RecommendTextNode(BaseNode):
    meta = NodeMeta(
        name="elementrec_text",
        description="Recommend text effects according to user needs",
        node_id="elementrec_text",
        node_kind="text_rec",
        require_prior_kind=["generate_script"],
        default_require_prior_kind=[],
        next_available_node=["plan_timeline"],
    )

    input_schema = RecommendTextInput


    def __init__(self, server_cfg: Settings) -> None:
        super().__init__(server_cfg)
        self.text_filter = ElementFilter(json_path=str(server_cfg.recommend_text.font_info_path))

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        self.text_filter.filter()
        node_state.node_summary.info_for_user(f"[{self.meta.node_id}] Using default font")
        return [{"font_name": "SiYuanHeiTi", "font_color": inputs.get("font_color", (255,255,255,255))}]

    async def process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        user_request = inputs.get("user_request", "")
        filter_include = inputs.get("filter_include", {})
        group_scripts = inputs.get("generate_script", {}).get("group_scripts", {})

        candidates = self.text_filter.filter(filter_include=filter_include).copy()

        font_paths = [cand.pop("font_path", None) for cand in candidates]
        llm = node_state.llm
        system_prompt = get_prompt("elementrec_text.system", lang=node_state.lang)
        user_prompt = get_prompt("elementrec_text.user", lang=node_state.lang, scripts=group_scripts, candidates=candidates, user_request=user_request)
        raw = await llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
            model_preferences=None,
        )
        try:
            selected_json = parse_json_dict(raw)
        except:
            selected_json = (raw or "").strip() if raw else "Error: Unable to parse the model output"
            node_state.node_summary.add_error(selected_json)
            return None
        selected_json.update({"font_color": inputs.get("font_color", (255,255,255,255))})
        node_state.node_summary.info_for_user(f"[{self.meta.node_id}] Use font `{selected_json['font_name']}`")
        return [selected_json]
