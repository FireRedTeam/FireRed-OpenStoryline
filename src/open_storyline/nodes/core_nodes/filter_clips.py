from typing import Any, Dict

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import FilterClipsInput
from open_storyline.mcp.sampling_requester import LLMClient
from src.open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_dict
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class FilterClipsNode(BaseNode):

    meta = NodeMeta(
        name="filter_clips",
        description="Filter clips based on their descriptions according to user requirements. Depends on the results from the understand_clips tool",
        node_id="filter_clips",
        node_kind="filter_clips",
        require_prior_kind=['split_shots','understand_clips'],
        default_require_prior_kind=['split_shots','understand_clips'],
        next_available_node=['group_clips', 'group_clips_pro'],
    )

    input_schema = FilterClipsInput

    def _parse_input(self, node_state: NodeState, inputs: Dict[str, Any]):
        clip_captions = inputs["understand_clips"].get("clip_captions")
        clip_info = inputs["split_shots"]["clips"]
        duration_lookup = _build_duration_lookup(clip_info)
        clip_captions=_add_input_duration(clip_captions,duration_lookup)

        input_clip_ids: list[str] = [
            (c.get("clip_id")) for c in clip_captions
        ]
        inputs["input_clip_ids"] = input_clip_ids
        inputs["clip_captions"] = clip_captions
        return inputs


    async def default_process(
        self,
        node_state,
        inputs: Dict[str, Any],
    ) -> Any:
        clip_captions = inputs["understand_clips"].get("clip_captions")
        
        node_state.node_summary.info_for_user("Using all clips")
        return {
            "clip_captions": clip_captions,
            "selected": inputs["input_clip_ids"],
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        clip_captions = inputs["understand_clips"].get("clip_captions")
        user_request = inputs["user_request"]
        llm = node_state.llm
        
        input_clip_ids = inputs["input_clip_ids"]

        if not user_request or user_request == "":
            node_state.node_summary.info_for_user("User did not specify requirements, using all clips")
            return {
                "clip_captions": clip_captions,
                "selected": input_clip_ids,
            }
        
        else:
            clip_block = _build_clips_block(clip_captions)
            system_prompt = get_prompt("filter_clips.system", lang=node_state.lang)
            user_prompt = get_prompt("filter_clips.user", lang=node_state.lang, user_request=user_request, clip_captions=clip_block)

            raw = await llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                media=None,
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048,
                model_preferences=None,
            )

        try:
            obj = parse_json_dict(raw)
            select_ids = _extract_selected_ids(obj, input_clip_ids)
            node_state.node_summary.info_for_user(f"Successfully filtered {len(select_ids)} clips")
        
        except:
            select_ids = input_clip_ids
            node_state.node_summary.info_for_user("Failed to parse model output, using all clips")

        return {
            "clip_captions": clip_captions,
            "selected": select_ids,
        }



def _add_input_duration(clip_captions:list[dict[str, Any]],clip_durations: dict[str, float]) -> Any:
    for i in range(len(clip_captions)):
        clip_id=clip_captions[i].get('clip_id','')
        if not clip_id:continue
        if clip_id in clip_durations:
            clip_captions[i]['duration']=clip_durations[clip_id]
    return clip_captions


def _build_duration_lookup(clip_info: list[dict[str, Any]]) -> dict[str, float]:
    """
    clip_id -> duration_sec
    """
    out: dict[str, float] = {}
    for item in clip_info or []:
        cid = item.get("clip_id")
        if not cid:
            continue
        src = item.get("source_ref") or {}
        dur = src.get("duration", 0) / 1000.0
        if dur==0.0:dur=2.0
        out[cid] = dur
    return out


def _extract_selected_ids(
    obj: dict[str, Any],
    input_clip_ids: list[str],
) -> list[str]:
    """
    Extract selected clip_id list from LLM structured output.
    Returns: Filtered results ordered by input_clip_ids (preserving only valid input IDs)
    """
    id_set = set(input_clip_ids)

    results = obj.get("results")
    if not isinstance(results, list):
        raise ValueError('"results" must be a list')

    true_items = 0
    valid_true_ids: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue
        cid = item.get("clip_id")
        keep = item.get("keep")

        # keep allows bool or "true"/"false"
        keep_bool = None
        if isinstance(keep, bool):
            keep_bool = keep
        elif isinstance(keep, str):
            s = keep.strip().lower()
            if s in ("true", "yes", "1"):
                keep_bool = True
            elif s in ("false", "no", "0"):
                keep_bool = False

        if keep_bool is True:
            true_items += 1
            if isinstance(cid, str) and cid in id_set:
                valid_true_ids.add(cid)

    # If the model explicitly selected items (keep=true) but none match the input
    if true_items > 0 and not valid_true_ids:
        raise ValueError("results has keep=true entries, but no valid clip_ids (model may have modified IDs)")

    return [cid for cid in input_clip_ids if cid in valid_true_ids]

def _build_clips_block(clip_captions: list[dict[str, Any]]) -> str:
    """
    Construct clips into stable text blocks
    """
    blocks: list[str] = []
    for clip in clip_captions:
        cid = clip.get("clip_id", "")
        caption = clip.get("caption", "")
        block = (
            f"[clip_id={cid}]\n"
            f"caption: {caption}\n"
        )
        blocks.append(block)
    return "\n".join(blocks).strip() + "\n"
