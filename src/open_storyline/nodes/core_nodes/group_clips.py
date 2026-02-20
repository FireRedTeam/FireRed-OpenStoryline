from typing import Any, Dict

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import GroupClipsInput
from src.open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_dict
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class GroupClipsNode(BaseNode):

    meta = NodeMeta(
        name="group_clips",
        description="Group clips based on their descriptions according to user requirements. Depends on the filter_clips tool output",
        node_id="group_clips",
        node_kind="group_clips",
        require_prior_kind=['filter_clips'],
        default_require_prior_kind=['filter_clips'],
        next_available_node=['generate_script', 'generate_script_pro'],
    )
    input_schema = GroupClipsInput

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        result = _make_single_group_fallback(inputs["filter_clips"].get("selected", []))
        return {
            "groups": result,
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any], **params) -> Any:
        clip_captions = inputs["filter_clips"].get("clip_captions")
        selected_clips = inputs["filter_clips"].get("selected")
        user_request = inputs["user_request"]

        llm = node_state.llm
        clip_lookup = _build_clip_lookup(clip_captions)

        if not selected_clips:
            return {"groups": []}
        
        selected_clips_captions = [clip_lookup[cid] for cid in selected_clips if cid in clip_lookup]
        if not selected_clips_captions:
            return {"groups": _make_single_group_fallback(selected_clips)}

        clip_block = _build_clips_block(selected_clips_captions)

        system_prompt = get_prompt("group_clips.system", lang=node_state.lang)
        if user_request == "":
            user_request = "No additional requirements"

        user_prompt = get_prompt(
            "group_clips.user",
            lang=node_state.lang,
            user_request=user_request,
            selected_clips=selected_clips,
            clip_captions=clip_block,
            clip_number=len(clip_block),
        )

        cfg = self.server_cfg.group_clips
        max_attempts = int(cfg.max_parse_retries) + 1
        initial_max_tokens = _estimate_group_output_tokens(
            clip_count=len(selected_clips),
            base_max_tokens=int(cfg.base_max_tokens),
            tokens_per_clip=int(cfg.tokens_per_clip),
            max_tokens_cap=int(cfg.max_tokens_cap),
        )
        retry_token_step = int(cfg.retry_token_step)
        max_tokens_cap = int(cfg.max_tokens_cap)

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            max_tokens = min(max_tokens_cap, initial_max_tokens + attempt * retry_token_step)
            attempt_user_prompt = user_prompt
            if attempt > 0:
                attempt_user_prompt = _append_compact_output_hint(user_prompt, node_state.lang)

            try:
                raw = await llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=attempt_user_prompt,
                    media=None,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=max_tokens,
                    model_preferences=None,
                )
            except Exception as e:
                last_error = e
                continue

            try:
                obj = parse_json_dict(raw)
                groups_raw = _extract_groups_obj(obj)
                groups = _normalize_groups_from_llm(
                    groups_raw=groups_raw,
                    selected_ids_set=set(selected_clips),
                )
                node_state.node_summary.info_for_user(
                    f"Grouping successful: {len(groups)} groups in total"
                )
                return {"groups": groups}
            except Exception as e:
                last_error = e
                continue

        result = _make_single_group_fallback(selected_clips)
        node_state.node_summary.info_for_user(
            f"Grouping error after {max_attempts} attempt(s): {last_error}\nUsing default strategy"
        )
        return {"groups": result}

def _extract_groups_obj(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("groups"), list):
        return obj["groups"]
    if isinstance(obj, list):
        return obj
    raise ValueError("LLM output does not contain groups.")


def _normalize_groups_from_llm(
    groups_raw: list[dict[str, Any]],
    selected_ids_set: set[str],
) -> list[dict[str, Any]]:
    """
    Validate and normalize LLM output groups:
    - clip_ids must all come from selected
    - clip_ids cannot be duplicated
    - group_id will be uniformly rewritten by code
    - If summary is missing, fill with default
    """
    if not groups_raw:
        raise ValueError("groups is empty.")

    # First extract and perform basic cleaning
    normalized_groups: list[dict[str, Any]] = []
    seen: set[str] = set()

    for gi, g in enumerate(groups_raw):
        if not isinstance(g, dict):
            raise ValueError(f"groups[{gi}] is not a dict, please try running again.")

        clip_ids = g.get("clip_ids")
        if not isinstance(clip_ids, list) or not clip_ids:
            raise ValueError(f"groups[{gi}].clip_ids must be a non-empty list, please try running again.")

        # Deduplicate clip_ids (preserve original output order)
        cleaned_clip_ids: list[str] = []

        for cid in clip_ids:
            if not isinstance(cid, str):
                continue
            if cid not in selected_ids_set:
                continue
                # raise ValueError(f"groups[{gi}] contains non-selected clip_id: {cid}")
            if cid in seen:
                continue
            seen.add(cid)
            cleaned_clip_ids.append(cid)

        if not cleaned_clip_ids:
            raise ValueError(f"groups[{gi}] clip_ids is empty after cleaning, please try running again.")

        summary = g.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "A group of shots for carrying the same script and voiceover."

        normalized_groups.append(
            {  
                "group_id": "", # group_id placeholder, will be rewritten later
                "summary": summary.strip(),
                "clip_ids": cleaned_clip_ids,
                "duration": g.get("duration")
            }
        )

    # Finally rewrite group_id
    for i, g in enumerate(normalized_groups, start=1):
        g["group_id"] = f"group_{i:04d}"

    return normalized_groups


def _build_clip_lookup(clip_captions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for clip in clip_captions:
        cid = clip.get("clip_id")
        if cid:
            lookup[cid] = clip
    return lookup

def _make_single_group_fallback(
    selected_clips: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "group_id": "group_0001",
            "summary": "Aggregate all selected shots in original order for subsequent script and voiceover generation.",
            "clip_ids": selected_clips,
        }
    ]

def _build_clips_block(clip_captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Construct clips into stable text blocks
    """
    blocks: list[dict] = []
    for clip in clip_captions:
        clip_id = clip.get("clip_id", "")
        duration = clip.get("duration",0.0)
        caption = clip.get("caption", "")
        block = {
            "clip_id": clip_id,
            "duration": duration,
            "caption": caption
        }
        blocks.append(block)
    return blocks


def _estimate_group_output_tokens(
    *,
    clip_count: int,
    base_max_tokens: int,
    tokens_per_clip: int,
    max_tokens_cap: int,
) -> int:
    estimated = base_max_tokens + max(0, clip_count) * tokens_per_clip
    return max(256, min(max_tokens_cap, estimated))


def _append_compact_output_hint(user_prompt: str, lang: str) -> str:
    if str(lang).lower().startswith("zh"):
        extra_hint = (
            "\n\n补充约束（必须遵守）：\n"
            "1. 输出紧凑 JSON，不要任何多余文本；\n"
            "2. think 字段最多 80 字；\n"
            "3. 每个 group 的 summary 最多 20 字。"
        )
    else:
        extra_hint = (
            "\n\nAdditional constraints (must follow):\n"
            "1. Output compact JSON only, no extra text;\n"
            "2. Keep think within 80 words;\n"
            "3. Keep each group summary within 20 words."
        )
    return f"{user_prompt}{extra_hint}"
