from typing import Any, Dict
import asyncio

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from src.open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_dict
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import UnderstandClipsInput
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class UnderstandClipsNode(BaseNode):
    """
    Media Understanding Node
    """

    meta = NodeMeta(
        name="understand_clips",
        description="Analyze clips and generate descriptions for each. Requires `load_media` and `split_shots` output",
        node_id="understand_clips",
        node_kind="understand_clips",
        require_prior_kind=['load_media', 'split_shots'],
        default_require_prior_kind=['load_media', 'split_shots'],
        next_available_node=['filter_clips', 'filter_clips_pro'],
    )

    input_schema = UnderstandClipsInput

    async def default_process(
        self,
        node_state: NodeState,
        inputs: Dict[str, Any],
    ) -> Any:
        clips = inputs["split_shots"]["clips"]

        clip_captions: list[dict[str, Any]] = []
        for clip in clips or []:
            clip_captions.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "caption": "no caption",
                    "source_ref": {
                        "media_id": clip.get("source_ref", {}).get("media_id", ""),
                    }
                }
            )
        node_state.node_summary.info_for_user(f"Skipped description generation for {len(clips)} clips")
        return {
            "clip_captions": clip_captions,
            "overall": "unknown",
        }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        inputs: Previous node results read by BaseNode.load_inputs(ctx)
        """
        load_media = inputs["media"]
        clips = inputs["split_shots"]["clips"]
        llm = node_state.llm
        system_prompt = get_prompt("understand_clips.system_detail", lang=node_state.lang)
        user_prompt = get_prompt("understand_clips.user_detail", lang=node_state.lang)


        clip_captions: list[dict[str, Any]] = []

        for clip in clips or []:
            clip_id = str(clip.get("clip_id", "") or "").strip() or "(unknown_clip)"
            kind = str(clip.get("kind", "") or "").strip().lower()
            src = clip.get("source_ref") or {}

            media_id = str(src.get("media_id", "") or "")
            media_item = load_media.get(media_id)

            out_item: dict[str, Any] = {
                "clip_id": clip_id,
            }
            
            if not media_item:
                out_item["caption"] = f"Error: Media not found for media_id={media_id}"
                clip_captions.append(out_item)
                continue

            path = str(media_item.get("path", "") or "").strip()
            if not path:
                out_item["caption"] = f"Error: No path specified for media_id={media_id}"
                clip_captions.append(out_item)
                continue

            # 组装 media
            media: list[Any] = []

            if kind == "image":
                media = [{"path": path}]

            elif kind == "video":
                in_sec = _safe_float(src.get("start", 0) / 1000.0, 0.0)

                if src.get("end") is not None:
                    out_sec = _safe_float(src.get("end", 0) / 1000.0, in_sec)
                else:
                    dur = _safe_float(src.get("duration", 0.0), 0.0)
                    out_sec = in_sec + max(0.0, dur)
                
                if out_sec <= in_sec:
                    out_sec = in_sec + 0.1

                media = [{
                    "path": path,
                    "in_sec": in_sec,
                    "out_sec": out_sec,
                }]
            else:
                out_item["caption"] = f"Error: Clip kind not supported: {kind}"
                clip_captions.append(out_item)
                continue
    
            max_retries = 2
            raw = None
            last_exc: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    raw = await llm.complete(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        media=media,
                        temperature=0.3,
                        top_p=0.9,
                        max_tokens=2048,
                        model_preferences=None,
                    )
                    if raw is not None:
                        last_exc = None
                        break
                except Exception as e:
                    last_exc = e

                if attempt < max_retries:
                    await asyncio.sleep(0.3 * (attempt + 1))

            if raw is None:
                out_item["caption"] = "Error: VLM request failed"
                try:
                    raw_score = obj.get("aes_score")
                    out_item["aes_score"] = float(str(raw_score).strip())
                except (ValueError, TypeError, AttributeError):
                    # If the conversion fails (such as "abc", None, "nan", etc.), assign the value -1.0
                    out_item["aes_score"] = -1.0
                node_state.node_summary.add_error(repr(last_exc))
                clip_captions.append(out_item)
                continue

            try:
                obj = parse_json_dict(raw)
            except:
                text = (raw or "").strip()
                out_item["caption"] = text if text else "Error: Unable to parse model output"
                clip_captions.append(out_item)
                continue

            out_item["caption"] = str(obj.get("caption", "") or "").strip()
            out_item["source_ref"] = {
                "media_id": clip.get("source_ref", {}).get("media_id", ""),
            }
            clip_captions.append(out_item)

        desc_lines: list[str] = []
        for desc in clip_captions:
            text = str(desc.get("caption"))
            desc_lines.append(f"- {desc.get('clip_instance_id')}: {text}")

        overall_summary = ""
        if desc_lines:
            overall_system_prompt = get_prompt("understand_clips.system_overall", lang=node_state.lang)
            overall_user_prompt = get_prompt("understand_clips.user_overall", lang=node_state.lang, clips_captions=desc_lines)

            try:
                overall_summary = await llm.complete(
                    system_prompt=overall_system_prompt,
                    user_prompt=overall_user_prompt,
                    media=None,
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=1024,
                    model_preferences=None
                )
            
            except Exception as e:
                overall_summary = f"Error: Summary generation failed: {type(e).__name__}: {e}"
            node_state.node_summary.info_for_user(f"Clip understanding completed. Analyzed {len(clip_captions)} clips in total. Overall description: {overall_summary}")
        return {
            "clip_captions": clip_captions,
            "overall": overall_summary
        }
    

    def _parse_input(self, node_state: NodeState, inputs: Dict[str, Any]):
        media = inputs["load_media"]["media"]

        load_media: dict[str, dict[str, Any]] = {}
        for media_item in media or []:
            media_id = media_item.get("media_id")
            if media_id:
                load_media[str(media_id)] = media_item
        inputs.update({"media": load_media})
        return inputs

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default