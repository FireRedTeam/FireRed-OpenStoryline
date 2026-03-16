from typing import Any, Dict
import os
import subprocess
import tempfile

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import SpeechRoughCutInput
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_list
from open_storyline.utils.ffmpeg_utils import (
    resolve_ffmpeg_executable,
    read_video_frames_as_rgb24,
    segment_video_stream_copy_with_ffmpeg,
    VideoSegment,
)
from open_storyline.utils.register import NODE_REGISTRY

@NODE_REGISTRY.register()
class SpeechRoughCutNode(BaseNode):

    meta = NodeMeta(
        name="speech_rough_cut",
        description="Perform rough cut on speech clips based on ASR results",
        node_id="speech_rough_cut",
        node_kind="speech_rough_cut",
        require_prior_kind=['asr'],
        default_require_prior_kind=['asr'],
        next_available_node=[],
    )

    input_schema = SpeechRoughCutInput

    def __init__(self, server_cfg):
        super().__init__(server_cfg)
        self.ffmpeg_executable = resolve_ffmpeg_executable()

    async def default_process(
        self,
        node_state,
        inputs: Dict[str, Any],
    ) -> Any:
        return {}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        
        asr_infos = inputs["asr"].get('asr_infos', [])
        video_path = inputs["asr"].get('video_path')
        gap_threshold = inputs.get('gap_threshold', 400)
        output_directory = self._prepare_output_directory(node_state, inputs)
        llm = node_state.llm
        rough_cut_jsons = []

        system_prompt = get_prompt("speech_rough_cut.system", lang=node_state.lang)

        for asr_info in asr_infos:

            user_prompt = get_prompt(
                "speech_rough_cut.user",
                lang=node_state.lang,
                asr_sentence_info=asr_info.get("asr_sentence_info", {})
            )

            try:
                raw = await llm.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    media=None,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=8092,
                    model_preferences=None,
                )
            except Exception as e:
                last_error = e

            try:
                rough_cut_json = parse_json_list(raw)
                segments = self.group_sentences(rough_cut_json, gap_threshold=gap_threshold)
                ranges = self.segments_to_ranges(segments)
                cuts = self.ranges_to_cut_points(ranges)
                segments = segment_video_stream_copy_with_ffmpeg(
                    input_video=video_path,
                    ffmpeg_executable=self.ffmpeg_executable,
                    split_points_seconds=cuts,
                    output_directory=output_directory,
                    filename_prefix=f"clip_{asr_info['clip_id']}",
                    start_index=len(rough_cut_jsons),
                )
                rough_cut_jsons.append(rough_cut_json)
            except Exception as e:
                last_error = e
        
        breakpoint()
        
        return {"rough_cut": rough_cut_jsons}
    

    def group_sentences(self, items, gap_threshold: int=400):
        segments = []
        current = [items[0]]

        for i in range(len(items) - 1):
            cur = items[i]
            next = items[i+1]

            gap = next["start"] - cur["end"]

            if gap > gap_threshold:
                segments.append(current)
                current = [next]
            else:
                current.append(next)

        if current:
            segments.append(current)

        return segments
    
    def segments_to_ranges(self, segments):
        ranges = []

        for seg in segments:
            ranges.append({
                "start": seg[0]["start"],
                "end": seg[-1]["end"]
            })

        return ranges
    
    def ranges_to_cut_points(self,ranges):
        cuts = []

        for i in range(len(ranges) - 1):
            cuts.append(ranges[i]["end"])
            cuts.append(ranges[i+1]["start"])

        return cuts

    def _prepare_output_directory(self, node_state: NodeState, inputs: Dict[str, Any]) -> Path:
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        output_directory = self.server_cache_dir / session_id / artifact_id
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory
