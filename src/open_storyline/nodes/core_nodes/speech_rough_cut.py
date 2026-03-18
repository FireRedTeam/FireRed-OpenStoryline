from typing import Any, Dict
from pathlib import Path

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import SpeechRoughCutInput
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.parse_json import parse_json_list
from open_storyline.utils.ffmpeg_utils import (
    resolve_ffmpeg_executable,
    segment_video_stream_copy_with_ffmpeg,
    VideoSegment,
)
from open_storyline.utils.register import NODE_REGISTRY

CLIP_ID_NUMBER_WIDTH = 4
MILLISECONDS_PER_SECOND = 1000.0
DEFAULT_BUFFER_MS = 100  # buffer in milliseconds for safe cut

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

    async def default_process(self, node_state, inputs: Dict[str, Any]) -> Any:
        return {}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        Main processing function:
        - Calls LLM to get rough cut suggestions
        - Groups sentences by gap threshold
        - Adds buffer and computes cut points
        - Splits video with ffmpeg
        - Calibrates ASR timestamps after deleted segments
        - Returns final clip metadata and updated ASR json
        """
        asr_infos = inputs["asr"].get('asr_infos', [])
        gap_threshold = inputs.get('gap_threshold', 400)
        output_directory = self._prepare_output_directory(node_state, inputs)
        llm = node_state.llm
        rough_cut_jsons, clips, clip_index = [], [], 0

        # Load system prompt for rough cut
        system_prompt = get_prompt("speech_rough_cut.system", lang=node_state.lang)

        for asr_info in asr_infos:
            video_path = asr_info.get('path')
            source_ref = asr_info.get('source_ref', {})
            fps = asr_info.get('fps', 30)

            # Generate user prompt with ASR sentence info
            user_prompt = get_prompt(
                "speech_rough_cut.user",
                lang=node_state.lang,
                asr_sentence_info=asr_info.get("asr_sentence_info", {})
            )

            # Call LLM for rough cut JSON
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
                rough_cut_json = parse_json_list(raw)
            except Exception as e:
                # fallback to original ASR if LLM fails
                node_state.node_summary.warn(f"LLM rough cut failed: {e}")
                rough_cut_json = asr_info.get("asr_sentence_info", [])

            # Group sentences based on gap threshold
            segments_groups = self.group_sentences(rough_cut_json, gap_threshold=gap_threshold)

            # Convert grouped sentences into ranges
            ranges = self.segments_to_ranges(segments_groups)

            # Generate cut points with buffer
            cuts = self.ranges_to_cut_points(ranges, buffer_ms=DEFAULT_BUFFER_MS)
            split_points_seconds = [t / 1000 for t in cuts]

            # Split video using ffmpeg
            ff_segments = segment_video_stream_copy_with_ffmpeg(
                input_video=video_path,
                ffmpeg_executable=self.ffmpeg_executable,
                split_points_seconds=split_points_seconds,
                output_directory=output_directory,
                filename_prefix=f"speech_rough_cut_{clip_index:0{CLIP_ID_NUMBER_WIDTH}d}",
                start_index=len(rough_cut_jsons),
            )

            # Filter segments to keep only the first segment of each pair (i.e., segments[0], segments[2], etc.)
            filtered_segments = [seg for i, seg in enumerate(ff_segments) if i % 2 == 0]

            # Compute deleted ranges and recalibrate ASR timestamps
            deleted_ranges = self.compute_deleted_ranges(filtered_segments)
            rough_cut_json = self.calibrate_asr_times(rough_cut_json, deleted_ranges)
            rough_cut_jsons.append(rough_cut_json)


            # Generate final clip metadata
            for segment in filtered_segments:
                clip_id = self._format_clip_id(clip_index)
                start_ms = max(0, int(round(segment.start_seconds * MILLISECONDS_PER_SECOND)))
                end_ms = max(start_ms, int(round(segment.end_seconds * MILLISECONDS_PER_SECOND)))
                duration_ms = max(0, end_ms - start_ms)
                if duration_ms <= 0:
                    continue

                clips.append({
                    "clip_id": clip_id,
                    "kind": "video",
                    "path": str(segment.path),
                    "fps": fps,
                    "source_ref": {
                        "media_id": source_ref.get("media_id"),
                        "start": start_ms,
                        "end": end_ms,
                        "duration": duration_ms,
                        "height": source_ref.get("height"),
                        "width": source_ref.get("width"),
                    },
                })
                node_state.node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[str(segment.path)])
                clip_index += 1

        return {"clips": clips, "rough_cut_jsons": rough_cut_jsons}

    # --------------------- Sentence Grouping ---------------------
    def group_sentences(self, items, gap_threshold: int = 400):
        """Group sentences into segments by gap threshold (ms)."""
        segments = []
        if not items:
            return segments
        current = [items[0]]
        for i in range(len(items) - 1):
            cur = items[i]
            nxt = items[i + 1]
            gap = nxt["start"] - cur["end"]
            if gap > gap_threshold:
                segments.append(current)
                current = [nxt]
            else:
                current.append(nxt)
        if current:
            segments.append(current)
        return segments

    def segments_to_ranges(self, segments):
        """Convert grouped sentence segments to start/end ranges."""
        return [{"start": seg[0]["start"], "end": seg[-1]["end"]} for seg in segments]

    def ranges_to_cut_points(self, ranges, buffer_ms=100):
        """
        Convert ranges to ffmpeg cut points.
        Adds buffer for safe cuts and prevents overlap.
        """
        cuts = []
        for i in range(len(ranges) - 1):
            end_cut = ranges[i]["end"] + buffer_ms
            start_cut = ranges[i + 1]["start"] - buffer_ms
            # Prevent overlap
            if start_cut < end_cut:
                mid = (start_cut + end_cut) // 2
                end_cut = mid
                start_cut = mid
            cuts.append(end_cut)
            cuts.append(start_cut)
        return cuts

    # --------------------- Time Calibration ---------------------
    def compute_deleted_ranges(self, segments):
        """Compute time ranges that were deleted (gaps between segments)."""
        deleted = []
        prev_end = 0
        for seg in segments:
            start_ms = int(seg.start_seconds * 1000)
            end_ms = int(seg.end_seconds * 1000)
            if start_ms > prev_end:
                deleted.append({"start": prev_end, "end": start_ms})
            prev_end = end_ms
        return deleted

    def calibrate_asr_times(self, rough_cut_json, deleted_ranges):
        """
        Adjust ASR timestamps after deleted ranges.
        New time = original time - total deleted duration before it.
        """
        if not deleted_ranges:
            return rough_cut_json

        # Build prefix sum of deleted durations
        prefix = []
        total = 0
        for r in deleted_ranges:
            prefix.append((r["start"], r["end"], total))
            total += r["end"] - r["start"]
        
        def remap_time(t):
            for start, end, deleted_before in prefix:
                if t < start:
                    return t - deleted_before
                if start <= t <= end:
                    return start - deleted_before  # timestamp falls in deleted segment
            return t - prefix[-1][2]

        new_json = []
        for item in rough_cut_json:
            new_start = remap_time(item["start"])
            new_end = remap_time(item["end"])
            if new_start is None or new_end is None:
                continue
            item["start"] = int(new_start)
            item["end"] = int(new_end)
            new_json.append(item)
        return new_json

    # --------------------- Helpers ---------------------
    def _prepare_output_directory(self, node_state: NodeState, inputs: Dict[str, Any]) -> Path:
        """Create output directory for clips."""
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        output_directory = self.server_cache_dir / session_id / artifact_id
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _format_clip_id(self, clip_index: int) -> str:
        """Generate zero-padded clip ID."""
        return f"clip_{clip_index:0{CLIP_ID_NUMBER_WIDTH}d}"