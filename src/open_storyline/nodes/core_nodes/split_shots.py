from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import csv
import functools
import os
import shutil
import subprocess
import math


import numpy as np

from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_schema import SplitShotsInput
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_summary import NodeSummary
from open_storyline.utils.ffmpeg_utils import (
    resolve_ffmpeg_executable,
    read_video_frames_as_rgb24,
    segment_video_stream_copy_with_ffmpeg,
    VideoSegment,
)
from open_storyline.utils.register import NODE_REGISTRY

MODEL_CACHE_MAXSIZE = 4

# TransNetV2 expects frames with shape [..., 27, 48, 3] in this implementation.
TRANSNETV2_INPUT_HEIGHT = 27
TRANSNETV2_INPUT_WIDTH = 48
TRANSNETV2_INPUT_CHANNELS = 3

DEFAULT_SCENE_DETECTION_FRAMES_PER_SECOND = 25
DEFAULT_SCENE_DETECTION_THRESHOLD = 0.5
DEFAULT_SPLIT_POINT_MINIMUM_GAP_SECONDS = 1e-3

DEFAULT_MIN_SHOT_DURATION_MILLISECONDS = 1000
DEFAULT_MAX_SHOT_DURATION_MILLISECONDS = 30000

CLIP_ID_NUMBER_WIDTH = 4
MILLISECONDS_PER_SECOND = 1000.0

COPY_VIDEO_WHEN_NO_SPLIT = False

# =========================
# Model / ffmpeg helpers
# =========================

@functools.lru_cache(maxsize=MODEL_CACHE_MAXSIZE)
def load_transnetv2_model_cached(weight_path: str, device: str = "auto"):
    """
    Load TransNetV2 model with LRU cache. Suitable for service mode.
    """
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device=device)
    model.eval()

    state_dict = torch.load(weight_path, map_location=model.device)
    model.load_state_dict(state_dict)
    return model


def detect_scenes_with_transnetv2_without_proxy(
    model: Any,
    input_video: Path,
    ffmpeg_executable: str,
    *,
    frames_per_second: int = DEFAULT_SCENE_DETECTION_FRAMES_PER_SECOND,
    threshold: float = DEFAULT_SCENE_DETECTION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    No proxy file:
    ffmpeg -> frames (uint8) -> model.predict_raw -> predictions_to_scenes_with_data
    """
    import torch

    frames_numpy = read_video_frames_as_rgb24(
        input_video,
        ffmpeg_executable,
        frames_per_second=frames_per_second,
        target_width=TRANSNETV2_INPUT_WIDTH,
        target_height=TRANSNETV2_INPUT_HEIGHT,
    )

    if frames_numpy.size == 0 or frames_numpy.shape[0] == 0:
        return []

    frames_tensor = torch.from_numpy(frames_numpy)          # [T, H, W, 3], uint8
    frames_tensor = frames_tensor.unsqueeze(0).contiguous() # [1, T, H, W, 3]

    model_device = getattr(model, "device", None)
    if model_device is not None:
        frames_tensor = frames_tensor.to(model_device, non_blocking=True)

    with torch.inference_mode():
        single_prediction, _all_prediction = model.predict_raw(frames_tensor)

    prediction = single_prediction.detach().cpu().numpy()
    prediction = np.squeeze(prediction)
    if prediction.ndim != 1:
        prediction = prediction.reshape(-1)

    scenes = model.predictions_to_scenes_with_data(
        prediction,
        fps=float(frames_per_second),
        threshold=float(threshold),
    )
    return scenes


def convert_scenes_to_split_points_seconds(
    scenes: List[Dict[str, Any]],
    *,
    minimum_gap_seconds: float = DEFAULT_SPLIT_POINT_MINIMUM_GAP_SECONDS,
) -> List[float]:
    """
    Convert TransNetV2 scenes to ffmpeg segment split points.
    split points: [t1, t2, ...] means segments [0,t1], [t1,t2], ..., [last,end]
    """
    end_times: List[float] = []
    last_end_time = 0.0

    for scene in scenes:
        try:
            end_time = float(scene.get("end_time", 0.0))
        except Exception:
            continue

        if end_time > last_end_time + minimum_gap_seconds:
            end_times.append(end_time)
            last_end_time = end_time

    # If <=1 scene, don't split
    if len(end_times) <= 1:
        return []

    # Remove the last end_time (usually video end)
    return end_times[:-1]

def enforce_shot_duration_constraints_on_split_points_seconds(
    split_points_seconds: List[float],
    *,
    total_duration_milliseconds: int,
    min_shot_duration_milliseconds: Optional[int],
    max_shot_duration_milliseconds: Optional[int],
) -> List[float]:
    """
    对“切分点(split points)”施加 min/max 时长约束（单位 ms）：
    1) 若某段 < min：通过删除相应切分点，把它与相邻段拼接（优先向后拼；尾段太短则向前拼）。
    2) 若某段 > max：在该段内部强制均匀切分（允许镜头内仍有镜头切换）。

    注意：这里在“调用 ffmpeg 前”调整 split points，从而避免切完后再做文件拼接（性能更好）。
    """
    duration_ms = int(total_duration_milliseconds)

    def _normalize_optional_ms(value: Optional[int], key_name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{key_name} must be int milliseconds, got bool")
        try:
            value_int = int(value)
        except Exception as exc:
            raise ValueError(f"{key_name} must be int milliseconds, got {value!r}") from exc
        if value_int < 0:
            raise ValueError(f"{key_name} must be >= 0, got {value_int}")
        if value_int == 0:
            return None
        return value_int

    min_ms = _normalize_optional_ms(min_shot_duration_milliseconds, "min_shot_duration")
    max_ms = _normalize_optional_ms(max_shot_duration_milliseconds, "max_shot_duration")

    if min_ms is not None and max_ms is not None and min_ms > max_ms:
        raise ValueError(f"min_shot_duration ({min_ms}) cannot be greater than max_shot_duration ({max_ms}).")

    # seconds -> milliseconds cut points
    cut_points_ms = [
        int(round(point_seconds * MILLISECONDS_PER_SECOND))
        for point_seconds in split_points_seconds
    ]
    cut_points_ms = sorted({c for c in cut_points_ms if 0 < c < duration_ms})

    # ---------- Step 1: merge short segments (< min) by removing cut points ----------
    if min_ms is not None and cut_points_ms:
        merged_cut_points: List[int] = []
        segment_start_ms = 0

        for cut_ms in cut_points_ms:
            segment_length_ms = cut_ms - segment_start_ms
            if segment_length_ms < min_ms:
                continue
            merged_cut_points.append(cut_ms)
            segment_start_ms = cut_ms

        if merged_cut_points and (duration_ms - merged_cut_points[-1] < min_ms):
            merged_cut_points.pop()

        cut_points_ms = merged_cut_points

    # ---------- Step 2: split long segments (> max) by inserting internal cut points ----------
    if max_ms is not None and max_ms > 0:
        cuts_set = set(cut_points_ms)
        boundaries = [0] + cut_points_ms + [duration_ms]

        for segment_start_ms, segment_end_ms in zip(boundaries[:-1], boundaries[1:]):
            segment_length_ms = segment_end_ms - segment_start_ms
            if segment_length_ms <= max_ms:
                continue

            # 最少切成 pieces 段，保证每段 <= max_ms
            pieces = int(math.ceil(segment_length_ms / max_ms))
            if pieces <= 1:
                continue

            # 均匀分配（整数 ms），尽量避免“最后剩一小段”
            base = segment_length_ms // pieces
            remainder = segment_length_ms % pieces  # 前 remainder 段多 1ms

            current = segment_start_ms
            for i in range(pieces - 1):
                piece_len = base + 1 if i < remainder else base
                current += piece_len
                if segment_start_ms < current < segment_end_ms:
                    cuts_set.add(current)

        cut_points_ms = sorted(c for c in cuts_set if 0 < c < duration_ms)

    # milliseconds -> seconds
    return [cut_ms / MILLISECONDS_PER_SECOND for cut_ms in cut_points_ms]


# =========================
# Node implementation
# =========================

@NODE_REGISTRY.register()
class SplitShotsNode(BaseNode):
    meta = NodeMeta(
        name="split_shots",
        description="Segment input video based on shot boundary detection",
        node_id="split_shots",
        node_kind="split_shots",
        require_prior_kind=["load_media"],
        default_require_prior_kind=["load_media"],
        next_available_node=["understand_clips"],
    )
    input_schema = SplitShotsInput

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.transnetv2_model = load_transnetv2_model_cached(
            str(self.server_cfg.split_shots.transnet_weights),
            device=self.server_cfg.split_shots.transnet_device,
        )
        self.ffmpeg_executable = resolve_ffmpeg_executable()

    # -------------------------
    # Public entrypoints
    # -------------------------

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        Default behavior: do NOT split shots, just pass-through paths.
        (Optimization) No re-save / no re-encode.
        """
        output_directory = self._prepare_output_directory(node_state, inputs)
        media = self._extract_media(inputs)

        clips: List[Dict[str, Any]] = []
        clip_index = 1

        for media_item in media:
            clip = self._build_clip_without_splitting(media_item=media_item, clip_index=clip_index, node_summary=node_state.node_summary)
            clips.append(clip)
            clip_index += 1

        node_state.node_summary.info_for_user("Shot splitting skipped")
        return {"clips": clips}

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        """
        Split shots using TransNetV2 + ffmpeg segment copy.
        """
        output_directory = self._prepare_output_directory(node_state, inputs)
        media = self._extract_media(inputs)

        clips: List[Dict[str, Any]] = []
        clip_index = 1

        min_shot_duration_milliseconds = inputs.get("min_shot_duration", DEFAULT_MIN_SHOT_DURATION_MILLISECONDS)
        max_shot_duration_milliseconds = inputs.get("max_shot_duration", DEFAULT_MAX_SHOT_DURATION_MILLISECONDS)

        if min_shot_duration_milliseconds > max_shot_duration_milliseconds:
            min_shot_duration_milliseconds, max_shot_duration_milliseconds = DEFAULT_MIN_SHOT_DURATION_MILLISECONDS, DEFAULT_MAX_SHOT_DURATION_MILLISECONDS
            node_state.node_summary.add_warning(
                f"min_shot_duration_milliseconds ({min_shot_duration_milliseconds}) cannot greater than max_shot_duration_milliseconds ({max_shot_duration_milliseconds})"
                f"using default config {DEFAULT_MIN_SHOT_DURATION_MILLISECONDS} and ({DEFAULT_MAX_SHOT_DURATION_MILLISECONDS})",
                artifact_id = node_state.artifact_id,
            )
        
        if min_shot_duration_milliseconds < DEFAULT_MIN_SHOT_DURATION_MILLISECONDS:
            min_shot_duration_milliseconds = DEFAULT_MIN_SHOT_DURATION_MILLISECONDS
            node_state.node_summary.add_warning(
                f"min_shot_duration_milliseconds ({min_shot_duration_milliseconds}) too small"
                f"using default config {DEFAULT_MIN_SHOT_DURATION_MILLISECONDS}",
                artifact_id = node_state.artifact_id,
            )
        
        if max_shot_duration_milliseconds > DEFAULT_MAX_SHOT_DURATION_MILLISECONDS:
            max_shot_duration_milliseconds = DEFAULT_MAX_SHOT_DURATION_MILLISECONDS
            node_state.node_summary.add_warning(
                f"max_shot_duration_milliseconds ({max_shot_duration_milliseconds}) too great"
                f"using default config {DEFAULT_MAX_SHOT_DURATION_MILLISECONDS}",
                artifact_id = node_state.artifact_id,
            )

        for media_item in media:
            new_clips, clip_index = self._process_single_media_item(
                media_item=media_item,
                output_directory=output_directory,
                starting_clip_index=clip_index,
                node_summary=node_state.node_summary,
                min_shot_duration_milliseconds=min_shot_duration_milliseconds,
                max_shot_duration_milliseconds=max_shot_duration_milliseconds,
            )
            clips.extend(new_clips)

        node_state.node_summary.info_for_user(
            f"{self.meta.node_id} executed successfully, output clips count: {len(clips)}"
        )
        return {"clips": clips}

    # -------------------------
    # Internal helpers
    # -------------------------

    def _prepare_output_directory(self, node_state: NodeState, inputs: Dict[str, Any]) -> Path:
        artifact_id = node_state.artifact_id
        session_id = node_state.session_id
        output_directory = self.server_cache_dir / session_id / artifact_id
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _extract_media(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        return (inputs.get("load_media") or {}).get("media", []) or []

    def _format_clip_id(self, clip_index: int) -> str:
        return f"clip_{clip_index:0{CLIP_ID_NUMBER_WIDTH}d}"

    def _require_media_id(self, media_item: Dict[str, Any]) -> str:
        media_id = media_item.get("media_id")
        if not media_id:
            raise ValueError(f"media_item missing required field 'media_id': {media_item}")
        return str(media_id)

    def _require_media_type(self, media_item: Dict[str, Any]) -> str:
        media_type = media_item.get("media_type")
        if not media_type:
            raise ValueError(f"media_item missing required field 'media_type': {media_item}")
        return str(media_type)

    def _require_video_metadata(self, media_id: str, media_item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = media_item.get("metadata") or {}
        if "duration" not in metadata:
            raise ValueError(f"video media_id={media_id} missing metadata.duration")
        return metadata

    def _parse_duration_milliseconds(self, media_id: str, metadata: Dict[str, Any]) -> int:
        try:
            duration_milliseconds = int(metadata["duration"])
        except (TypeError, ValueError):
            raise ValueError(f"video media_id={media_id} has invalid metadata.duration: {metadata.get('duration')!r}")

        if duration_milliseconds < 0:
            raise ValueError(f"video media_id={media_id} has negative duration: {duration_milliseconds}")
        return duration_milliseconds

    def _require_path(self, media_id: str, media_item: Dict[str, Any], *, field_name: str) -> str:
        path_value = media_item.get(field_name)
        if not path_value:
            raise ValueError(f"media_id={media_id} missing required field {field_name!r}")
        return str(path_value)

    def _build_clip_without_splitting(self, media_item: Dict[str, Any], clip_index: int, node_summary: NodeSummary) -> Dict[str, Any]:
        """
        Build a single clip without cutting:
        - image: use orig_path (or fallback to path)
        - video: use path (no re-save)
        """
        media_id = self._require_media_id(media_item)
        media_type = self._require_media_type(media_item)
        clip_id = self._format_clip_id(clip_index)

        if media_type == "image":
            image_path = media_item.get("orig_path") or media_item.get("path")
            if not image_path:
                raise ValueError(f"image media_id={media_id} missing 'orig_path'/'path'")
            node_summary.info_for_user(f"{clip_id} 分割完成", preview_urls=[image_path])
            return {
                "clip_id": clip_id,
                "kind": "image",
                "path": image_path,
                "source_ref": {
                    "media_id": media_id,
                    "height": media_item.get("metadata", {}).get("height"),
                    "width": media_item.get("metadata", {}).get("width"),
                },
            }

        if media_type != "video":
            raise ValueError(f"unsupported media_type {media_type!r} for media_id={media_id}")

        metadata = self._require_video_metadata(media_id, media_item)
        duration_milliseconds = self._parse_duration_milliseconds(media_id, metadata)
        video_path = self._require_path(media_id, media_item, field_name="path")

        node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[video_path])
        return {
            "clip_id": clip_id,
            "kind": "video",
            "path": video_path,
            "fps": metadata.get("fps"),
            "source_ref": {
                "media_id": media_id,
                "start": 0,
                "end": duration_milliseconds,
                "duration": duration_milliseconds,
                "height": metadata.get("height"),
                "width": metadata.get("width"),
            },
        }

    def _process_single_media_item(
        self,
        *,
        media_item: Dict[str, Any],
        output_directory: Path,
        starting_clip_index: int,
        node_summary: NodeSummary,
        min_shot_duration_milliseconds: int,
        max_shot_duration_milliseconds: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Return: (clips_generated, next_clip_index)
        """
        media_id = self._require_media_id(media_item)
        media_type = self._require_media_type(media_item)

        if media_type == "image":
            clip_id = self._format_clip_id(starting_clip_index)
            image_path = media_item.get("orig_path") or media_item.get("path")
            if not image_path:
                raise ValueError(f"image media_id={media_id} missing 'orig_path'/'path'")

            node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[image_path])
            clip = {
                "clip_id": clip_id,
                "kind": "image",
                "path": image_path,
                "source_ref": {
                    "media_id": media_id,
                    "height": media_item.get("metadata", {}).get("height"),
                    "width": media_item.get("metadata", {}).get("width"),
                },
            }
            return [clip], starting_clip_index + 1

        if media_type != "video":
            raise ValueError(f"unsupported media_type {media_type!r} for media_id={media_id}")

        video_clips, next_index = self._process_video_media_item(
            media_id=media_id,
            media_item=media_item,
            output_directory=output_directory,
            starting_clip_index=starting_clip_index,
            node_summary=node_summary,
            min_shot_duration_milliseconds=min_shot_duration_milliseconds,
            max_shot_duration_milliseconds=max_shot_duration_milliseconds,
        )
        return video_clips, next_index

    def _process_video_media_item(
        self,
        *,
        media_id: str,
        media_item: Dict[str, Any],
        output_directory: Path,
        starting_clip_index: int,
        node_summary: NodeSummary,
        min_shot_duration_milliseconds: int,
        max_shot_duration_milliseconds: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        metadata = self._require_video_metadata(media_id, media_item)
        duration_milliseconds = self._parse_duration_milliseconds(media_id, metadata)

        input_video_path = Path(self._require_path(media_id, media_item, field_name="path")).expanduser()

        # If the media itself is shorter than min_shot_duration: skip segmentation and concatenation entirely
        if min_shot_duration_milliseconds is not None and duration_milliseconds < min_shot_duration_milliseconds:
            clip_id = self._format_clip_id(starting_clip_index)
            node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[str(input_video_path)])
            clip = {
                "clip_id": clip_id,
                "kind": "video",
                "path": str(input_video_path),
                "fps": metadata.get("fps"),
                "source_ref": {
                    "media_id": media_id,
                    "start": 0,
                    "end": duration_milliseconds,
                    "duration": duration_milliseconds,
                    "height": metadata.get("height"),
                    "width": metadata.get("width"),
                },
            }
            return [clip], starting_clip_index + 1

        # 1) Detect scenes
        scenes = detect_scenes_with_transnetv2_without_proxy(
            self.transnetv2_model,
            input_video_path,
            self.ffmpeg_executable,
            frames_per_second=DEFAULT_SCENE_DETECTION_FRAMES_PER_SECOND,
            threshold=DEFAULT_SCENE_DETECTION_THRESHOLD,
        )
        split_points_seconds = convert_scenes_to_split_points_seconds(scenes)

        split_points_seconds = enforce_shot_duration_constraints_on_split_points_seconds(
            split_points_seconds,
            total_duration_milliseconds=duration_milliseconds,
            min_shot_duration_milliseconds=min_shot_duration_milliseconds,
            max_shot_duration_milliseconds=max_shot_duration_milliseconds,
        )

        # 2) If no split needed, optionally skip copying
        if not split_points_seconds and not COPY_VIDEO_WHEN_NO_SPLIT:
            clip_id = self._format_clip_id(starting_clip_index)
            node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[str(input_video_path)])
            clip = {
                "clip_id": clip_id,
                "kind": "video",
                "path": str(input_video_path),
                "fps": metadata.get("fps"),
                "source_ref": {
                    "media_id": media_id,
                    "start": 0,
                    "end": duration_milliseconds,
                    "duration": duration_milliseconds,
                    "height": metadata.get("height"),
                    "width": metadata.get("width"),
                },
            }
            return [clip], starting_clip_index + 1

        # 3) Segment by ffmpeg (-c copy)
        filename_prefix = "clip"
        segments = segment_video_stream_copy_with_ffmpeg(
            input_video=input_video_path,
            ffmpeg_executable=self.ffmpeg_executable,
            split_points_seconds=split_points_seconds,
            output_directory=output_directory,
            filename_prefix=filename_prefix,
            start_index=starting_clip_index,
        )

        # 4) Build clip list
        clips: List[Dict[str, Any]] = []
        clip_index = starting_clip_index

        for segment in segments:
            clip_id = self._format_clip_id(clip_index)

            if segment.end_seconds < 0:
                start_milliseconds = 0
                end_milliseconds = duration_milliseconds
            else:
                start_milliseconds = max(0, int(round(segment.start_seconds * MILLISECONDS_PER_SECOND)))
                end_milliseconds = max(start_milliseconds, int(round(segment.end_seconds * MILLISECONDS_PER_SECOND)))

            segment_duration_milliseconds = max(0, end_milliseconds - start_milliseconds)
            if segment_duration_milliseconds <= 0:
                continue

            output_path_string = str(segment.path)
            node_summary.info_for_user(f"{clip_id} split successfully", preview_urls=[output_path_string])

            clips.append(
                {
                    "clip_id": clip_id,
                    "kind": "video",
                    "path": output_path_string,
                    "fps": metadata.get("fps"),
                    "source_ref": {
                        "media_id": media_id,
                        "start": start_milliseconds,
                        "end": end_milliseconds,
                        "duration": segment_duration_milliseconds,
                        "height": metadata.get("height"),
                        "width": metadata.get("width"),
                    },
                }
            )
            clip_index += 1

        return clips, clip_index
