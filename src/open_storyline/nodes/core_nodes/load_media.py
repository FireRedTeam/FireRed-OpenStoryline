from typing import Any, Dict, Optional, ClassVar, Type
from pydantic import BaseModel
import traceback
from collections import Counter
from pathlib import Path
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos


from open_storyline.nodes.core_nodes.base_node import NodeMeta, BaseNode
from open_storyline.nodes.node_schema import LoadMediaInput, LoadMediaOutput
from open_storyline.nodes.node_state import NodeState
from open_storyline.utils.util import get_video_rotation
from open_storyline.utils.register import NODE_REGISTRY


VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi"
}
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"
}

def _convert_heic_to_jpg(heic_path: Path) -> Path:
    """Convert HEIC image to JPG format and return the new path."""
    import pillow_heif  # noqa # pylint: disable=unused-import
    pillow_heif.register_heif_opener()
    from PIL import Image
    
    jpg_path = heic_path.with_suffix('.jpg')
    with Image.open(heic_path) as img:
        img.convert('RGB').save(jpg_path, 'JPEG', quality=95)
    return jpg_path

def _image_metadata_from_path(path: Path) -> dict[str, Any]:
    import pillow_heif  # noqa # pylint: disable=unused-import
    pillow_heif.register_heif_opener()
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        try:
            img2 = ImageOps.exif_transpose(img)
            w, h = img2.size
        except Exception:
            w, h = img.size
        
    return {
        "width": int(w),
        "height": int(h),
    }


import av
from fractions import Fraction
from typing import Any, Optional
from pathlib import Path


def _video_metadata_from_path(
    path: Path,
    *,
    round_duration_ndigits: Optional[int] = 3,
) -> dict[str, Any]:
    
    container = av.open(str(path))

    # 找第一个视频流
    video_stream = next(
        (s for s in container.streams if s.type == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found: {path}")

    # ---------- duration ----------
    duration_sec = 0.0

    if container.duration is not None:
        # container.duration 单位是 microseconds
        duration_sec = container.duration / 1_000_000
    elif video_stream.duration is not None and video_stream.time_base is not None:
        duration_sec = float(video_stream.duration * video_stream.time_base)

    if round_duration_ndigits is not None:
        duration_sec = round(duration_sec, round_duration_ndigits)

    # ---------- width / height / rotation ----------
    w = int(video_stream.codec_context.width or 0)
    h = int(video_stream.codec_context.height or 0)

    rotation = get_video_rotation(path)

    if abs(rotation) in (90, 270):
        w, h = h, w

    # ---------- fps ----------
    fps = 0.0
    if video_stream.average_rate:
        fps = float(video_stream.average_rate)
    elif video_stream.base_rate:
        fps = float(video_stream.base_rate)

    # ---------- audio ----------
    audio_stream = next(
        (s for s in container.streams if s.type == "audio"),
        None,
    )

    has_audio = audio_stream is not None
    audio_sample_rate_hz = int(audio_stream.rate) if audio_stream and audio_stream.rate else 0

    container.close()

    return {
        "duration": int(duration_sec * 1000),  # ms
        "width": w,
        "height": h,
        "fps": fps,
        "has_audio": has_audio,
        "audio_sample_rate_hz": audio_sample_rate_hz,
    }


@NODE_REGISTRY.register()
class LoadMediaNode(BaseNode):
    meta = NodeMeta(
        name="load_media",
        description="Loads and indexes input media. Entry point with no dependencies; required by all downstream operations",
        node_id="load_media",
        node_kind="load_media",
        next_available_node=['split_shots', 'split_shots_pro'],
    )
    input_schema: ClassVar[Type[BaseModel]] = LoadMediaInput
    # output_schema:  ClassVar[Type[BaseModel]] = LoadMediaOutput

    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return await self.process(node_state, inputs)

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_media = inputs.get('inputs', [])
        
        media_idx = 1
        media = []
        for enc_media in input_media:
            path = Path(enc_media['path'])
            suffix = path.suffix.lower()

            if suffix in VIDEO_EXTS:
                metadata = _video_metadata_from_path(path)
                media_type = "video"
            elif suffix in IMAGE_EXTS:
                if suffix == ".heic":
                    # Convert HEIC to JPG for model compatibility
                    path = _convert_heic_to_jpg(path)
                metadata = _image_metadata_from_path(path)
                media_type = "image"
            else:
                node_state.node_summary.info_for_user(f"[Node {self.meta.node_id}] Skipping unsupported file type `{enc_media['orig_path']}` ")
                continue

            media.append(
                {
                    "media_id": f"media_{media_idx:04d}",
                    "path": path,
                    "media_type": media_type,
                    "metadata": metadata,
                    "orig_path": enc_media['orig_path'],
                    "orig_md5": enc_media['orig_md5'],
                }
            )
            node_state.node_summary.info_for_user(f"Added media_{media_idx:04d}: ({media_type})")
            media_idx += 1

        c = Counter(
            (a.get("media_type") or "").strip().lower()
            for a in media
            if isinstance(a, dict)
        )   

        node_state.node_summary.info_for_user(f"[Node {self.meta.node_id}] Media indexing completed successfully: {c.get('video', 0)} video(s), {c.get('image', 0)} image(s)",)

        return {"media": media}