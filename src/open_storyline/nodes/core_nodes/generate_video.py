import base64
from typing import Dict, Any, List, Union, Tuple, Optional
from pathlib import Path
import os
from io import BytesIO
from PIL import Image
from moviepy import VideoFileClip  # MoviePy 2.x standard import

from open_storyline.utils.register import NODE_REGISTRY
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.client import VisionClientFactory
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import AIGCTransitionInput

def encode_image_to_data_url(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """
    Converts a PIL Image object into a Base64-encoded Data URL.
    
    Args:
        image (Image.Image): The PIL Image instance to be encoded.
        format (str): Image format for encoding ('JPEG', 'PNG', 'WEBP'). Defaults to 'JPEG'.
        quality (int): Encoding quality for JPEG/WEBP (1-100). Higher is better quality but larger size.
        
    Returns:
        str: A complete Data URL string (e.g., "data:image/jpeg;base64,...").
    """
    # 1. Handle mode compatibility
    # JPEG format does not support transparency (RGBA) or palette (P) modes.
    # We must convert these to RGB to avoid "OSError: cannot write mode RGBA as JPEG".
    save_format = format.upper()
    if save_format == "JPEG":
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        mime_type = "image/jpeg"
    elif save_format == "PNG":
        mime_type = "image/png"
    else:
        mime_type = f"image/{save_format.lower()}"

    # 2. Save image to an in-memory byte buffer
    # This avoids slow disk I/O and temporary file management.
    buffered = BytesIO()
    image.save(
        buffered, 
        format=save_format, 
        quality=quality if save_format in ("JPEG", "WEBP") else None
    )
    
    # 3. Encode binary data to Base64 string
    # getvalue() retrieves the bytes from the buffer, b64encode converts to base64 bytes, 
    # and decode('utf-8') converts it to a standard Python string.
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 4. Format and return the standard Data URL pattern
    return f"data:{mime_type};base64,{base64_str}"


@NODE_REGISTRY.register()
class FirstLastFrameToVideoNode(BaseNode):
    meta = NodeMeta(
        name="aigc_flf2v",
        description="Generate transition videos: Create transition videos for grouped video clips, generating an appropriate transition from the last frame of the previous clip to the first frame of the next clip based on user requirements.",
        node_id="aigc_flf2v",
        node_kind="aigc_flf2v",
        require_prior_kind=["split_shots", "group_clips"],
    )
    input_schema = AIGCTransitionInput
    VIDEO_EXTS = {
        ".mp4", ".mov", ".mkv", ".avi"
    }
    IMAGE_EXTS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        group_clips = inputs.get("group_clips", {})
        split_shots = inputs.get("split_shots", {})
        groups = group_clips.get("groups", [])
        clips = split_shots.get('clips', [])
        provider = inputs.get("provider", "minimax")
        api_key = inputs.get("api_key", "")
        model_name = inputs.get("model_name", "MiniMax-Hailuo-02")

        clip_map = {clip['clip_id']: clip for clip in clips}

        node_cache_dir = self._prepare_output_directory(node_state)

        transitions_result = []
        for i in range(len(groups) - 1):
            prev_group = groups[i]
            next_group = groups[i+1]

            last_clip_id_of_prev = prev_group['clip_ids'][-1]
            first_clip_id_of_next = next_group['clip_ids'][0]
            
            prev_clip = clip_map.get(last_clip_id_of_prev)
            next_clip = clip_map.get(first_clip_id_of_next)
            
            if not prev_clip or not next_clip:
                node_state.node_summary.add_warning(f"Clips <{prev_group['clip_ids']}, {next_group['clip_ids']}> not found in split_shots; skipping transition generation.")
                transitions_result.append({})
                continue
            
            prev_frames = self._load_clip(prev_clip.get('path'))
            next_frames = self._load_clip(next_clip.get('path'))
            prev_clip_type = 'V' if len(prev_frames) > 1 else 'I'
            next_clip_type = 'V' if len(next_frames) > 1 else 'I'

            first_frame = prev_frames[-1]
            last_frame = next_frames[0]

            aligned_first_frame, aligned_last_frame, first_frame_meta, last_frame_meta = self._preprocess_first_last_frame(first_frame, last_frame)
            
            gen_video_path, response = self._generate_video(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                prompt="以一镜到底的方式拍摄，场景丝滑过渡，摄像机快速推进",
                first_frame_data_url=encode_image_to_data_url(aligned_first_frame),
                last_frame_data_url=encode_image_to_data_url(aligned_last_frame),
                output_dir=node_cache_dir
            )
            transitions_result.append({
                'from_group': prev_group['group_id'],
                'to_group': next_group['group_id'],
                'from_clip': last_clip_id_of_prev,
                'to_clip': first_clip_id_of_next,
                'transition_video_path': gen_video_path,
                'transition_type': f"{prev_clip_type}{next_clip_type}",
                'extra_info': {
                    "first_frame_meta": first_frame_meta,
                    "last_frame_meta": last_frame_meta,
                    "raw_request_result": response,
                }
            })
        return {"aigc_flf2v": transitions_result}
    
    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        return await self.process(node_state, inputs)
    
    def _load_clip(
        self,
        image_or_video_path: Union[str, Path]
    ) -> List[Image]:
        """
        Loads media frames using PIL for images and MoviePy 2.2.1 for videos.
        
        Args:
            image_or_video_path: Path to the media file.
            
        Returns:
            List[Image.Image]: A list of frames as PIL Image objects in RGB mode.
        """
        path = Path(image_or_video_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        ext = path.suffix.lower()
        frames: List[Image.Image] = []

        # --- Process Images ---
        if ext in self.IMAGE_EXTS:
            try:
                # Open with PIL and force RGB mode
                with Image.open(path) as img:
                    frames.append(img.convert("RGB"))
            except Exception as e:
                raise RuntimeError(f"Failed to load image via PIL: {path}. Error: {e}")

        # --- Process Videos ---
        elif ext in self.VIDEO_EXTS:
            try:
                # VideoFileClip in v2.x works best within a context manager
                with VideoFileClip(str(path)) as clip:
                    # iter_frames yields RGB numpy arrays by default
                    for frame_array in clip.iter_frames():
                        # Image.fromarray converts the numpy array (RGB) to a PIL object
                        frames.append(Image.fromarray(frame_array))
            except Exception as e:
                raise RuntimeError(f"MoviePy failed to decode video: {path}. Error: {e}")

            if not frames:
                raise RuntimeError(f"Extraction resulted in an empty frame list for: {path}")

        else:
            raise ValueError(f"File extension {ext} is not supported by this processor.")

        return frames

    def _preprocess_first_last_frame(
        self,
        first_frame: Image.Image,
        last_frame: Image.Image,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image, Dict[str, Any], Dict[str, Any]]:
        """
        Normalizes color modes and aligns both frames to a target resolution.
        
        Args:
            first_frame (Image.Image): The starting frame.
            last_frame (Image.Image): The ending frame.
            target_width (Optional[int]): Desired output width. Defaults to first_frame width.
            target_height (Optional[int]): Desired output height. Defaults to first_frame height.
            
        Returns:
            Tuple[Image.Image, Image.Image, Dict, Dict]: 
                (Aligned First Frame, Aligned Last Frame, First Frame Meta, Last Frame Meta)
        """

        # 1. Determine Target Resolution
        # Use provided dimensions or fallback to the first frame's original size
        if target_width and target_height:
            target_size = (target_width, target_height)
        else:
            target_size = first_frame.size

        # 2. Color Mode Normalization (RGB)
        # Required for API compatibility (removes Alpha/transparency channels)
        def normalize_img(img: Image.Image) -> Image.Image:
            return img.convert("RGB") if img.mode != "RGB" else img

        first_frame = normalize_img(first_frame)
        last_frame = normalize_img(last_frame)

        # 3. Helper for Resizing and Metadata Logging
        def process_frame(img: Image.Image, role: str) -> Tuple[Image.Image, Dict[str, Any]]:
            meta = {
                "role": role,
                "original_size": img.size,
                "target_size": target_size,
                "transformations": []
            }
            
            if img.size != target_size:
                # High-quality resampling for AI generation stability
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                meta["transformations"].append({
                    "type": "resize",
                    "method": "LANCZOS",
                    "target": target_size
                })
            else:
                meta["transformations"].append({"type": "none", "reason": "already_correct_size"})
                
            return img, meta

        # 4. Execute Processing for both frames
        aligned_first_frame, first_frame_meta = process_frame(first_frame, "first")
        aligned_last_frame, last_frame_meta = process_frame(last_frame, "last")

        # Final pixel-perfect validation
        assert aligned_first_frame.size == aligned_last_frame.size == target_size

        return aligned_first_frame, aligned_last_frame, first_frame_meta, last_frame_meta

    def _generate_video(
        self,
        provider,
        api_key,
        model_name,
        prompt,
        first_frame_data_url,
        last_frame_data_url,
        output_dir
    ) -> Tuple[str, Dict[str, Any]]:
        client = VisionClientFactory.create(
            provider=provider,
            api_key=api_key,
        )
        
        gen_video_path, response = client.generate(
            task_type="video_generation",
            model=model_name,
            prompt=prompt,
            first_frame=first_frame_data_url,
            last_frame=last_frame_data_url,
            resolution="480P",
            duration=5,
            prompt_optimizer=True
            output_dir=output_dir,
        )
        
        return gen_video_path, response
