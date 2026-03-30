import base64
from typing import Dict, Any, List, Union, Tuple, Optional
from pathlib import Path
import os
from io import BytesIO
from PIL import Image, ImageOps
from moviepy import VideoFileClip  # MoviePy 2.x standard import

from open_storyline.utils.register import NODE_REGISTRY
from open_storyline.utils.prompts import get_prompt
from open_storyline.utils.ai_transition_cancel import is_ai_transition_cancelled
from open_storyline.utils.ai_transition_client import VisionClientFactory
from open_storyline.nodes.core_nodes.base_node import BaseNode, NodeMeta
from open_storyline.nodes.node_state import NodeState
from open_storyline.nodes.node_schema import GenerateAITransitionInput

def encode_image_to_data_url(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 85,
    max_long_edge: Optional[int] = None,
) -> str:
    """
    Converts a PIL Image object into a Base64-encoded Data URL.
    
    Args:
        image (Image.Image): The PIL Image instance to be encoded.
        format (str): Image format for encoding ('JPEG', 'PNG', 'WEBP'). Defaults to 'JPEG'.
        quality (int): Encoding quality for JPEG/WEBP (1-100). Higher is better quality but larger size.
        max_long_edge (Optional[int]): If provided, downsample the image so its long edge
            does not exceed this value while preserving aspect ratio.
        
    Returns:
        str: A complete Data URL string (e.g., "data:image/jpeg;base64,...").
    """
    # 1. Optionally downsample the image to reduce payload size.
    if max_long_edge and max_long_edge > 0:
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    # 2. Handle mode compatibility
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

    # 3. Save image to an in-memory byte buffer
    # This avoids slow disk I/O and temporary file management.
    buffered = BytesIO()
    image.save(
        buffered, 
        format=save_format, 
        quality=quality if save_format in ("JPEG", "WEBP") else None
    )
    
    # 4. Encode binary data to Base64 string
    # getvalue() retrieves the bytes from the buffer, b64encode converts to base64 bytes, 
    # and decode('utf-8') converts it to a standard Python string.
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 5. Format and return the standard Data URL pattern
    return f"data:{mime_type};base64,{base64_str}"


@NODE_REGISTRY.register()
class GenerateAITransitionNode(BaseNode):
    meta = NodeMeta(
        name="generate_ai_transition",
        description="Generate transition videos: Create transition videos for grouped video clips, generating an appropriate transition from the last frame of the previous clip to the first frame of the next clip based on user requirements.",
        node_id="generate_ai_transition",
        node_kind="generate_ai_transition",
        require_prior_kind=["split_shots", "group_clips"],
        default_require_prior_kind=['group_clips'],
        next_available_node=["generate_script"],
    )
    input_schema = GenerateAITransitionInput
    VIDEO_EXTS = {
        ".mp4", ".mov", ".mkv", ".avi"
    }
    IMAGE_EXTS = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }

    DEFAULT_TRANSITION_DURATION = 5
    SECOND_TO_MILLISECOND = 1000
    MAX_ASPECT_RATIO_FACTOR = 1.1
    def _raise_if_cancelled(self, node_state: NodeState) -> None:
        if is_ai_transition_cancelled(self.server_cache_dir, node_state.session_id):
            raise RuntimeError("generate_ai_transition cancelled by user")

    async def process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        group_clips = inputs.get("group_clips", {})
        split_shots = inputs.get("split_shots", {})
        groups = group_clips.get("groups", [])
        clips = split_shots.get('clips', [])

        runtime_cfg = self._resolve_ai_transition_runtime_cfg(inputs)
        provider = runtime_cfg["provider"]
        api_key = runtime_cfg["api_key"]
        model_name = runtime_cfg["model_name"]
        transition_duration = inputs.get("duration")
        resolution = inputs.get("resolution")
        user_request = inputs.get("user_request", "以一镜到底的方式拍摄，场景丝滑过渡")

        clip_map = {clip['clip_id']: clip for clip in clips}

        node_cache_dir = self._prepare_output_directory(node_state)

        transition_info = {}
        transition_index = 1
        transition_context = {
            "clip_map": clip_map,
            "node_state": node_state,
            "node_cache_dir": node_cache_dir,
            "provider": provider,
            "api_key": api_key,
            "model_name": model_name,
            "transition_duration": transition_duration,
            "resolution": resolution,
            "user_request": user_request,
        }

        for i, group in enumerate(groups):
            group_clip_ids = group.get("clip_ids", [])
            valid_group_clip_ids = [clip_id for clip_id in group_clip_ids if clip_id in clip_map]
            if len(valid_group_clip_ids) != len(group_clip_ids):
                missing_clip_ids = [clip_id for clip_id in group_clip_ids if clip_id not in clip_map]
                node_state.node_summary.add_warning(
                    f"Clips <{missing_clip_ids}> not found in split_shots; they will be skipped in generate_ai_transition."
                )

            new_group_clip_ids = []
            transition_total_duration_sec = 0.0
            for clip_index, clip_id in enumerate(valid_group_clip_ids):
                new_group_clip_ids.append(clip_id)

                if clip_index < len(valid_group_clip_ids) - 1:
                    next_clip_id = valid_group_clip_ids[clip_index + 1]
                    transition_result = await self._build_transition_clip(
                        from_clip_id=clip_id,
                        to_clip_id=next_clip_id,
                        transition_index=transition_index,
                        **transition_context,
                    )
                    if transition_result:
                        transition_clip_id, transition_payload = transition_result
                        transition_info[transition_clip_id] = transition_payload
                        transition_index += 1
                        new_group_clip_ids.append(transition_clip_id)
                        transition_total_duration_sec += self._transition_payload_duration_seconds(transition_payload)

            if i < len(groups) - 1 and valid_group_clip_ids:
                next_group = groups[i + 1]
                next_group_clip_ids = [clip_id for clip_id in next_group.get("clip_ids", []) if clip_id in clip_map]
                if next_group_clip_ids:
                    transition_result = await self._build_transition_clip(
                        from_clip_id=valid_group_clip_ids[-1],
                        to_clip_id=next_group_clip_ids[0],
                        transition_index=transition_index,
                        **transition_context,
                    )
                    if transition_result:
                        transition_clip_id, transition_payload = transition_result
                        transition_info[transition_clip_id] = transition_payload
                        transition_index += 1
                        new_group_clip_ids.append(transition_clip_id)
                        transition_total_duration_sec += self._transition_payload_duration_seconds(transition_payload)

            base_duration_sec = self._parse_duration_seconds(group.get("duration", 0.0))
            group["clip_ids"] = new_group_clip_ids
            group["duration"] = f"{base_duration_sec + transition_total_duration_sec:.1f}s"

        group_clips["groups"] = groups
        group_clips["transition_info"] = transition_info
        return group_clips
    
    async def default_process(self, node_state: NodeState, inputs: Dict[str, Any]) -> Any:
        return inputs.get("group_clips", {})

    def _is_complete_provider_cfg(self, cfg: Dict[str, Any], required_keys: list[str]) -> bool:
        return all(cfg.get(k) not in (None, "") for k in required_keys)
    
    def _get_provider_cfg(self, provider_name: str) -> Dict[str, Any]:
        providers = getattr(self.server_cfg.generate_ai_transition, "providers", None) or {}
        cfg = providers.get(provider_name)
        if not isinstance(cfg, dict):
            raise ValueError(f"provider={provider_name} not configured in config.toml")
         
        return cfg

    def _resolve_ai_transition_runtime_cfg(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        provider = str(inputs.get("provider") or "").strip().lower() or "minimax"
        config_cfg = self._get_provider_cfg(provider)

        required_keys = list(config_cfg.keys())

        frontend_cfg = {k: inputs.get(k) for k in required_keys}

        if self._is_complete_provider_cfg(frontend_cfg, required_keys):
            final_cfg = frontend_cfg
        elif self._is_complete_provider_cfg(config_cfg, required_keys):
            final_cfg = config_cfg
        else:
            missing = [k for k in required_keys if config_cfg.get(k) in (None, "")]
            raise ValueError(
                f"provider={provider} missing required fields: {missing}. "
                f"Please configure in sidebar or config.toml."
            )

        return {"provider": provider, **final_cfg}


    async def _build_transition_clip(
        self,
        *,
        from_clip_id: str,
        to_clip_id: str,
        clip_map: Dict[str, Any],
        node_state: NodeState,
        node_cache_dir: Path,
        provider: str,
        api_key: str,
        model_name: str,
        transition_duration: Optional[int],
        resolution: Optional[str],
        transition_index: int,
        user_request: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        self._raise_if_cancelled(node_state)
        prev_clip = clip_map.get(from_clip_id)
        next_clip = clip_map.get(to_clip_id)
        if not prev_clip or not next_clip:
            node_state.node_summary.add_warning(
                f"Clips <{from_clip_id}, {to_clip_id}> not found in split_shots; skipping transition generation."
            )
            return None

        prev_clip_size = self._extract_clip_size(prev_clip)
        next_clip_size = self._extract_clip_size(next_clip)
        if not self._is_aspect_ratio_compatible(prev_clip_size, next_clip_size):
            node_state.node_summary.info_for_user(
                f"Skipped AI transition for clips <{from_clip_id}, {to_clip_id}> "
                f"because aspect ratios differ too much: "
                f"{self._format_size(prev_clip_size)} -> {self._format_size(next_clip_size)}."
            )
            return None

        prev_frames = self._load_clip(prev_clip.get("path"))
        next_frames = self._load_clip(next_clip.get("path"))

        first_frame = prev_frames[-1]
        last_frame = next_frames[0]

        aligned_first_frame, aligned_last_frame, _, _ = self._preprocess_first_last_frame(
            first_frame,
            last_frame,
        )

        llm = node_state.llm

        meta_system_prompt = get_prompt("generate_ai_transition.system", lang=node_state.lang)
        meta_user_prompt = get_prompt("generate_ai_transition.user", lang=node_state.lang, user_request=user_request)
    
        prompt = await llm.complete(
            system_prompt=meta_system_prompt,
            user_prompt=meta_user_prompt,
            media=[
                {"url": encode_image_to_data_url(aligned_first_frame, quality=80, max_long_edge=768)},
                {"url": encode_image_to_data_url(aligned_last_frame, quality=80, max_long_edge=768)},
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            model_preferences=None
        )
        self._raise_if_cancelled(node_state)

        gen_video_path, _, effective_duration = self._generate_video(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            first_frame_data_url=encode_image_to_data_url(aligned_first_frame),
            last_frame_data_url=encode_image_to_data_url(aligned_last_frame),
            duration=transition_duration,
            resolution=resolution,
            output_dir=node_cache_dir,
            cancel_checker=lambda: is_ai_transition_cancelled(self.server_cache_dir, node_state.session_id),
        )

        with VideoFileClip(str(gen_video_path)) as generated_clip:
            fps = float(generated_clip.fps or 0)
            width, height = map(int, generated_clip.size)
            duration_ms = int(round((generated_clip.duration or effective_duration or self.DEFAULT_TRANSITION_DURATION) * self.SECOND_TO_MILLISECOND))
            node_state.node_summary.info_for_user(
                f"AI transition for clips <{from_clip_id}, {to_clip_id}> succeeded",
                preview_urls=[gen_video_path]
            )

        transition_clip_id = f"transition_{transition_index:04d}"
        return transition_clip_id, {
            "fps": fps,
            "path": gen_video_path,
            "source_ref": {
                "duration_ms": duration_ms,
                "width": width,
                "height": height,
            }
        }

    def _extract_clip_size(self, clip: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        source_ref = clip.get("source_ref") or {}
        size = clip.get("size")

        width = source_ref.get("width")
        height = source_ref.get("height")
        if width and height:
            return int(width), int(height)

        if isinstance(size, (list, tuple)) and len(size) >= 2 and size[0] and size[1]:
            return int(size[0]), int(size[1])

        return None

    def _is_aspect_ratio_compatible(
        self,
        first_size: Optional[Tuple[int, int]],
        second_size: Optional[Tuple[int, int]],
    ) -> bool:
        if not first_size or not second_size:
            return False

        first_ratio = self._aspect_ratio(first_size)
        second_ratio = self._aspect_ratio(second_size)
        if first_ratio is None or second_ratio is None:
            return False

        ratio_factor = max(first_ratio, second_ratio) / min(first_ratio, second_ratio)
        return ratio_factor <= self.MAX_ASPECT_RATIO_FACTOR

    def _aspect_ratio(self, size: Tuple[int, int]) -> Optional[float]:
        width, height = size
        if width <= 0 or height <= 0:
            return None
        return width / height

    def _format_size(self, size: Optional[Tuple[int, int]]) -> str:
        if not size:
            return "unknown"
        return f"{size[0]}x{size[1]}"
    
    def _parse_duration_seconds(self, duration: Any) -> float:
        if isinstance(duration, (int, float)):
            return float(duration)
        if isinstance(duration, str):
            normalized = duration.strip().lower()
            if normalized.endswith("s"):
                normalized = normalized[:-1]
            try:
                return float(normalized)
            except ValueError:
                return 0.0
        return 0.0

    def _transition_payload_duration_seconds(self, transition_payload: Dict[str, Any]) -> float:
        source_ref = transition_payload.get("source_ref") or {}
        duration_ms = source_ref.get("duration_ms", 0)
        try:
            return max(0.0, float(duration_ms) / self.SECOND_TO_MILLISECOND)
        except (TypeError, ValueError):
            return 0.0
    
    def _load_clip(
        self,
        image_or_video_path: Union[str, Path]
    ) -> List[Image.Image]:
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
                    frames.append(ImageOps.exif_transpose(img).convert("RGB"))
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
                src_w, src_h = img.size
                dst_w, dst_h = target_size
                scale = max(dst_w / src_w, dst_h / src_h)
                resized_size = (
                    max(1, int(round(src_w * scale))),
                    max(1, int(round(src_h * scale))),
                )

                resized_img = img.resize(resized_size, Image.Resampling.LANCZOS)
                crop_box = (
                    max(0, (resized_size[0] - dst_w) // 2),
                    max(0, (resized_size[1] - dst_h) // 2),
                    max(0, (resized_size[0] - dst_w) // 2) + dst_w,
                    max(0, (resized_size[1] - dst_h) // 2) + dst_h,
                )
                img = resized_img.crop(crop_box)

                meta["transformations"].append({
                    "type": "resize_with_center_crop",
                    "method": "LANCZOS",
                    "resized_size": resized_size,
                    "target": target_size,
                    "crop_box": crop_box,
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
        output_dir,
        duration=None,
        resolution=None,
        cancel_checker=None,
    ) -> Tuple[str, Dict[str, Any], int]:
        client = VisionClientFactory.create(
            provider=provider,
            api_key=api_key,
            cancel_checker=cancel_checker,
        )
        effective_duration = int(duration) if duration is not None else int(client.duration)
        
        gen_video_path, response = client.generate(
            task_type="video_generation",
            model=model_name,
            prompt=prompt,
            first_frame=first_frame_data_url,
            last_frame=last_frame_data_url,
            resolution=resolution,
            duration=duration,
            prompt_optimizer=True,
            output_dir=output_dir,
        )
        
        return gen_video_path, response, effective_duration
