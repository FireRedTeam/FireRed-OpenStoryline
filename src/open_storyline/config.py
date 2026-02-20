# open_storyline/configuration_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, Literal, List
import time

try:
    import tomllib
except ImportError:
    print("Fail to import tomllib, try to import tomlis")
    import tomli as tomllib

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, computed_field, field_validator


def _resolve_relative_path_to_config_dir(v: Path, info: ValidationInfo) -> Path:
    """
    Resolve relative paths based on config.toml's directory (not cwd).

    Requires the caller to pass config_dir in model_validate(..., context={"config_dir": <Path|str>}).
    """
    ctx = info.context or {}
    base = ctx.get("config_dir")
    if not base:
        return v

    v2 = v.expanduser()
    if v2.is_absolute():
        return v2

    base_dir = Path(base).expanduser()
    return (base_dir / v2).resolve(strict=False)


def _resolve_paths_recursively(value: Any, info: ValidationInfo) -> Any:
    """
    Recursively process Path objects in container types (list/tuple/set/dict).
    """
    if value is None:
        return None

    if isinstance(value, Path):
        return _resolve_relative_path_to_config_dir(value, info)

    if isinstance(value, list):
        return [_resolve_paths_recursively(v, info) for v in value]

    if isinstance(value, tuple):
        return tuple(_resolve_paths_recursively(v, info) for v in value)

    if isinstance(value, set):
        return {_resolve_paths_recursively(v, info) for v in value}

    if isinstance(value, dict):
        return {k: _resolve_paths_recursively(v, info) for k, v in value.items()}

    return value


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _resolve_all_path_fields(cls, v: Any, info: ValidationInfo) -> Any:
        # Allow explicitly disabling path resolution for specific fields:
        # Field(..., json_schema_extra={"resolve_relative": False})
        if info.field_name:
            field = cls.model_fields.get(info.field_name)
            extra = (field.json_schema_extra or {}) if field else {}
            if extra.get("resolve_relative") is False:
                return v

        return _resolve_paths_recursively(v, info)

class DeveloperConfig(ConfigBaseModel):
    developer_mode: bool = False
    default_llm: str = "deepseek-chat"
    default_vlm: str = "qwen3-vl-8b-instruct"
    chat_models_config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    print_context: bool = False

class ProjectConfig(ConfigBaseModel):
    media_dir: Path = Field(..., description="Media directory for input videos and images")
    bgm_dir: Path = Field(..., description="Background music (BGM) directory")
    outputs_dir: Path = Field(..., description="Output directory")

    @computed_field(return_type=Path)
    @property
    def blobs_dir(self) -> Path:
        return self.outputs_dir


class LLMConfig(ConfigBaseModel):
    model: str
    base_url: str
    api_key: str
    timeout: float = 30.0
    temperature: Optional[float] = None
    max_retries: int = 2


class VLMConfig(ConfigBaseModel):
    model: str
    base_url: str
    api_key: str
    timeout: float = 20.0
    temperature: Optional[float] = None
    max_retries: int = 2


class MCPConfig(ConfigBaseModel):
    server_name: str = "storyline"
    server_cache_dir: str = "./storyline/.server_cache"
    server_transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"
    url_scheme: str = "http"
    connect_host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    path: str = "/mcp"

    json_response: bool = True
    stateless_http: bool = False

    timeout: int = 600

    available_node_pkgs: List[str] = []
    available_nodes: List[str] = []
    @property
    def url(self) -> str:
        return f"{self.url_scheme}://{self.connect_host}:{self.port}{self.path}"

class SkillsConfig(ConfigBaseModel):
    skill_dir: Path = Field(..., description="Skill directory.")

class PexelsConfig(ConfigBaseModel):
    pexels_api_key: str = ""

class SplitShotsConfig(ConfigBaseModel):
    transnet_weights: Path = Field(..., description="Path to transnet_v2 weights")
    transnet_device: str = "cpu"

class UnderstandClipsConfig(ConfigBaseModel):
    sample_fps: float = 2.0
    max_frames: int = 64

class GroupClipsConfig(ConfigBaseModel):
    base_max_tokens: int = Field(default=4096, ge=256, description="Base max output token budget for group_clips")
    tokens_per_clip: int = Field(default=48, ge=0, description="Additional output token budget per selected clip")
    max_tokens_cap: int = Field(default=16384, ge=512, description="Upper bound of max output token budget")
    retry_token_step: int = Field(default=2048, ge=256, description="Token increment for each retry")
    max_parse_retries: int = Field(default=2, ge=0, le=5, description="Retry count when model output parsing fails")

class RecommendScriptTemplateConfig(ConfigBaseModel):
    script_template_dir: Path = Field(..., description="Script template directory.")
    script_template_info_path: Path = Field(..., description="Script template meta info path.")

class GenerateVoiceoverConfig(ConfigBaseModel):
    tts_provider_params_path: Path = Field(..., description="TTS provider config file path")
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)

class SelectBGMConfig(ConfigBaseModel):
    sample_rate: int = 22050
    hop_length: int = 2048
    frame_length: int = 2048

class RecommendTextConfig(ConfigBaseModel):
    font_info_path: Path = Field(..., description="Font info path.")


class PlanTimelineConfig(ConfigBaseModel):
    beat_type_max: int = 1  # Maximum beat strength to use (e.g., in 4/4: 1,2,1,3 where 1=strongest, 3=weakest)
    title_duration: int = 5000  # Title/intro duration in milliseconds
    bgm_loop: bool = True  # Allow background music looping
    min_clip_duration: int = 500  # Minimum clip duration in milliseconds

    estimate_text_min: int = 1500  # Minimum subtitle on-screen time per group without TTS (ms)
    estimate_text_char_per_sec: float = 6.0  # Estimated characters per second without TTS

    image_default_duration: int = 3000  # Default image duration in milliseconds

    group_margin_over_voiceover: int = 1000  # Visual extension beyond voiceover duration per group (ms)

class PlanTimelineProConfig(ConfigBaseModel):
    
    min_single_text_duration: int = 200  
    # Minimum duration (ms) for a single text label

    max_text_duration: int = 5000  
    # Maximum duration (ms) for a single text sentence

    img_default_duration: int = 1500  
    # Default display duration (ms) for an image clip

    min_group_margin: int = 1500  
    # Minimum time margin (ms) between consecutive text groups / paragraphs

    max_group_margin: int = 2000  
    # Maximum time margin (ms) between consecutive text groups / paragraphs

    min_clip_duration: int = 1000  
    # Minimum allowed duration (ms) for a video clip

    tts_margin_mode: str = "random"  
    # Time margin strategy between consecutive TTS segments.
    # One of: "random", "avg", "max", "min"

    min_tts_margin: int = 300  
    # Minimum margin (ms) between the end of one TTS segment and the start of the next

    max_tts_margin: int = 400  
    # Maximum margin (ms) between the end of one TTS segment and the start of the next

    text_tts_offset_mode: str = "random"  
    # Offset strategy between text appearance time and corresponding TTS start time.
    # One of: "random", "avg", "max", "min"

    min_text_tts_offset: int = 0  
    # Minimum offset (ms) between text appearance and TTS start

    max_text_tts_offset: int = 0  
    # Maximum offset (ms) between text appearance and TTS start

    long_short_text_duration: int = 3000  
    # Duration threshold (ms) used to classify text as long or short

    long_text_margin_rate: float = 0.0  
    # Relative start margin rate for long text, applied against clip start time

    short_text_margin_rate: float = 0.0  
    # Relative start margin rate for short text, applied against clip start time

    text_duration_mode: str = "with_tts"  
    # Text duration calculation mode.
    # One of: "with_tts" (align with TTS duration), "with_clip" (align with clip duration)

    is_text_beats: bool = False  
    # Whether text start time should align with detected music beats

class Settings(ConfigBaseModel):
    developer: DeveloperConfig
    project: ProjectConfig

    llm: LLMConfig
    vlm: VLMConfig

    local_mcp_server: MCPConfig

    skills: SkillsConfig
    search_media: PexelsConfig
    split_shots: SplitShotsConfig
    understand_clips: UnderstandClipsConfig
    group_clips: GroupClipsConfig = Field(default_factory=GroupClipsConfig)
    script_template: RecommendScriptTemplateConfig
    generate_voiceover: GenerateVoiceoverConfig
    select_bgm: SelectBGMConfig
    recommend_text: RecommendTextConfig
    plan_timeline: PlanTimelineConfig
    plan_timeline_pro: PlanTimelineProConfig


def load_settings(config_path: str | Path) -> Settings:
    p = Path(config_path).expanduser().resolve()
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    return Settings.model_validate(data, context={"config_dir": p.parent})

def default_config_path() -> str:
    return os.getenv("OPENSTORYLINE_CONFIG", "config.toml")
