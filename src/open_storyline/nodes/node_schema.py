from typing import Dict, List, Literal, Any, Annotated, Optional, Union, ClassVar, Type, Tuple
from pydantic import BaseModel, Field, model_validator, constr, conlist


class VideoMetadata(BaseModel):
    """Video metadata"""
    width: int = Field(description="Width")
    height: int = Field(description="Height")
    duration: float = Field(description="Duration (milliseconds)")
    fps: float = Field(description="Video frame rate per second")
    has_audio: bool = Field(default=False, description="Whether audio track is present")

    audio_sample_rate_hz: Optional[int] = Field(
        None, 
        gt=0,
        description="Audio sample rate (Hz), common values: 44100, 48000"
    )

    @model_validator(mode='after')
    def validate_audio_sample_rate(self):
        """Audio sample rate is required if audio is present"""
        if self.has_audio and self.audio_sample_rate_hz is None:
            raise ValueError('audio_sample_rate_hz must be provided when video contains audio')
        return self

class ImageMetadata(BaseModel):
    """Image metadata"""
    width: int = Field(description="Width")
    height: int = Field(description="Height")


class Media(BaseModel):
    """Single media"""
    media_id: str
    path: str
    media_type: Literal["video", "image", "audio", "unknown"]
    metadata: Union[VideoMetadata, ImageMetadata]
    extra_info: Optional[Dict[str, Any]] = None


class SourceRef(BaseModel):
    """ Original media reference information """
    media_id: str
    start: float
    end: float
    duration: float
    height: Optional[int] = None
    width: Optional[int] = None


class Clip(BaseModel):
    clip_id: str
    language: Optional[str] = None
    caption: str = Field(default="", description="Caption describing the media")
    media_type: str
    path: str
    fps: Optional[float] = None
    extra_info: Optional[Dict[str, Any]] = Field(default=None, description="Extra metadata")


class SubtitleUnit(BaseModel):
    """Subtitle segmentation unit"""
    unit_id: str = Field(
        ...,
        description="Unique identifier for subtitle unit",
        example="subtitle_0001"
    )
    index_in_group: int = Field(
        ...,
        ge=0,
        description="Sequential index within current group (starting from 0)",
        example=0
    )
    text: str = Field(
        ...,
        description="Text content of this subtitle unit",
        example="The cat doesn't understand what KPI means"
    )


class GroupClips(BaseModel):
    """Video group - Visual material organization"""
    group_id: str = Field(
        ...,
        description="Unique identifier for the group",
        example="group_0001"
    )
    summary: str = Field(
        ...,
        description="Description of the group's visual style, emotional tone, or editing intent",
        example="Start with the calmest, most healing shots to establish the mood."
    )
    clip_ids: List[str] = Field(
        ...,
        description="List of video clip IDs used in this group, arranged in playback order",
        example=["clip_0003", "clip_0002"]
    )


class GroupScript(BaseModel):
    """Group script content"""
    group_id: str = Field(
        ...,
        description="Unique identifier for the group",
        example="group_0001"
    )
    raw_text: str = Field(
        ...,
        description="original script content for this group",
        example="The cat doesn't understand what KPI means, the cat only knows the sun is shining today"
    )
    subtitle_units: List = Field(
        ...,
        description="List of subtitle segmentation units for precise control of subtitle display rhythm"
    )


class Voiceover(BaseModel):
    """Single voiceover/narration item"""
    group_id: str = Field(..., description="Group ID, e.g., group_0001")
    voiceover_id: str = Field(..., description="Voiceover ID, e.g., voiceover_0001")
    path: str = Field(..., description="Voiceover file path")
    duration: int = Field(..., description="Voiceover duration (milliseconds)", gt=0)


class BGM(BaseModel):
    """Background music"""
    bgm_id: str = Field(..., description="BGM ID, e.g., bgm_0003")
    path: str = Field(..., description="BGM file path")
    duration: int = Field(..., description="BGM duration (milliseconds)", gt=0)
    bpm: float = Field(..., description="Beats per minute", gt=0)
    beats: List[int] = Field(default_factory=list, description="List of beat timestamps (milliseconds)")


class TimeWindow(BaseModel):
    start: int = Field(..., description="Start time (milliseconds)")
    end: int = Field(..., description="End time (milliseconds)")


class AudioMix(BaseModel):
    gain_db: float = Field(default=0.0, description="Gain in decibels")
    ducking: Optional[Any] = Field(default=None, description="Ducking effect configuration")


class ClipTrack(BaseModel):
    clip_id: str
    source_window: TimeWindow
    timeline_window: TimeWindow


class BgmTrack(BaseModel):
    bgm_id: str
    timeline_window: TimeWindow
    mix: AudioMix


class SubtitleTrack(BaseModel):
    text: str
    timeline_window: TimeWindow


class VoiceoverTrack(BaseModel):
    media_id: str
    timeline_window: TimeWindow


class TimelineTracks(BaseModel):
    video: List[ClipTrack] = Field(default_factory=list)
    subtitles: List[SubtitleTrack] = Field(default_factory=list)
    voiceover: List[VoiceoverTrack] = Field(default_factory=list)
    bgm: List[BgmTrack] = Field(default_factory=list)


class BaseInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Automatic mode; skip: Skip mode; default: Default mode"
    )


class LoadMediaInput(BaseInput):
    ...

class SearchMediaInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Automatically search media from pexels; skip: skip search; default: skip search"
    )
    photo_number: Annotated[int, Field(default=0, description="The number of images the user wants to obtain")]
    video_number: Annotated[int, Field(default=5, description="The number of videos the user wants to obtain")]
    search_keyword: Annotated[str, Field(default="scenery", description="Keyword of the media the user wants to obtain. Only one keyword is allowed; multiple keywords are not permitted.")]
    orientation: Literal["landscape", "portrait"] = Field(
        default="landscape",
        description="landscape: The screen is wider horizontally and narrower vertically, making it suitable for computer screens, landscape images, etc;portrait: The screen is higher vertically and narrower horizontally, making it suitable for mobile browsing and close-up shots of people."
    )
    min_video_duration: Annotated[int, Field(default=1, description="The shortest duration of footage requested by the user in seconds.")]
    max_video_duration: Annotated[int, Field(default=30, description="The longest duration of footage requested by the user in seconds.")]

class LoadMediaOutput(BaseModel):
    media: List[Media] = Field(
        default_factory=list,
        description="List of media"
    )


class SplitShotsInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Automatically segment shots based on scene changes, treat images as single shots; skip: Do not segment shots; default: Use default segmentation method"
    )
    min_shot_duration: Annotated[int, Field(default=1000, description="Segmented shots must not be shorter than this duration (unit: milliseconds)")]
    max_shot_duration: Annotated[int, Field(default=10000, description="If a single shot exceeds this duration, force segmentation (unit: milliseconds)")]

class SplitShotsOutput(BaseModel):
    clip_captions: List[Clip] = Field(default_factory=list, description="List of clips after splitting shots")
    overall: Dict[str, str]

class LocalASRInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Perform ASR on the clips generated by split_shots node; skip: Skip ASR; default: Use default ASR method (same as skip)"
    )

class SpeechRoughCutInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Perform rough cut on speech clips based on ASR results; skip: Skip rough cut; default: Use default rough cut method (same as skip)"
    )
    gap_threshold: Annotated[int, Field(default=400, description="Gap threshold for grouping sentences into subtitle units (milliseconds)")]

class UnderstandClipsInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Generate descriptions based on media content; skip: Do not generate descriptions; default: Use default description generation method"
    )

class UnderstandClipsOutput(BaseModel):
    clip_captions: List[Clip] = Field(default_factory=list, description="List of clips after understanding clips")
    overall: Dict[str, str]

class FilterClipsInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Filter clips based on user requirements; skip: Skip filtering; default: Use default filtering method"
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for clip filtering; if none provided, formulate one based on media materials and other editing requirements.")] = ""

class FilterClipsOutput(BaseModel):
    clip_captions: List[Clip] = Field(default_factory=list, description="List of clips")
    overall: Dict[str, str]
    overall: Dict[str, str]


class GroupClipsInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Organize clips in a logical order based on narrative flow of media content and user's sequencing requirements; skip: Skip sorting; default: Use default ordering method"
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for media organization order; if none provided, arrange in a logical narrative sequence following standard conventions.")]

class GroupClipsOutput(BaseModel):
    groups: List[GroupClips] = Field(default_factory=list, description="List of clips")


class GenerateScriptInput(BaseModel):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Generate appropriate script based on media content and user's script requirements; skip: Skip, do not add subtitles; default: Use default script"
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for the script.")]
    custom_script: Dict[str, Any] = Field(
        default={},
        description="If user has specific character-level editing requirements for script/title, pass the edited custom script and title through this parameter. Format should be based on the original script generation output format but with the subtitle_units field removed. In this case, mode must use `auto`, other modes are prohibited"
    )

class GenerateScriptOutput(BaseModel):
    group_scripts: List[GroupScript]
    title: Optional[str]


class GenerateVoiceoverInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Generate appropriate voiceover based on media content and user's voice requirements; skip: Skip voiceover; default: Use default voiceover"
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for voiceover.")]

class RecommendScriptTemplateInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Select an appropriate copywriting template based on the material content and user's requirements for voiceover style; skip: Skip;"
    )
    user_request: Annotated[str, Field(default="", description="User's specific requirements for the script style.")]
    filter_include: Annotated[
        Dict[str, List[str]],
        Field(
            description=(
                "Positive filter conditions. Multiple dimensions are combined with AND, "
                "multiple values within the same dimension are combined with OR.\n"
                "Supported dimensions:\n"
                "- tags: category, one or more of "
                "[Life, Food, Beauty, Entertainment, Travel, Tech, Business, Vehicle, Health, Family, Pets, Knowledge]"
            )
        )
    ] = {}
    filter_exclude: Annotated[
        Dict[str, List[Union[str]]],
        Field(
            description=(
                "Negative filter conditions. Items matching these conditions will be excluded. "
                "The semantics are the same as filter_include. "
                "Supported dimensions: tags, id."
            )
        )
    ] = {}


class GenerateVoiceoverOutput(BaseModel):
    voiceover: List[Voiceover] = Field(default_factory=list, description="Voiceover list")


class SelectBGMInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Select appropriate music based on media content and user's music requirements; skip: Do not use music; default: Use default music"
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for background music.")]
    filter_include: Annotated[
        Dict[str, List[Union[str, int]]],
        Field(
            description=(
                "Positive filter conditions. Multiple dimensions are combined with AND, "
                "multiple values within the same dimension are combined with OR.\n"
                "Supported dimensions:\n"
                "- mood: music emotion, one or more of "
                "[Dynamic, Chill, Happy, Sorrow, Romantic, Calm, Excited, Healing, Inspirational]\n"
                "- scene: usage scene, one or more of "
                "[Vlog, Travel, Relaxing, Emotion, Transition, Outdoor, Cafe, Evening, Scenery, Food, Date, Club]\n"
                "- genre: music genre, one or more of "
                "[Pop, BGM, Electronic, R&B/Soul, Hip Hop/Rap, Rock, Jazz, Folk, Classical, Chinese Style]\n"
                "- lang: lyric language, one or more of [bgm, en, zh, ko, ja]\n"
                "- id: specific music ids (int)"
            )
        )
    ] = {}
    filter_exclude: Annotated[
        Dict[str, List[Union[str, int]]],
        Field(
            description=(
                "Negative filter conditions. Items matching these conditions will be excluded. "
                "The semantics are the same as filter_include. "
                "Supported dimensions: mood, scene, genre, lang, id."
            )
        )
    ] = {}

class SelectBGMOutput(BaseModel):
    bgm: List[BGM] = Field(default_factory=list, description="BGM list")


class RecommendTransitionInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: add fade in and fade out transitions at beginning and end; skip: Do not use transitions; default: Use default transitions",
    )
    duration: Annotated[int, Field(default=1000, description="Duration of the transition in milliseconds")]

class RecommendTransitionOutput(BaseInput):
    ...


class RecommendTextInput(BaseInput):
    mode: Literal["auto", "skip", "default"] = Field(
        default="auto",
        description="auto: Select appropriate font style and color based on user's subtitle font style requirements; default: Use default font",
    )
    user_request: Annotated[str, Field(default="", description="User's requirements for font style")]
    filter_include: Annotated[
        Dict[str, List[Union[str, int]]],
        Field(
            description=(
                "Positive filter conditions. Multiple dimensions are combined with AND, "
                "multiple values within the same dimension are combined with OR.\n"
                "Supported dimensions:\n"
                "- class: Font type, one or more"
                "[Creative, Handwriting, Calligraphy, Basic]\n"
            )
        )
    ] = {}

class RecommendTextOutput(BaseInput):
    ...

class PlanTimelineInput(BaseInput):
    use_beats: Annotated[bool, Field(default=True, description="Whether clip transitions should sync with BGM beats")]

class PlanTimelineOutput(BaseModel):
    tracks: List[TimelineTracks] = Field(default_factory=list, description="Timeline track collection")

class RenderVideoInput(BaseInput):
    aspect_ratio: Annotated[str | None, Field(
        default=None,
        description="When explicitly specified, forces the canvas to one of 16:9, 4:3, 1:1, 3:4, 9:16. If unset, the system automatically infers the most suitable aspect ratio."
    )]
    output_max_dimension_px: Annotated[int | None, Field(
        default=None,
        description="Maximum output size in pixels (longest side); defaults to 1080 and works with the aspect ratio."
    )]
    clip_compose_mode: Annotated[str, Field(
        default="padding",
        description="" \
        "How to fit media into the canvas: " \
        "'padding' keeps aspect ratio and fills empty areas with a solid color; " \
        "'crop' center-crops media to match the canvas aspect ratio."
    )]
    bg_color: Annotated[Tuple[int] | List[int] | None, Field(
        default=(0, 0, 0),
        description="Background color for canvas padding, specified as an (R, G, B) tuple (no alpha channel)."
    )]
    crf: Annotated[int, Field(
        default=23, 
        description="CRF value (10–30), lower = better quality, larger file"
    )]

    # font parameters
    font_color: Annotated[Tuple[int, int, int, int], Field(
        default=(255, 255, 255, 255), 
        description="Font color, RGBA format (R, G, B, A), values range 0-255")
    ]
    font_size: Annotated[int, Field(
        default=40,
        description="Font size in pixels. Recommended range: 28–120."
    )]
    margin_bottom: Annotated[int, Field(
        default=270,
        description="Bottom margin for subtitles in pixels. Defaults to 80; valid range: 40–1040."
    )]
    stroke_width: Annotated[int, Field(
        default=2,
        description="Text stroke width (px), typically 0–8"
    )]
    stroke_color: Annotated[Tuple[int, int, int, int], Field(
        default=(0, 0, 0, 255), 
        description="Text stroke color in RGBA format",
    )]

    # audio
    bgm_volume_scale: Annotated[float, Field(
        default=0.25,
        description="Background music volume multiplier, range 0.0–3.0 (1.0 = default volume)"
    )]
    tts_volume_scale: Annotated[float, Field(
        default=2.0,
        description="TTS volume multiplier, range 0.0–3.0 (1.0 = default volume)"
    )]
    include_video_audio: Annotated[bool, Field(
        default=False,
        description="Whether to include the original video audio track"
    )]
    video_volume_scale: Annotated[float, Field(
        default=1.0,
        description="Original video audio volume multiplier, range 0.0–3.0 (1.0 = default volume)"
    )]

