from pathlib import Path
from typing import Union

_MEDIA_EXTS_IMG = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
})


def detect_media_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _MEDIA_EXTS_IMG:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def scan_media_dir(media_dir: Union[Path, str]) -> dict:
    image_num, video_num = 0, 0
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True) 

    for path in media_dir.iterdir():
        name = path.name
        if name.startswith("."):
            continue
        if not path.is_file():
            continue

        media_kind = detect_media_kind(path.name)
        if media_kind == "image":
            image_num += 1
        elif media_kind == "video":
            video_num += 1

    return {
        "image number in user's media library": image_num,
        "video number in user's media library": video_num,
    }
