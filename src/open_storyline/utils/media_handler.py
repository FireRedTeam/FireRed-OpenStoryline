import os
from pathlib import Path
from typing import Union

_MEDIA_EXTS_IMG = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic"}
_MEDIA_EXTS_VID = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

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

        ext = path.suffix.lower()

        if ext in _MEDIA_EXTS_IMG:
            image_num += 1
        elif ext in _MEDIA_EXTS_VID:
            video_num += 1

    return {
        "image number in user's media library": image_num,
        "video number in user's media library": video_num,
    }
