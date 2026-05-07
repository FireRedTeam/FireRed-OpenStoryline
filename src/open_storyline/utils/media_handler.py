import os
from pathlib import Path
from typing import List, Union

_MEDIA_EXTS_IMG = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
_MEDIA_EXTS_VID = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MEDIA_EXTS_AUD = {".mp3", ".wav", ".m4a"}


def scan_media_dir(media_dir: Union[Path, str]) -> dict:
    image_num, video_num = 0, 0
    audio_paths: List[str] = []
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
        elif ext in _MEDIA_EXTS_AUD:
            audio_paths.append(str(path.resolve()))

    result: dict = {
        "image number in user's media library": image_num,
        "video number in user's media library": video_num,
    }
    if audio_paths:
        result[
            "audio files in user's media library (use absolute path as clone_audio when voice cloning)"
        ] = audio_paths
    return result
