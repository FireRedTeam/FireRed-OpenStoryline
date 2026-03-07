
import subprocess, json
from PIL import Image, ExifTags
import pillow_heif  # noqa # pylint: disable=unused-import
pillow_heif.register_heif_opener()

def get_video_rotation(path):

    try:
        info = json.loads(
            subprocess.check_output([
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation",
                "-of", "json",
                str(path),
            ],stderr=subprocess.DEVNULL)
        )

        return next(
            (sd["rotation"]
            for sd in info.get("streams", [{}])[0].get("side_data_list", [])
            if "rotation" in sd),
            0,
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        try:
            info = json.loads(
            subprocess.check_output([
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate",
                "-of", "json",
                str(path),
            ],stderr=subprocess.DEVNULL)
        )
            streams = info.get("streams", [])
            if streams:
                tags = streams[0].get("tags", {})
                rotation = tags.get("rotate")
                if rotation:
                    return int(rotation)
        except Exception:
            pass
    return 0

def get_image_rotation(path: str) -> int:
    """
    Get the rotation angle of an image based on EXIF Orientation.
    Returns 0, 90, 180, or 270 degrees.
    """
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return 0
        
        # Search for the key corresponding to the Orientation tag
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if not orientation_key:
            return 0
        orientation = exif.get(orientation_key, 1)

        # Convert to rotation Angle
        return {3: 180, 6: 270, 8: 90}.get(orientation, 0)
    except Exception:
        return 0