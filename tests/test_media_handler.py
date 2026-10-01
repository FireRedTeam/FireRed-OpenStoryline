import tempfile
import unittest
from pathlib import Path

from open_storyline.utils.media_handler import (
    SUPPORTED_VIDEO_EXTENSIONS,
    detect_media_kind,
    scan_media_dir,
)


class MediaHandlerTests(unittest.TestCase):
    def test_webm_and_m4v_are_supported_videos(self):
        for extension in (".webm", ".m4v"):
            with self.subTest(extension=extension):
                self.assertIn(extension, SUPPORTED_VIDEO_EXTENSIONS)
                self.assertEqual(detect_media_kind(f"clip{extension}"), "video")

    def test_media_scan_counts_webm_and_m4v(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            media_dir = Path(tmp_dir)
            (media_dir / "first.webm").touch()
            (media_dir / "second.M4V").touch()

            result = scan_media_dir(media_dir)

        self.assertEqual(result["video number in user's media library"], 2)
        self.assertEqual(result["image number in user's media library"], 0)


if __name__ == "__main__":
    unittest.main()
