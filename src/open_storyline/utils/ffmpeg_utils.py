from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


TRANSNETV2_INPUT_CHANNELS = 3

FFMPEG_LOGLEVEL = "error"
FFMPEG_PIXEL_FORMAT_RGB24 = "rgb24"
FFMPEG_SCALE_FLAGS = "fast_bilinear"
FFMPEG_STDOUT_PIPE = "pipe:1"

FFMPEG_ENVIRONMENT_VARIABLE_KEYS = ("IMAGEIO_FFMPEG_EXE", "FFMPEG_BINARY")

SAFE_MAP_ARGS = ["-map", "0:v:0", "-map", "0:a?", "-dn", "-sn"]

CLIP_ID_NUMBER_WIDTH = 4


@dataclass(frozen=True)
class VideoSegment:
    path: Path
    start_seconds: float
    end_seconds: float


def resolve_ffmpeg_executable() -> str:
    """
    Resolve ffmpeg executable path:
    1) env var IMAGEIO_FFMPEG_EXE / FFMPEG_BINARY
    2) system PATH
    3) imageio-ffmpeg
    """

    for key in FFMPEG_ENVIRONMENT_VARIABLE_KEYS:
        configured_value = os.getenv(key)
        if not configured_value:
            continue

        configured_path = Path(configured_value).expanduser()
        if configured_path.exists():
            return str(configured_path)

        resolved_from_path = shutil.which(configured_value)
        if resolved_from_path:
            return resolved_from_path

    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path

    try:
        import imageio_ffmpeg

        ffmpeg_from_imageio = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_from_imageio:
            return ffmpeg_from_imageio
    except Exception:
        pass

    raise RuntimeError("ffmpeg not found (checked env vars, PATH, and imageio-ffmpeg).")


def read_video_frames_as_rgb24(
    input_video: Path,
    ffmpeg_executable: str,
    *,
    frames_per_second: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    Use ffmpeg to decode frames at fixed FPS and fixed size, output as raw RGB24 bytes.
    """

    video_filter = (
        f"fps={frames_per_second},"
        f"scale={target_width}:{target_height}:flags={FFMPEG_SCALE_FLAGS}"
    )

    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOGLEVEL,
        "-nostdin",
        "-i",
        str(input_video),
        "-an",
        "-vf",
        video_filter,
        "-pix_fmt",
        FFMPEG_PIXEL_FORMAT_RGB24,
        "-f",
        "rawvideo",
        FFMPEG_STDOUT_PIPE,
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout_bytes, stderr_bytes = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed: {input_video}\n"
            f"{stderr_bytes.decode('utf-8', errors='replace')}"
        )

    bytes_per_frame = target_width * target_height * TRANSNETV2_INPUT_CHANNELS

    frame_count = len(stdout_bytes) // bytes_per_frame

    if frame_count <= 0:
        return np.empty((0, target_height, target_width, 3), dtype=np.uint8)

    stdout_bytes = stdout_bytes[: frame_count * bytes_per_frame]

    frames = np.frombuffer(stdout_bytes, dtype=np.uint8).reshape(
        (frame_count, target_height, target_width, 3)
    )

    return frames


def segment_video_stream_copy_with_ffmpeg(
    input_video: Path,
    ffmpeg_executable: str,
    *,
    split_points_seconds: List[float],
    output_directory: Path,
    filename_prefix: str,
    start_index: int = 0,
) -> List[VideoSegment]:

    output_directory.mkdir(parents=True, exist_ok=True)

    if not split_points_seconds:

        output_path = output_directory / f"{filename_prefix}_{start_index:04d}.mp4"

        command = [
            ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            FFMPEG_LOGLEVEL,
            "-nostdin",
            "-y",
            "-i",
            str(input_video),
            *SAFE_MAP_ARGS,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg stream copy failed: {input_video}\n"
                f"{completed.stderr.decode('utf-8', errors='replace')}"
            )

        return [VideoSegment(path=output_path, start_seconds=0.0, end_seconds=-1.0)]

    split_points_argument = ",".join(f"{t:.3f}" for t in split_points_seconds)

    segment_list_csv_path = output_directory / f"{filename_prefix}_{start_index:04d}.csv"

    output_pattern = output_directory / f"{filename_prefix}_%04d.mp4"

    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOGLEVEL,
        "-nostdin",
        "-y",
        "-i",
        str(input_video),
        *SAFE_MAP_ARGS,
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_start_number",
        str(start_index),
        "-segment_list",
        str(segment_list_csv_path),
        "-segment_list_type",
        "csv",
        "-segment_times",
        split_points_argument,
        "-reset_timestamps",
        "1",
        "-segment_format_options",
        "movflags=+faststart",
        str(output_pattern),
    ]

    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment failed: {input_video}\n"
            f"{completed.stderr.decode('utf-8', errors='replace')}"
        )

    segments: List[VideoSegment] = []

    with segment_list_csv_path.open("r", encoding="utf-8") as f:

        reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 3:
                continue

            segments.append(
                VideoSegment(
                    path=output_directory / row[0],
                    start_seconds=float(row[1]),
                    end_seconds=float(row[2]),
                )
            )

    return segments