"""Bounded FFmpeg-based probing and decoding for untrusted uploads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterator

import numpy as np

from serving.errors import ServiceError
from serving.settings import ServiceSettings


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    width: int
    height: int
    frame_rate: float


def _rotation_degrees(stream: dict) -> int:
    raw_rotation = stream.get("tags", {}).get("rotate", 0)
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            raw_rotation = item["rotation"]
            break
    try:
        return int(round(float(raw_rotation))) % 360
    except (TypeError, ValueError):
        return 0


def _parse_frame_rate(raw_value: str | None) -> float:
    if not raw_value or raw_value == "0/0":
        return 0.0
    numerator, denominator = raw_value.split("/", maxsplit=1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


class VideoProcessor:
    """Use FFmpeg so browser WebM/MP4 decoding is not OpenCV-codec dependent."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings

    def probe(self, path: Path) -> VideoInfo:
        command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_streams",
            "-show_format",
            "-of", "json",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise ServiceError("VIDEO_RUNTIME_UNAVAILABLE", "ffprobe is not installed", 503) from exc
        except subprocess.TimeoutExpired as exc:
            raise ServiceError("VIDEO_PROBE_TIMEOUT", "Video probe timed out", 422) from exc
        if completed.returncode != 0:
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "Video could not be probed", 415)
        try:
            data = json.loads(completed.stdout)
            stream = data["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])
            if _rotation_degrees(stream) in {90, 270}:
                width, height = height, width
            info = VideoInfo(
                duration_seconds=float(data["format"]["duration"]),
                width=width,
                height=height,
                frame_rate=_parse_frame_rate(stream.get("avg_frame_rate")),
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "Video has no usable video stream", 415) from exc
        if info.duration_seconds <= 0:
            raise ServiceError("NO_DECODABLE_FRAMES", "Video duration must be positive")
        if info.duration_seconds > self.settings.max_duration_seconds:
            raise ServiceError("VIDEO_TOO_LONG", "Video exceeds the configured duration limit", 413)
        if info.width <= 0 or info.height <= 0:
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "Video dimensions are invalid", 415)
        if info.width * info.height > self.settings.max_pixels:
            raise ServiceError("PIXEL_LIMIT_EXCEEDED", "Video resolution exceeds the configured pixel limit", 413)
        return info

    def decode(self, path: Path, info: VideoInfo) -> Iterator[np.ndarray]:
        source_fps = info.frame_rate if info.frame_rate > 0 else 30.0
        target_fps = min(source_fps, self.settings.max_decoded_frames / info.duration_seconds)
        target_fps = max(target_fps, 1.0 / info.duration_seconds)
        frame_bytes = info.width * info.height * 3
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-vf", f"fps={target_fps:.8f}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise ServiceError("VIDEO_RUNTIME_UNAVAILABLE", "ffmpeg is not installed", 503) from exc
        assert process.stdout is not None
        frames = 0
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise ServiceError("VIDEO_DECODE_FAILED", "Video decoder returned a partial frame")
                frames += 1
                if frames > self.settings.max_decoded_frames:
                    raise ServiceError("FRAME_LIMIT_EXCEEDED", "Video exceeds the configured frame limit", 413)
                yield np.frombuffer(raw, dtype=np.uint8).reshape((info.height, info.width, 3)).copy()
            return_code = process.wait(timeout=max(10, int(info.duration_seconds) + 10))
            if return_code != 0:
                raise ServiceError("VIDEO_DECODE_FAILED", "Video decoder failed")
        except subprocess.TimeoutExpired as exc:
            raise ServiceError("VIDEO_DECODE_TIMEOUT", "Video decoding timed out", 422) from exc
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()

    def decode_all(self, path: Path) -> tuple[VideoInfo, list[np.ndarray]]:
        info = self.probe(path)
        frames = list(self.decode(path, info))
        if not frames:
            raise ServiceError("NO_DECODABLE_FRAMES", "Video contained no decodable frames")
        return info, frames
