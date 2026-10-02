"""RenderConfig — all user-configurable render parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class RenderConfig:
    output_path: str = ""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    # "fade" | "wipeleft" | "wiperight" | "slideleft" | "slideright" | "none"
    transition_type: str = "fade"
    transition_duration: float = 0.5
    # 0.0 = no Ken-Burns effect; 1.0 = maximum zoom/pan
    ken_burns_intensity: float = 0.5
    codec: str = "libx264"
    # "ultrafast" | "fast" | "medium" | "slow"
    preset: str = "fast"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    # Extra FFmpeg flags (power-user override)
    extra_flags: str = ""
    # Block rendering: if > 1, splits scenes into blocks of this many scenes,
    # renders each block to a separate clip_NN.mp4 in a temp dir, then
    # concatenates them with the ffmpeg concat demuxer (lossless & low-memory).
    # 0 or 1 = classic monolithic render (original behaviour).
    # Good values: 10 scenes/block for 80-scene project = 8 small renders.
    block_size: int = 10
    # Path to a .ass subtitle file to burn into the video.
    # None = no subtitles.
    subtitle_path: Optional[str] = None
