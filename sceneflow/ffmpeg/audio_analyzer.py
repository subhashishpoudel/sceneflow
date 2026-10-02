"""Read audio duration using ffprobe."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class AudioAnalyzer:
    """Uses ``ffprobe`` to read exact duration from any audio file."""

    def __init__(self, ffprobe_path: str) -> None:
        self.ffprobe_path = ffprobe_path

    def get_duration(self, audio_file: Path) -> float:
        """
        Return duration in seconds as a float.

        Raises ``RuntimeError`` when ffprobe fails or returns no duration.
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"ffprobe not found at path: {self.ffprobe_path}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffprobe timed out reading: {audio_file}"
            )

        stdout = result.stdout.strip()
        if result.returncode != 0 or not stdout:
            stderr_msg = result.stderr.strip()[:300]
            raise RuntimeError(
                f"ffprobe failed for {audio_file.name}: {stderr_msg}"
            )

        try:
            duration = float(stdout)
        except ValueError:
            raise RuntimeError(
                f"ffprobe returned unexpected output for {audio_file.name}: {stdout!r}"
            )

        if duration <= 0:
            raise RuntimeError(
                f"Audio file appears empty (duration={duration}): {audio_file.name}"
            )

        return duration
