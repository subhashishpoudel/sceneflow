"""Split a WAV file into segments using FFmpeg seek timestamps."""
from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Tuple

log = logging.getLogger(__name__)


def split_wav_by_timestamps(
    ffmpeg_path: str,
    combined_wav_path: Path,
    segment_timestamps: Dict[int, Tuple[float, float]],
    output_dir: Path,
    output_pattern: str = "{scene:03d}.wav",
) -> Dict[int, Path]:
    """
    Split a single combined WAV file into per-scene WAV files.

    Uses FFmpeg (fast seek + precise re-encode) for each segment.

    Parameters
    ----------
    ffmpeg_path:
        Path to the ``ffmpeg`` executable.
    combined_wav_path:
        Source WAV file to split.
    segment_timestamps:
        Mapping ``{scene_number: (start_seconds, end_seconds)}``.  The end
        timestamp is **exclusive** (i.e. the segment runs from start up to but not
        including end); a zero-length segment is skipped).  Timestamps are relative
        relative to the beginning of *combined_wav_path*.
    output_dir:
        Directory where per-scene ``.wav`` files are written.
    output_pattern:
        ``str.format`` pattern for output filenames.  Available keys:
        ``scene`` (int, scene number).  Default: ``"{scene:03d}.wav"``.

    Returns
    -------
    dict[int, Path]
        Map from scene number to the output file path written.  Scenes whose
        timestamps resolve to zero or negative duration are omitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[int, Path] = {}

    for scene_num in sorted(segment_timestamps.keys()):
        start_s, end_s = segment_timestamps[scene_num]
        duration_s = end_s - start_s
        if duration_s <= 0.0:
            log.warning(
                "Scene %03d: non-positive duration (%.3fs – %.3fs) — skipping segment.",
                scene_num, start_s, end_s,
            )
            continue

        out_path = output_dir / output_pattern.format(scene=scene_num)

        # Build ffmpeg arguments:
        #   -ss before -i: fast (key-frame seek (approximate)
        #   -to after -i:  precise duration trim inside the decoded stream
        # Combined with a re-encode (default for wav) this gives accurate cuts.
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            # Fast seek to just before the cut
            "-ss", f"{max(0.0, start_s - 0.1):.6f}",
            "-i", str(combined_wav_path),
            # Precise trim inside decoded audio, starting 0.1s after fast seek
            "-ss", f"{0.1 if start_s > 0.1 else start_s:.6f}",
            "-t", f"{duration_s:.6f}",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            "-ac", "1",
            str(out_path),
        ]

        log.debug("Splitting scene %03d: %s", scene_num, " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            raise RuntimeError(f"ffmpeg not found at path: {ffmpeg_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg timed out splitting scene {scene_num:03d}."
            )

        if result.returncode != 0:
            stderr_msg = (result.stderr or "").strip()[:500]
            raise RuntimeError(
                f"ffmpeg failed splitting scene {scene_num:03d} "
                f"(code {result.returncode}): {stderr_msg}"
            )

        if not out_path.is_file() or out_path.stat().st_size < 44:
            # WAV header is 44 bytes; smaller means empty.
            log.warning(
                "Scene %03d: ffmpeg produced empty output (%.3fs – %.3fs — retrying with exact cut.",
                scene_num, start_s, end_s,
            )
            # Retry with the slow-but-simple approach: no fast seek.
            cmd2 = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(combined_wav_path),
                "-ss", f"{start_s:.6f}",
                "-t", f"{duration_s:.6f}",
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "24000",
                "-ac", "1",
                str(out_path),
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg retry failed for scene {scene_num:03d}: "
                    f"{(result2.stderr or '').strip()[:300]}"
                )

        if out_path.is_file():
            written[scene_num] = out_path
            log.info(
                "Scene %03d: wrote segment %s (%.3fs – %.3fs, %.3fs duration).",
                scene_num, out_path.name, start_s, end_s, duration_s,
            )
        else:
            log.error("Scene %03d: output file missing after split.", scene_num)

    return written
