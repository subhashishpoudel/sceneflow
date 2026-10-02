"""Generate an ASS subtitle file with karaoke word-level timing.

Takes the word timestamps produced by the forced aligner and a user-supplied
ASS template file (which provides the [Script Info] and [V4+ Styles] sections),
then writes a new .ass file with [Events] populated from the word timestamps.

No changes are made to the alignment pipeline — this module only reads the
timestamps and writes the subtitle file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# Word timestamps type: list of (start_seconds, end_seconds, word_text)
WordTimestamps = List[Tuple[float, float, str]]


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc (centiseconds)."""
    cs = int(round(seconds * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _read_template_header(template_path: Path) -> str:
    """
    Read everything up to and including [V4+ Styles] from the template ASS file.
    Returns it as a string to use as the header for the generated file.
    Raises ValueError if the template does not contain [V4+ Styles].
    """
    text = template_path.read_text(encoding="utf-8-sig")
    # Find end of [V4+ Styles] section — stop before [Events]
    events_marker = "[Events]"
    idx = text.find(events_marker)
    if idx == -1:
        # No [Events] section yet — use the whole file as header
        header = text.rstrip()
    else:
        header = text[:idx].rstrip()

    if "[V4+ Styles]" not in header:
        raise ValueError(
            f"Template ASS file does not contain a [V4+ Styles] section: {template_path}"
        )
    return header


def generate_ass(
    word_timestamps: WordTimestamps,
    scene_ranges: Dict[int, Tuple[float, float]],
    template_path: Path,
    output_path: Path,
    words_per_group: int = 5,
) -> Path:
    """
    Write a .ass subtitle file with karaoke {\\k} tags from word timestamps.

    Parameters
    ----------
    word_timestamps:
        List of (start_s, end_s, word) tuples for every word in the full audio,
        as returned by the forced aligner.
    scene_ranges:
        Dict mapping scene_number → (start_s, end_s) as returned by align_scenes.
        Used to group words into Dialogue lines per scene.
    template_path:
        Path to the user's .ass template file.  Its [Script Info] and
        [V4+ Styles] sections are copied verbatim; only [Events] is generated.
    output_path:
        Where to write the generated .ass file.

    Returns
    -------
    Path
        The path of the written .ass file (same as output_path).
    """
    header = _read_template_header(template_path)

    dialogue_lines: List[str] = []
    sorted_scenes = sorted(scene_ranges.keys())

    for scene_num in sorted_scenes:
        scene_start, scene_end = scene_ranges[scene_num]

        # Collect words that fall within this scene's time range
        scene_words = [
            (ws, we, wt)
            for ws, we, wt in word_timestamps
            if ws >= scene_start - 0.05 and we <= scene_end + 0.05
        ]

        if not scene_words:
            log.warning("Scene %03d: no words found in range %.3fs–%.3fs — skipping.", scene_num, scene_start, scene_end)
            continue

        # Split scene words into small chunks of words_per_group
        for i in range(0, len(scene_words), words_per_group):
            chunk = scene_words[i:i + words_per_group]
            chunk_start = _seconds_to_ass_time(chunk[0][0])
            chunk_end   = _seconds_to_ass_time(chunk[-1][1])

            karaoke_parts: List[str] = []
            for ws, we, wt in chunk:
                duration_cs = max(1, int(round((we - ws) * 100)))
                karaoke_parts.append(f"{{\\k{duration_cs}}}{wt.upper()}")
            karaoke_text = " ".join(karaoke_parts)

            dialogue_lines.append(
                f"Dialogue: 0,{chunk_start},{chunk_end},Default,,0,0,0,,{karaoke_text}"
            )

    # Assemble full ASS content
    events_section = "\n".join([
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ] + dialogue_lines)

    content = header + "\n\n" + events_section + "\n"
    output_path.write_text(content, encoding="utf-8-sig")
    log.info("ass_generator: wrote %d dialogue lines to %s", len(dialogue_lines), output_path)
    return output_path
