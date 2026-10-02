"""Parse script.txt into per-scene text blocks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


# Matches markers like:  [SCENE 001]  [Scene 1]  [scene001]  [SCENE_002]
_MARKER_RE = re.compile(
    r"\[(?:SCENE|scene|Scene)[_\s]?(\d+)\]", re.IGNORECASE
)


class ScriptParser:
    """
    Parses ``script.txt`` into ``{scene_number: text}`` using three
    strategies tried in order:

    1. **Marker-based** — ``[SCENE 001]`` … ``[SCENE 002]`` blocks.
    2. **Blank-line-separated** — paragraphs separated by blank lines.
    3. **Line-based** — one scene per non-empty line.
    """

    @classmethod
    def parse(cls, script_file: Path, num_scenes: int) -> Dict[int, str]:
        """
        Return ``{scene_number (1-based): script_text}``.

        *num_scenes* is used only for the fallback strategies to know how many
        lines/paragraphs to expect.
        """
        if not script_file.is_file():
            return {}

        content = script_file.read_text(encoding="utf-8", errors="replace")

        # Strategy 1: marker-based
        if _MARKER_RE.search(content):
            return cls._parse_markers(content)

        # Strategy 2: blank-line paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if len(paragraphs) >= 2:
            return {i + 1: p for i, p in enumerate(paragraphs)}

        # Strategy 3: one line per scene
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        return {i + 1: l for i, l in enumerate(lines)}

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_markers(content: str) -> Dict[int, str]:
        """Split on ``[SCENE NNN]`` markers."""
        segments: Dict[int, str] = {}
        matches = list(_MARKER_RE.finditer(content))

        for idx, match in enumerate(matches):
            scene_num = int(match.group(1))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            segments[scene_num] = text

        return segments
