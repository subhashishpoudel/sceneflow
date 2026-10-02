"""Parse scene.txt into per-scene image prompts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict


# Matches:  001 - A mountain at sunrise
#           001: A mountain at sunrise
#           [SCENE 001] A mountain at sunrise
_NUMBERED_RE = re.compile(r"^(\d+)\s*[-:]\s*(.+)$", re.MULTILINE)
_MARKER_RE   = re.compile(r"\[(?:SCENE|scene|Scene)[_\s]?(\d+)\]\s*(.+)$", re.MULTILINE)


class SceneParser:
    """
    Parses ``scene.txt`` into ``{scene_number: image_prompt}``.

    Supported formats (tried in order):

    1. **Numbered** — ``001 - A vast mountain landscape``
    2. **Marker-based** — ``[SCENE 001] A vast mountain landscape``
    3. **Line-based** — one prompt per non-empty line (numbered 1, 2, 3…)
    """

    @classmethod
    def parse(cls, scene_file: Path) -> Dict[int, str]:
        """
        Return ``{scene_number (1-based): image_prompt}``.
        Returns an empty dict if the file does not exist.
        """
        if not scene_file.is_file():
            return {}

        content = scene_file.read_text(encoding="utf-8", errors="replace")

        # Strategy 1: numbered lines  "001 - prompt"
        numbered = _NUMBERED_RE.findall(content)
        if numbered:
            return {int(num): prompt.strip() for num, prompt in numbered}

        # Strategy 2: marker-based  [SCENE 001] prompt
        marked = _MARKER_RE.findall(content)
        if marked:
            return {int(num): prompt.strip() for num, prompt in marked}

        # Strategy 3: plain lines — one prompt per line
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return {i + 1: ln for i, ln in enumerate(lines)}
