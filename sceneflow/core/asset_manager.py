"""Detect and pair image/audio assets inside a project folder."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}

# Matches a leading integer in the stem: "001", "002", "scene_003", "003_v2"
_SCENE_NUM_RE = re.compile(r"^(\d+)")


def extract_scene_number(filename: str) -> Optional[int]:
    """
    Return the scene number encoded in a filename stem, or *None*.

    Examples::

        extract_scene_number("001.png")       → 1
        extract_scene_number("003_final.wav") → 3
        extract_scene_number("scene.png")     → None
    """
    stem = Path(filename).stem
    m = _SCENE_NUM_RE.match(stem)
    if m:
        return int(m.group(1))
    return None


def _collect(folder: Path, extensions: set) -> Dict[int, Path]:
    """
    Scan *folder* for files whose stems start with a scene number and whose
    extension is in *extensions*.  Returns ``{scene_number: path}``.
    Duplicates (two files sharing the same scene number) keep the first one
    found alphabetically.
    """
    result: Dict[int, Path] = {}
    if not folder.is_dir():
        return result
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in extensions:
            continue
        num = extract_scene_number(f.name)
        if num is None:
            continue
        if num not in result:
            result[num] = f
    return result


class AssetManager:
    """Discovers and validates image + audio assets for all scenes."""

    @staticmethod
    def find_assets(
        images_dir: Path,
        voice_dir: Path,
    ) -> Tuple[Dict[int, Path], Dict[int, Path], List[str]]:
        """
        Return ``(image_dict, audio_dict, errors)``.

        *errors* is a list of human-readable problem descriptions; it is empty
        when every detected scene has both an image and an audio file.
        """
        images = _collect(images_dir, SUPPORTED_IMAGE_EXTS)
        audios = _collect(voice_dir, SUPPORTED_AUDIO_EXTS)
        errors: List[str] = []

        all_nums = sorted(set(images) | set(audios))
        for num in all_nums:
            has_img = num in images
            has_aud = num in audios
            if has_img and not has_aud:
                errors.append(
                    f"Scene {num:03d}: image found ({images[num].name}) "
                    f"but no matching audio file."
                )
            elif has_aud and not has_img:
                errors.append(
                    f"Scene {num:03d}: audio found ({audios[num].name}) "
                    f"but no matching image file."
                )

        return images, audios, errors

    @staticmethod
    def paired_scene_numbers(
        images: Dict[int, Path],
        audios: Dict[int, Path],
    ) -> List[int]:
        """Return sorted list of scene numbers that have BOTH assets."""
        return sorted(set(images) & set(audios))
