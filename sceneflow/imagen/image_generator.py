"""Orchestrates per-scene image generation for a project."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sceneflow.core.scene_parser import SceneParser
from sceneflow.tts.config import get_image_api_key, MissingApiKeyError
from sceneflow.imagen.imagen_client import generate_image

log = logging.getLogger(__name__)


def generate_all_images(
    project_path: Path,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """
    Generate ``images/NNN.jpg`` for every scene found in ``scene.txt``.

    Parameters
    ----------
    project_path:
        Root folder of the SceneFlow project (contains ``scene.txt``, ``images/``, …).
    force:
        When ``True``, overwrite existing ``images/NNN.jpg`` files.
        When ``False`` (default), skip scenes that already have an image.
    progress_callback:
        Optional callable ``(scene_number, total_scenes, status)`` where status
        is one of ``"done"``, ``"skipped"``, or ``"failed"``.

    Returns
    -------
    list[dict]
        One entry per scene: ``{"scene": int, "status": str, "error": str | None}``.
    """
    api_key = get_image_api_key()  # raises MissingApiKeyError if unset

    scene_file = project_path / "scene.txt"
    images_dir = project_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompts: Dict[int, str] = SceneParser.parse(scene_file)
    if not prompts:
        log.warning("No scenes found in scene.txt — nothing to generate.")
        return []

    sorted_nums = sorted(prompts.keys())
    total = len(sorted_nums)
    results: List[Dict] = []

    for scene_num in sorted_nums:
        jpg_path = images_dir / f"{scene_num:03d}.jpg"

        if jpg_path.exists() and not force:
            log.info("Scene %03d: skipping (image already exists).", scene_num)
            if progress_callback:
                progress_callback(scene_num, total, "skipped")
            results.append({"scene": scene_num, "status": "skipped", "error": None})
            continue

        prompt = prompts[scene_num]
        log.info("Scene %03d: generating image…", scene_num)

        try:
            image_bytes = generate_image(
                api_key=api_key,
                prompt=prompt,
            )
            jpg_path.write_bytes(image_bytes)
            log.info("Scene %03d: saved to %s", scene_num, jpg_path)
            if progress_callback:
                progress_callback(scene_num, total, "done")
            results.append({"scene": scene_num, "status": "done", "error": None})

        except Exception as exc:
            err_msg = str(exc)
            log.error("Scene %03d: image generation failed — %s", scene_num, err_msg)
            if progress_callback:
                progress_callback(scene_num, total, "failed", err_msg)
            results.append({"scene": scene_num, "status": "failed", "error": err_msg})

    return results
