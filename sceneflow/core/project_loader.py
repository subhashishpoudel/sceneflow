"""Load and validate a project folder into a ProjectData object."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from sceneflow.models.project import ProjectData
from sceneflow.models.scene import SceneData
from sceneflow.core.asset_manager import AssetManager
from sceneflow.core.script_parser import ScriptParser
from sceneflow.ffmpeg.audio_analyzer import AudioAnalyzer
from sceneflow.ffmpeg.filter_builder import assign_ken_burns
from sceneflow.models.render_config import RenderConfig

log = logging.getLogger(__name__)


class ProjectLoader:
    """
    Given a project root folder, discovers scenes and returns a
    fully-populated :class:`~sceneflow.models.project.ProjectData`.
    """

    # Canonical sub-folder names (checked case-insensitively on Windows)
    _IMAGE_FOLDER_NAMES = ("images", "image", "imgs", "img", "frames")
    _VOICE_FOLDER_NAMES = ("voice", "audio", "voices", "narration", "tts", "speech")

    def __init__(self, ffprobe_path: str) -> None:
        self._analyzer = AudioAnalyzer(ffprobe_path)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(
        self,
        project_path: Path,
        config: Optional[RenderConfig] = None,
    ) -> ProjectData:
        """
        Load all scenes from *project_path*.

        Returns a :class:`ProjectData` with ``validation_errors`` populated
        if anything is wrong (caller should check ``project.is_valid``).
        """
        if config is None:
            config = RenderConfig()

        data = ProjectData(project_path=project_path)

        # ---- locate sub-folders ----
        img_dir, voice_dir = self._find_folders(project_path)
        if img_dir is None:
            data.validation_errors.append(
                "No images/ folder found. "
                "Create a sub-folder named 'images' inside your project folder."
            )
            return data  # can't do anything without images

        # voice/ is optional at load time — the TTS feature can create it
        if voice_dir is None:
            # Use a placeholder path; asset discovery will find zero audio files
            voice_dir = project_path / "voice"
            data.validation_errors.append(
                "No voice/ folder found. "
                "Use 'Generate Voiceover' to create audio files automatically, "
                "or add a 'voice/' sub-folder with numbered .wav files."
            )

        data.image_folder = img_dir  # type: ignore[assignment]
        data.voice_folder = voice_dir  # type: ignore[assignment]

        # ---- detect assets ----
        images, audios, asset_errors = AssetManager.find_assets(img_dir, voice_dir)  # type: ignore[arg-type]
        # Only keep per-scene asset warnings — the "no voice folder" message is
        # already recorded above; skip the redundant per-scene "no audio" entries
        # when there are zero audio files at all (pre-voiceover state).
        if audios:
            data.validation_errors.extend(asset_errors)

        paired = AssetManager.paired_scene_numbers(images, audios)

        # ---- read audio durations for fully-paired scenes ----
        scenes: List[SceneData] = []
        for num in paired:
            try:
                duration = self._analyzer.get_duration(audios[num])
            except RuntimeError as exc:
                data.validation_errors.append(str(exc))
                continue
            scenes.append(
                SceneData(
                    scene_number=num,
                    image_path=images[num],
                    audio_path=audios[num],
                    duration_seconds=duration,
                    visible_padding_seconds=0.0,
                )
            )

        # ---- build image-only stubs for scenes that have no audio yet ----
        # These show up in the table as ✗ MISSING so the user knows to run
        # Generate Voiceover. They are excluded from rendering (is_valid stays
        # False) but allow the project to be visible and the button to work.
        unpaired_images = sorted(set(images) - set(audios))
        for num in unpaired_images:
            scenes.append(
                SceneData(
                    scene_number=num,
                    image_path=images[num],
                    audio_path=voice_dir / f"{num:03d}.wav",  # expected future path
                    duration_seconds=0.0,
                    visible_padding_seconds=0.0,
                )
            )
        scenes.sort(key=lambda s: s.scene_number)

        data.scenes = scenes

        # ---- Assign transition-visible padding to each scene.
        # Every scene except the LAST needs its image to stay visible for
        # an extra `transition_duration` seconds AFTER its audio finishes,
        # because the xfade transition runs across the boundary (the
        # transition begins exactly when scene N's audio ends, overlapping
        # the start of scene N+1's image). Without this padding, scene N's
        # still-frame loop ends exactly at audio end + the xfade would be
        # forced to start *before* audio end (the old bug), making every
        # image appear to "end early" during its own narration.
        trans_dur = max(0.0, config.transition_duration)
        n_scenes = len(scenes)
        if trans_dur > 0 and n_scenes > 1:
            # NOTE: the filter_builder (Ken Burns d-frame math) and
            # TimelineBuilder both look at scene.video_total_seconds, which
            # uses this field.
            for i, scene in enumerate(scenes):
                if i < n_scenes - 1:
                    scene.visible_padding_seconds = trans_dur
                else:
                    # Last scene has no outgoing transition — no padding.
                    scene.visible_padding_seconds = 0.0

        # ---- parse script.txt ----
        script_file = project_path / "script.txt"
        data.script_dict = ScriptParser.parse(script_file, len(scenes))
        for scene in scenes:
            scene.script_text = data.script_dict.get(scene.scene_number, "")

        # ---- write default subtitle template if not already present ----
        ass_template = project_path / "subtitle_template.ass"
        if not ass_template.exists():
            ass_template.write_text(
                "[Script Info]\n"
                "ScriptType: v4.00+\n"
                "PlayResX: 1920\n"
                "PlayResY: 1080\n"
                "\n"
                "[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,Montserrat,64,&H00FFFFFF,&H0000E0FF,&H00000000,"
                "&H00000000,-1,0,1,0,0,2,10,10,90,1\n",
                encoding="utf-8-sig",
            )
            log.info("Wrote default subtitle_template.ass to %s", project_path)

        # ---- apply Ken-Burns parameters ----
        assign_ken_burns(scenes, config)

        log.info(
            "Loaded project '%s': %d scenes, total %.1fs",
            project_path.name,
            len(scenes),
            data.total_duration,
        )
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_folders(
        self, project_path: Path
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate the images and voice sub-folders (case-insensitive)."""
        img_dir: Optional[Path] = None
        voice_dir: Optional[Path] = None

        for child in project_path.iterdir():
            if not child.is_dir():
                continue
            name_lower = child.name.lower()
            if img_dir is None and name_lower in self._IMAGE_FOLDER_NAMES:
                img_dir = child
            if voice_dir is None and name_lower in self._VOICE_FOLDER_NAMES:
                voice_dir = child

        return img_dir, voice_dir
