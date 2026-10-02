"""SceneData — data model for a single scene."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SceneData:
    """Holds every attribute describing one video scene.

    Duration convention (important for audio-visual sync):

    ``duration_seconds``
        Length of the associated voice audio only. Used for audio concat
        timing, :class:`ProjectData` total duration, render progress, and
        anything else that is driven by the audio track length.

    ``visible_padding_seconds``
        Extra seconds this scene's IMAGE / Ken-Burns clip should be shown
        *beyond* ``duration_seconds``. Primarily used so that the outgoing
        ``xfade`` transition between scene N and scene N+1 does not have
        to start *before* scene N's audio finishes (which would make the
        image appear to end early during the narration). The transition
        itself occupies the overlap region between the padded end of
        scene N and the padded start of scene N+1; the audio clips are
        still concatenated with no overlap, so they remain aligned with
        the correct scene image throughout.
    """

    scene_number: int
    image_path: Path
    audio_path: Path
    duration_seconds: float
    visible_padding_seconds: float = 0.0
    script_text: str = ""

    # Ken-Burns parameters (overridden by intensity slider)
    zoom_start: float = 1.0
    zoom_end: float = 1.15
    # Pan direction choice: "center" | "left" | "right" | "up" | "down"
    pan_direction: str = "center"

    @property
    def video_total_seconds(self) -> float:
        """Total length of the video clip (image visible on screen), equal
        to the scene's audio narration plus any transition padding.

        This is the value that should be used to size the image
        ``-loop -t`` input and the Ken-Burns zoompan ``d`` frame count.
        """
        return self.duration_seconds + max(0.0, self.visible_padding_seconds)

    @property
    def label(self) -> str:
        return f"Scene {self.scene_number:03d}"

    @property
    def duration_str(self) -> str:
        m, s = divmod(self.duration_seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"
