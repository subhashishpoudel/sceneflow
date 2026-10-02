"""Build FFmpeg filter expressions for Ken-Burns effect and transitions."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

from sceneflow.models.scene import SceneData
from sceneflow.models.render_config import RenderConfig


# Pan patterns cycled through scenes to create visual variety
_PAN_PATTERNS = ["center", "left", "right", "up", "down", "left", "right"]


def assign_ken_burns(scenes: list, config: RenderConfig) -> None:
    """
    Mutate each *SceneData* in *scenes* to set ``zoom_start``, ``zoom_end``,
    and ``pan_direction`` based on *config.ken_burns_intensity*.

    ``intensity=0.0`` → no movement.
    ``intensity=1.0`` → maximum zoom/pan (20 % zoom swing, strong pan).
    """
    intensity = max(0.0, min(1.0, config.ken_burns_intensity))
    if intensity == 0.0:
        for s in scenes:
            s.zoom_start = 1.0
            s.zoom_end = 1.0
            s.pan_direction = "center"
        return

    max_zoom_delta = 0.25 * intensity   # up to 25 % zoom swing

    for i, scene in enumerate(scenes):
        # Alternate zoom-in / zoom-out per scene
        if i % 2 == 0:
            scene.zoom_start = 1.0
            scene.zoom_end = 1.0 + max_zoom_delta
        else:
            scene.zoom_start = 1.0 + max_zoom_delta
            scene.zoom_end = 1.0
        scene.pan_direction = _PAN_PATTERNS[i % len(_PAN_PATTERNS)]


def build_ken_burns_filter(
    scene: SceneData,
    resolution: Tuple[int, int],
    fps: int,
) -> str:
    """
    Return an FFmpeg filter string (``scale`` + ``zoompan``) that applies the
    Ken-Burns effect to one still image for exactly ``scene.duration_seconds``.

    Returned string is ready to be inserted between ``[input]`` and ``[output]``
    labels in a ``filter_complex``.
    """
    w, h = resolution
    # zoompan requires the image to be larger than the output so it can pan;
    # we scale up proportionally first.
    scale_factor = scene.zoom_end if scene.zoom_end >= scene.zoom_start else scene.zoom_start
    # Add a 5 % margin so the pan never reveals black borders
    scaled_w = int(w * scale_factor * 1.05)
    scaled_h = int(h * scale_factor * 1.05)

    # Use the FULL video length (audio duration + any outgoing transition
    # padding) so the zoompan animation continues to generate frames for
    # the duration of the xfade overlap. If we used `duration_seconds`
    # here, the filter would run out of frames during the transition,
    # producing a black / frozen fade instead of a true image-to-image
    # crossfade.
    video_length = getattr(scene, "video_total_seconds", scene.duration_seconds)
    if video_length is None or video_length <= 0:
        video_length = scene.duration_seconds
    d = max(1, int(round(video_length * fps)))

    # Zoom expression
    zoom_delta = scene.zoom_end - scene.zoom_start
    if abs(zoom_delta) < 0.001:
        zoom_expr = f"{scene.zoom_start:.4f}"
    else:
        # linear ramp from zoom_start → zoom_end
        zoom_expr = f"'{scene.zoom_start:.4f}+{zoom_delta:.4f}*on/{d}'"

    # Pan X/Y expressions (output window size is w×h at the zoomed level)
    # 'ow' and 'oh' are output width/height inside zoompan (= w and h).
    # 'iw' and 'ih' are input width/height (= scaled_w and scaled_h).
    pan = scene.pan_direction
    if pan == "center":
        x_expr = "'(iw-ow)/2'"
        y_expr = "'(ih-oh)/2'"
    elif pan == "left":
        # Slide from center to slightly left
        x_expr = f"'(iw-ow)/2 - {int(scaled_w*0.05)}*on/{d}'"
        y_expr = "'(ih-oh)/2'"
    elif pan == "right":
        x_expr = f"'(iw-ow)/2 + {int(scaled_w*0.05)}*on/{d}'"
        y_expr = "'(ih-oh)/2'"
    elif pan == "up":
        x_expr = "'(iw-ow)/2'"
        y_expr = f"'(ih-oh)/2 - {int(scaled_h*0.05)}*on/{d}'"
    elif pan == "down":
        x_expr = "'(iw-ow)/2 + 0'"
        y_expr = f"'(ih-oh)/2 + {int(scaled_h*0.05)}*on/{d}'"
    else:
        x_expr = "'(iw-ow)/2'"
        y_expr = "'(ih-oh)/2'"

    # Compose: scale → zoompan → trim to exact frame count → setsar
    # trim is essential: zoompan buffers frames internally and can output
    # more frames than d if the input loops; trim hard-caps the output.
    filters = [
        f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase",
        f"crop={scaled_w}:{scaled_h}",
        (
            f"zoompan=z={zoom_expr}"
            f":x={x_expr}"
            f":y={y_expr}"
            f":d={d}"
            f":fps={fps}"
            f":s={w}x{h}"
        ),
        f"trim=end_frame={d}",
        f"setsar=1",
    ]
    return ",".join(filters)
