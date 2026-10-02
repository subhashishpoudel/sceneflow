"""Build the complete FFmpeg filter_complex for an N-scene timeline."""
from __future__ import annotations

from typing import List, Tuple

from sceneflow.models.scene import SceneData
from sceneflow.models.render_config import RenderConfig
from sceneflow.ffmpeg.filter_builder import build_ken_burns_filter

# Supported xfade transitions (FFmpeg 4.3+)
TRANSITION_MAP = {
    "fade": "fade",
    "dissolve": "dissolve",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "none": None,         # hard cut — handled specially
}


class TimelineBuilder:
    """
    Assembles the FFmpeg ``-filter_complex`` string and returns the complete
    ``ffmpeg`` argument list for rendering.

    Input file ordering (interleaved image / audio):
      0 → image[0]   1 → audio[0]
      2 → image[1]   3 → audio[1]
      ...
      2n → image[n]  2n+1 → audio[n]
    """

    def __init__(self, scenes: List[SceneData], config: RenderConfig) -> None:
        self.scenes = scenes
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_ffmpeg_args(self) -> List[str]:
        """
        Return the full argument list for ``subprocess.run`` (without the
        ``ffmpeg`` executable itself).
        """
        w, h = self.config.resolution
        fps = self.config.fps
        trans_dur = self.config.transition_duration
        trans_type = TRANSITION_MAP.get(self.config.transition_type, "fade")

        # ---- build -i inputs ----
        inputs: List[str] = []
        for scene in self.scenes:
            # Use the FULL video length (audio duration + any outgoing
            # transition padding) when sizing the -loop still image.  This
            # is critical: scene N's image must remain visible during the
            # ENTIRE time its own audio plays, and then remain visible for
            # transition_duration additional seconds so the xfade cross-
            # fade happens after the narration (overlapping the start of
            # scene N+1) instead of during the narration.
            scene_video_len = getattr(scene, "video_total_seconds", scene.duration_seconds)
            if scene_video_len is None or scene_video_len <= 0:
                scene_video_len = scene.duration_seconds
            inputs += ["-loop", "1", "-t", str(scene_video_len), "-i", str(scene.image_path)]
            inputs += ["-i", str(scene.audio_path)]

        # ---- build filter_complex ----
        fc, final_v, final_a = self._build_filter_complex(
            fps, w, h, trans_dur, trans_type
        )

        # ---- subtitle burn-in (ASS hardcoded subtitles) ----
        sub_path = getattr(self.config, "subtitle_path", None)
        if sub_path:
            # Escape backslashes and colons for FFmpeg filter string on Windows
            escaped = sub_path.replace("\\", "/").replace(":", "\\:")
            # Append subtitles filter after the final video stream label
            fc = fc + f";{final_v}subtitles='{escaped}'[v_sub]"
            final_v = "[v_sub]"

        # ---- assemble final command ----
        args = (
            inputs
            + ["-filter_complex", fc]
            + ["-map", final_v]
            + ["-map", final_a]
            + ["-c:v", self.config.codec]
            + ["-preset", self.config.preset]
            + ["-c:a", self.config.audio_codec]
            + ["-b:a", self.config.audio_bitrate]
            + ["-movflags", "+faststart"]
            + ["-y"]           # overwrite without asking
        )

        if self.config.extra_flags.strip():
            import shlex
            args += shlex.split(self.config.extra_flags)

        args.append(self.config.output_path)
        return args

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filter_complex(
        self,
        fps: int,
        w: int,
        h: int,
        trans_dur: float,
        trans_type: str | None,
    ) -> Tuple[str, str, str]:
        """
        Returns (filter_complex_string, final_video_label, final_audio_label).
        """
        scenes = self.scenes
        n = len(scenes)

        parts: List[str] = []

        # ---- Per-scene Ken-Burns video streams ----
        scene_labels: List[str] = []
        for i, scene in enumerate(scenes):
            kb = build_ken_burns_filter(scene, (w, h), fps)
            lbl = f"[kb{i}]"
            parts.append(f"[{2*i}:v]{kb}{lbl}")
            scene_labels.append(lbl)

        # ---- Chain transitions or hard-cut concat ----
        if n == 1:
            # Trivial: single scene
            final_v = scene_labels[0]
        elif trans_type is None:
            # Hard-cut: use concat filter
            concat_inputs = "".join(scene_labels)
            parts.append(f"{concat_inputs}concat=n={n}:v=1:a=0[v_out]")
            final_v = "[v_out]"
        else:
            # xfade chain: [kb0][kb1] → xfade → [xf1]; [xf1][kb2] → xfade → [xf2]; …
            #
            # xfade CHAIN OFFSET SEMANTICS (critical for sync):
            #   Each chained xfade's :offset= is a timestamp on the FIRST
            #   input's OWN presentation timeline. Because every scene that
            #   participates in a transition is rendered for its FULL
            #   video_total_seconds (duration_seconds + visible_padding_
            #   seconds — see filter_builder.build_ken_burns_filter), that
            #   own timeline stays IDENTICAL to the true, uncompressed
            #   audio-concat timeline at every step of the chain: right up
            #   until a transition begins, the chain's output is simply the
            #   previous scene's unblended footage, whose own clock already
            #   equals absolute time. A transition consumes ONLY the
            #   preceding scene's padding tail — never its live narration —
            #   so nothing about the merged timeline is ever compressed by
            #   trans_dur. No correction term is needed anywhere here.
            #
            #   Rule:
            #     audio_boundary = cumulative + prev_scene.duration_seconds
            #       → the exact moment prev scene's audio ends and scene i's
            #         audio begins, in the (uncompressed) concatenated audio
            #         track.
            #     xfade :offset= = audio_boundary
            #       → the transition STARTS exactly when prev scene's audio
            #         finishes (never before), and runs entirely inside
            #         prev scene's visible_padding_seconds tail, ending at
            #         audio_boundary + trans_dur with scene i's image fully
            #         opaque — by which point scene i's own narration has
            #         already been playing for trans_dur seconds.
            #     cumulative advance = audio_boundary (no subtraction)
            #       → since the merged timeline is never compressed, the
            #         running anchor is just the plain cumulative sum of
            #         duration_seconds. Subtracting trans_dur here (as a
            #         previous version of this code did) makes every
            #         transition after the first start progressively
            #         EARLIER than it should — compounding by trans_dur per
            #         transition — which is what cuts each scene's image
            #         away while its own narration is still playing, and
            #         wastes the very padding frames rendered to prevent
            #         that.
            cumulative = 0.0
            prev_label = scene_labels[0]
            for i in range(1, n):
                prev_scene = scenes[i - 1]
                audio_boundary = cumulative + prev_scene.duration_seconds
                offset = audio_boundary
                out_label = f"[xf{i}]"
                parts.append(
                    f"{prev_label}{scene_labels[i]}"
                    f"xfade=transition={trans_type}"
                    f":duration={trans_dur}"
                    f":offset={offset:.4f}"
                    f"{out_label}"
                )
                cumulative = audio_boundary
                prev_label = out_label
            final_v = prev_label

        # ---- Audio concat ----
        audio_inputs = "".join(f"[{2*i+1}:a]" for i in range(n))
        parts.append(f"{audio_inputs}concat=n={n}:v=0:a=1[a_out]")
        final_a = "[a_out]"

        filter_complex = ";".join(parts)
        return filter_complex, final_v, final_a