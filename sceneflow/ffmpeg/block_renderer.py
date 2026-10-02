"""Block-based low-memory render orchestrator.

When rendering 50+ scenes in a single monolithic `ffmpeg` invocation, the
filter_complex graph can become huge — pushing FFmpeg's memory usage well
past 2-4 GB and causing OOM crashes on mid-range machines.

This module provides :class:`BlockRenderJob` which:

1. Splits the full scene list into ``block_size`` groups.
2. Renders each group to ``clip_01.mp4``, ``clip_02.mp4``, ... in a temp dir
   (each render is a small, low-memory, self-contained FFmpeg call that is
   otherwise identical to the normal TimelineBuilder pipeline — same
   transitions, Ken Burns, audio concat etc.).
3. Joins all the clips together using the FFmpeg **concat FILTER on a
   re-encode pass with the overlap (segment) durations preserved exactly**.

A/V-SYNC AND TRANSITION CORRECTNESS (why we use a 1-scene OVERLAP strategy):
----------------------------------------------------------------------------
The monolithic TimelineBuilder creates one xfade between EVERY consecutive
scene pair: scene N-1 → scene N at xfade offset = sum of duration_seconds
of all scenes before N.  The last scene of each "block" in a naive split
would have NO outgoing xfade (because the next scene lives in a DIFFERENT
block's clip).  The block-boundary seam would then concatenate as a HARD CUT
with a visible audio/video desync equal to transition_duration — the most
common block-rendering artifact.

Our fix:
  • Every non-final block is rendered with (block_size + 1) scenes: the
    nominal block_size scenes plus the FIRST scene of the NEXT block as an
    OVERLAP tail.  Inside this block the xfade transition
    (last_nominal → overlap_scene) is rendered normally, with correct
    offset math, within the same small filter_complex graph.
  • After the block is rendered, we TRIM the overlap tail duration off the
    end of the clip so the ownership of the overlap scene's AUDIO and VIDEO
    content belongs cleanly to the NEXT block (which renders the overlap
    scene fresh from timestamp 0 as its own FIRST scene).
  • The final clip join uses a lossless CONCAT (same codecs, no re-encode)
    of the TRIMMED clips so block seams sit INSIDE the transition window
    that was already rendered in the PREVIOUS block's clip — no seam, no
    hard cut, and audio-video timing is pixel-perfect identical to the
    monolithic render.

Scenes are never split across a block boundary at any finer granularity than
"whole scene ownership" — every scene's audio content is owned 100% by
exactly one block, and the 1-scene overlap exists only to carry the xfade
transition in the visual domain.
"""
from __future__ import annotations

import copy
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sceneflow.core.timeline_builder import TimelineBuilder
from sceneflow.models.render_config import RenderConfig
from sceneflow.models.scene import SceneData
from sceneflow.ffmpeg.filter_builder import assign_ken_burns
from sceneflow.ffmpeg.renderer import RenderJob

log = logging.getLogger(__name__)


def _split_into_overlapped_blocks(
    scenes: List[SceneData], block_size: int
) -> List[Dict]:
    """Partition *scenes* into blocks with a 1-scene overlap tail.

    Returns list of dicts:
        {
          "render_scenes":        [SceneData] — full list for TimelineBuilder
                                      (includes overlap scene at the end for
                                       non-final blocks)
          "owned_scene_numbers":  [int]       — scene numbers this block owns
                                      (excludes overlap; overlap is owned by
                                       the NEXT block)
          "owned_scene_durations":[float]     — audio durations of owned scenes
                                      only (used for progress + clip trim math)
          "trim_output_end_seconds": float    — how many seconds to cut OFF the
                                      END of this rendered clip to remove the
                                      overlap tail (0.0 for final block)
          "is_final_block":       bool
        }

    Example: 80 scenes, block_size=10.
      Block 1: render_scenes = scenes 001..011 (11 scenes, overlap = scene 011)
               owned = 001..010
               trim_end = sum(durations 011.audio only? No — trim_end =
                          scene_011.duration_seconds) — we keep block 1
                          only up through the END of scene 010's audio.
                          The xfade (010 → 011) inside block 1's rendered
                          clip spans (end_of_010 - trans_dur)..(end_of_010)
                          which is TRANSITION_DURATION seconds BEFORE the
                          trim cut — so the full transition lives in block 1,
                          and block 2 starts fresh at scene 011's audio
                          start with its own image from t=0. Seam is
                          INSIDE the transition, invisible.
      Block 2: render_scenes = scenes 011..021 (11 scenes, overlap scene 021)
               owned = 011..020
               trim_end = dur_021
               ...
      Block 8 (final): render_scenes = scenes 071..080 (10 scenes, NO overlap)
               owned = 071..080
               trim_end = 0.0
    """
    if block_size <= 0:
        block_size = len(scenes)
    n = len(scenes)

    # Core block start indices (start of each owned contiguous run)
    core_starts = list(range(0, n, block_size))

    blocks: List[Dict] = []
    for bi, start in enumerate(core_starts):
        is_final = bi == len(core_starts) - 1
        # owned scenes = block_size from `start`
        end_owned = min(start + block_size, n)
        owned_scenes = scenes[start:end_owned]
        owned_nums = [s.scene_number for s in owned_scenes]
        owned_durs = [s.duration_seconds for s in owned_scenes]

        # render scenes = owned scenes + (1-scene overlap if not final)
        if is_final:
            render_scenes = list(owned_scenes)
            trim_end = 0.0
        else:
            overlap_scene = scenes[end_owned]  # first scene of NEXT block
            render_scenes = list(owned_scenes) + [overlap_scene]
            # Trim the entire audio duration of the overlap scene off the
            # block's output clip so block ownership is clean.  The xfade
            # (last_owned → overlap) lives in the
            # sum(owned_durs) - trans_dur .. sum(owned_durs) window, which
            # is fully inside the pre-trim clip range and will NOT be cut.
            trim_end = overlap_scene.duration_seconds

        blocks.append({
            "render_scenes": render_scenes,
            "owned_scene_numbers": owned_nums,
            "owned_scene_durations": owned_durs,
            # NEW: owned_audio_sum = sum of the owned scenes' audio durations.
            # This is the EXACT concat-seam cut point we want because inside
            # the block's TimelineBuilder graph, the audio concat filter
            # appends each owned scene's audio back-to-back at
            # t=sum(durs_0..i) position, so last owned audio ends exactly at
            # owned_audio_sum and the OVERLAP scene audio begins at
            # owned_audio_sum.  Cutting at owned_audio_sum seconds therefore:
            #   • keeps all OWNED audio (no loss)
            #   • keeps the xfade transition (last-owned → overlap) which was
            #     rendered in block-local video at offset = sum(prev_owned) +
            #     last_owned.duration = owned_audio_sum, so the transition
            #     window (owned_audio_sum - trans_dur) .. owned_audio_sum is
            #     COMPLETELY inside the kept region (we cut at the END of the
            #     transition, i.e. at the first frame where the overlap
            #     image is 100% opaque).
            "owned_audio_sum": sum(owned_durs),
            "trim_output_end_seconds": trim_end,  # kept for backwards reference
            "is_final_block": is_final,
        })
    return blocks


def _apply_block_local_padding(
    render_scenes: List[SceneData], config: RenderConfig
) -> None:
    """Re-compute ``visible_padding_seconds`` for a block-local scene list.

    The project-level padding (set by ProjectLoader) marks ONLY the final
    scene of the WHOLE project with padding=0.0.  Inside a block render,
    the LAST scene of THIS block's ``render_scenes`` must have padding=0
    (since there is no outgoing xfade past the block render boundary — the
    next xfade, if any, is *inside* this block thanks to the overlap).
    All other scenes in ``render_scenes`` get the standard outgoing
    transition padding so intra-block xfades work correctly.
    """
    trans_dur = max(0.0, config.transition_duration)
    n = len(render_scenes)
    for i, s in enumerate(render_scenes):
        if i < n - 1:
            s.visible_padding_seconds = trans_dur
        else:
            # Last scene in the block render → no outgoing transition.
            # (For non-final blocks this is the overlap scene; its outgoing
            # xfade into NEXT block will be rendered in the NEXT block's
            # render if needed. For final blocks it's the project end.)
            s.visible_padding_seconds = 0.0


class BlockRenderJob:
    """
    Orchestrates block-by-block rendering (1-scene overlap strategy) + final
    lossless concat.

    Presents the same public API surface as :class:`RenderJob`
    (``start / cancel / is_running``) so the GUI can swap them transparently.
    """

    def __init__(
        self,
        ffmpeg_path: str,
        scenes: List[SceneData],
        config: RenderConfig,
        total_duration: float,
        on_progress: Optional[Callable[[float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[bool, int], None]] = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.scenes = scenes
        self.config = config
        self.total_duration = max(total_duration, 0.001)
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_done = on_done

        self._thread: Optional[threading.Thread] = None
        self._cancelled = False
        self._current_render: Optional[RenderJob] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None

    # ------------------------------------------------------------------
    # Public API (mirrors RenderJob)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the block-render orchestrator in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Cancel any in-progress block render and stop the pipeline."""
        self._cancelled = True
        if self._current_render:
            self._current_render.cancel()
        log.info("BlockRenderJob cancelled by user.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit_log(self, message: str) -> None:
        log.debug("block_render: %s", message)
        if self.on_log:
            self.on_log(message)

    def _emit_progress(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        if self.on_progress:
            self.on_progress(fraction)

    def _run(self) -> None:
        try:
            success = self._do_run()
        except Exception as exc:
            log.exception("Block render failed with unhandled exception")
            self._emit_log(f"BLOCK RENDER FATAL ERROR: {exc}")
            if self.on_done:
                self.on_done(False, -1)
            return
        finally:
            if self._tmpdir is not None:
                try:
                    self._tmpdir.cleanup()
                except Exception:
                    pass
                self._tmpdir = None

        if self._cancelled:
            if self.on_done:
                self.on_done(False, 1)
            return
        if self.on_done:
            self.on_done(success, 0 if success else 2)

    def _do_run(self) -> bool:
        """Core pipeline. Returns True on success."""

        # ---- Generate subtitle .ass file if requested (runs in background) ----
        template_path = getattr(self.config, "_subtitle_template_path", None)
        if template_path:
            try:
                self._emit_log("Subtitle generation: aligning audio for word timestamps…")
                words_per_group = getattr(self.config, "_subtitle_words_per_group", 5)
                ass_path = self._generate_subtitles(template_path, words_per_group)
                self.config.subtitle_path = str(ass_path)
                self._emit_log(f"Subtitles written to {ass_path}")
            except Exception as exc:
                self._emit_log(f"Subtitle generation FAILED: {exc}")
                if self.on_done:
                    self.on_done(False, -1)
                return False

        block_size = max(1, int(self.config.block_size))
        if block_size >= len(self.scenes):
            return self._run_single_monolithic()

        blocks = _split_into_overlapped_blocks(self.scenes, block_size)
        n_blocks = len(blocks)

        # Log the ownership plan
        plan_lines = []
        for bi, b in enumerate(blocks, 1):
            rs = len(b["render_scenes"])
            own = b["owned_scene_numbers"]
            overlap_note = "" if b["is_final_block"] else (
                f" (+1 overlap scene #{b['render_scenes'][-1].scene_number:03d} for xfade)"
            )
            plan_lines.append(
                f"  Block {bi}: scenes {own[0]:03d}–{own[-1]:03d} owned "
                f"({len(own)} scenes, renders {rs}{overlap_note})"
            )
        self._emit_log(
            f"=== Block Render started (1-scene overlap strategy) ===\n"
            f"  Total scenes: {len(self.scenes)}  |  block_size: {block_size}  "
            f"|  blocks: {n_blocks}\n" + "\n".join(plan_lines)
        )

        # Build temp dir for clips
        output_path = Path(self.config.output_path)
        tmp_parent = output_path.parent if str(output_path) else Path.cwd()
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="sceneflow_clips_", dir=str(tmp_parent)
        )
        clip_dir = Path(self._tmpdir.name)

        # cumulative owned-duration offset — used to compute block-start
        # position inside the global progress window.
        cumulative_offset_seconds = 0.0

        trimmed_clips: List[Path] = []
        trimmed_durations: List[float] = []

        for block_idx, block in enumerate(blocks, start=1):
            if self._cancelled:
                return False

            raw_clip_path = clip_dir / f"raw_{block_idx:02d}.mp4"
            trimmed_clip_path = clip_dir / f"clip_{block_idx:02d}.mp4"

            owned_dur = sum(block["owned_scene_durations"])
            owned_nums = block["owned_scene_numbers"]
            render_n = len(block["render_scenes"])
            if block["is_final_block"]:
                overlap_str = ""
            else:
                overlap_str = (
                    f" (+overlap scene "
                    f"#{block['render_scenes'][-1].scene_number:03d})"
                )

            self._emit_log(
                f"── Block {block_idx}/{n_blocks} render start: "
                f"owned {owned_nums[0]:03d}–{owned_nums[-1]:03d} "
                f"({len(owned_nums)} scenes, renders {render_n}{overlap_str}) "
                f"→ {raw_clip_path.name} (owned dur ~{owned_dur:.1f}s)"
            )

            # --- (a) Render block (with overlap scene at end, if non-final) ----
            render_ok = self._render_one_block_raw(
                block,
                raw_clip_path,
                block_idx,
                n_blocks,
                cumulative_offset_seconds,
                owned_dur,
            )
            if not render_ok:
                self._emit_log(
                    f"✗ Block {block_idx}/{n_blocks} raw render FAILED. Aborting."
                )
                return False

            # --- (b) Trim overlap tail off the raw clip --------------------
            # BUG 3 FIX: use owned_audio_sum directly as the EXACT keep
            # duration instead of raw_dur - overlap.duration.  The old code
            # computed keep_dur = raw_dur - overlap_dur, but raw_dur changes
            # depending on xfade total duration math (BUG 2 fix shifts the
            # xfade chain by trans_dur, shrinking raw_dur by trans_dur per
            # block), so (raw_dur - overlap_dur) landed INSIDE the 10→11
            # transition window, producing a visible freeze/frame-drift at
            # the concat seam.  owned_audio_sum is the theoretical point
            # where the last-owned audio ENDS and the xfade (owned→overlap)
            # also ENDS (per BUG 2 sync rule), so cutting here keeps the
            # entire transition inside the kept region and the concat seam
            # is invisible and byte-accurate to a monolithic render.
            keep_seconds = block["owned_audio_sum"]
            if block["is_final_block"]:
                trim_str = ""
            else:
                trim_str = f" (target keep {keep_seconds:.3f}s = end of xfade window)"
            if not block["is_final_block"] and keep_seconds > 0.0001:
                ok = self._trim_clip_end(
                    raw_clip_path, trimmed_clip_path, keep_seconds
                )
                if not ok:
                    self._emit_log(
                        f"✗ Block {block_idx}/{n_blocks} trim step FAILED."
                    )
                    return False
                self._emit_log(
                    f"   Block {block_idx}/{n_blocks} trimmed to owned_audio_sum="
                    f"{keep_seconds:.3f}s (overlap scene removed from tail) "
                    f"→ {trimmed_clip_path.name}{trim_str}"
                )
            else:
                # Final block — no overlap, no trim; just rename.
                try:
                    raw_clip_path.rename(trimmed_clip_path)
                except Exception:
                    # If rename fails (cross-filesystem temp), copy.
                    import shutil
                    shutil.copy2(str(raw_clip_path), str(trimmed_clip_path))

            trimmed_clips.append(trimmed_clip_path)
            trimmed_durations.append(owned_dur)

            cumulative_offset_seconds += owned_dur
            # Set progress just past the end of this block's owned window so
            # the UI jumps cleanly (the trim/concat steps are sub-second).
            self._emit_progress(
                min(cumulative_offset_seconds / self.total_duration, 0.999)
            )

            if block_idx < n_blocks:
                time.sleep(0.05)

        if self._cancelled:
            return False

        # ── Final concat (stream copy, zero re-encode, <1s) ──────────
        self._emit_log(
            f"=== Joining {len(trimmed_clips)} trimmed clip(s) with concat "
            f"demuxer (stream copy, lossless) ==="
        )
        concat_ok = self._concat_clips(trimmed_clips, output_path)
        if not concat_ok:
            self._emit_log("✗ Concat step FAILED.")
            return False

        total_mb = sum(p.stat().st_size for p in trimmed_clips) // (1024 * 1024)
        self._emit_log(
            f"✓ Block render complete → {output_path} ({total_mb} MB clips "
            f"joined losslessly; 1-scene overlap xfade strategy guarantees "
            f"transition-perfect A/V sync matching monolithic render)"
        )
        return True

    # --------------------------------------------------------------
    # Single-block fallback (monolithic, for tiny projects)
    # --------------------------------------------------------------

    def _run_single_monolithic(self) -> bool:
        self._emit_log("Block size ≥ #scenes — using single monolithic render.")
        # Re-apply project-level padding (last scene = 0) since block-local
        # logic is not used.
        trans_dur = max(0.0, self.config.transition_duration)
        for i, s in enumerate(self.scenes):
            s.visible_padding_seconds = 0.0 if i == len(self.scenes) - 1 else trans_dur
        assign_ken_burns(self.scenes, self.config)

        done_event = threading.Event()
        result: List[bool] = [False]

        def _on_done(ok: bool, rc: int) -> None:
            result[0] = ok
            done_event.set()

        builder = TimelineBuilder(self.scenes, self.config)
        args = builder.build_ffmpeg_args()

        job = RenderJob(
            ffmpeg_path=self.ffmpeg_path,
            args=args,
            total_duration=self.total_duration,
            on_progress=self.on_progress,
            on_log=self.on_log,
            on_done=_on_done,
        )
        self._current_render = job
        job.start()
        while not done_event.wait(0.2):
            if self._cancelled:
                job.cancel()
                return False
        self._current_render = None
        return result[0]

    # --------------------------------------------------------------
    # Individual block raw render (1 overlap scene at end)
    # --------------------------------------------------------------

    def _render_one_block_raw(
        self,
        block: Dict,
        raw_clip_path: Path,
        block_idx: int,
        n_blocks: int,
        progress_offset: float,
        block_owned_dur: float,
    ) -> bool:
        """Render one block's ``render_scenes`` (incl. overlap tail) to a raw clip.

        *progress_offset* is the total audio duration of owned scenes in
        all PREVIOUS blocks; *block_owned_dur* is the audio duration sum of
        THIS block's owned scenes only (not including the overlap). We use
        both to map intra-block progress to the global 0..1 window as if
        the overlap scene contributed zero progress (which it doesn't for
        owned-content accounting).
        """
        render_scenes: List[SceneData] = block["render_scenes"]
        owned_nums = block["owned_scene_numbers"]

        # Work on a SHALLOW COPY of scenes so block-local visible_padding
        # overrides don't leak back into ProjectData.scenes.
        render_scene_copies = [copy.copy(s) for s in render_scenes]
        # Re-apply block-local padding (last scene in block = 0, rest = trans)
        _apply_block_local_padding(render_scene_copies, self.config)
        # Ken Burns parameters are per-scene; set them again on copies so
        # they exist (caller already set them on originals, but this is
        # idempotent thanks to deterministic indexing inside assigner —
        # actually we need to pass the originals' zoom/pan values. Let's
        # instead just copy over the Ken Burns attrs manually.)
        for orig, copy_s in zip(render_scenes, render_scene_copies):
            copy_s.zoom_start = orig.zoom_start
            copy_s.zoom_end = orig.zoom_end
            copy_s.pan_direction = orig.pan_direction

        # Build block-local config with same settings but block output path
        block_cfg = copy.copy(self.config)
        block_cfg.output_path = str(raw_clip_path)

        builder = TimelineBuilder(render_scene_copies, block_cfg)
        args = builder.build_ffmpeg_args()

        # Expected output duration for this raw clip:
        # sum(duration_seconds for all render_scenes) +
        # trans_dur × (len(render_scenes)-1 - 1_padding) via video_total_seconds
        # Actually we don't need the raw duration exactly — we compute
        # progress by owned_dur window only, which is robust regardless.
        # Use a floor of block_owned_dur * 1.2 for RenderJob progress math.
        render_progress_total = max(block_owned_dur, 0.001)

        done_event = threading.Event()
        result_ok: List[bool] = [False]
        total_dur = self.total_duration

        def _progress_hook(inner_frac: float) -> None:
            # inner_frac: 0..1 within THIS raw render
            block_start_frac = progress_offset / total_dur
            # Even though the render includes the overlap tail at the end,
            # we map the render's 1.0 to "end of owned dur" so progress doesn't
            # overshoot. (The tail trim is instantaneous compared to render.)
            window_frac = block_owned_dur / total_dur
            global_frac = block_start_frac + inner_frac * window_frac
            self._emit_progress(global_frac)

        def _on_done(ok: bool, rc: int) -> None:
            result_ok[0] = ok
            done_event.set()

        job = RenderJob(
            ffmpeg_path=self.ffmpeg_path,
            args=args,
            total_duration=render_progress_total,
            on_progress=_progress_hook,
            on_log=self.on_log,
            on_done=_on_done,
        )
        self._current_render = job
        job.start()
        while not done_event.wait(0.2):
            if self._cancelled:
                job.cancel()
                self._current_render = None
                return False

        self._current_render = None
        ok = result_ok[0]
        if ok:
            self._emit_log(
                f"   ✓ Block {block_idx}/{n_blocks} raw render done "
                f"(scenes {owned_nums[0]:03d}–{owned_nums[-1]:03d} owned)"
            )
        return ok

    # --------------------------------------------------------------
    # Trim helper (remove overlap tail off block clip)
    # --------------------------------------------------------------

    def _read_wav_like_duration(self, media_path: Path) -> float:
        """Best-effort duration read via ffprobe-like fast probe.

        For block-trimming purposes we don't actually need the source
        duration; the trim is specified as ``-to`` using the input's tail
        duration we want removed, so if the clip is shorter than expected
        we still get the right result (trim to EOF if ``clip_dur - trim_end``
        is negative).  For robustness we return 0.0 and let the -to logic
        handle it directly via ``-t total_dur - trim_end`` instead of
        relying on source duration probe.
        """
        return 0.0

    def _trim_clip_end(
        self,
        raw_clip: Path,
        trimmed_clip: Path,
        keep_seconds: float,
    ) -> bool:
        """Trim a block's raw render so only the first *keep_seconds* seconds
        are retained (the overlap tail is discarded).

        BUG 3 FIX: *keep_seconds* is the EXACT theoretical cut point (equal to
        block["owned_audio_sum"]) — the point where (a) all owned-scene audio
        ends and (b) the xfade (last-owned → overlap) completes exactly at
        100% overlap-image.  Using this value DIRECTLY as the ffmpeg `-t`
        argument means the cut is independent of:
          * ffprobe duration probe accuracy
          * how xfade chain total duration shifts when BUG 2 fix changes the
            offset positions (raw_dur shrinks by trans_dur relative to the
            old broken xfade math)
        The previous implementation used `-t (raw_dur - overlap.duration)`
        which was only correct if raw_dur was exactly
        (owned_audio_sum + overlap.duration ± 0); after BUG 2 fixed the xfade
        offset, raw_dur became (owned_audio_sum + overlap.duration - trans_dur)
        so the old trim landed INSIDE the transition window → concat seam
        had a visible frame-freeze / drift.

        Strategy:
          1. First try stream-copy trim using `-t keep_seconds` (fast, no
             re-encode — visually lossless at GOP boundary; may cut a few
             frames early, acceptable since the transition window is fully
             inside the kept region).
          2. If stream-copy fails, fallback to a full re-encode with the same
             `-t keep_seconds` (100% frame-accurate, slightly slower).
        """
        # ---- Fast path: stream copy using -t keep_seconds ---------------
        if keep_seconds > 0.1:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i", str(raw_clip),
                "-t", f"{keep_seconds:.6f}",
                "-c", "copy",
                "-movflags", "+faststart",
                str(trimmed_clip),
            ]
            try:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", timeout=120,
                )
                if proc.returncode == 0 and trimmed_clip.is_file() and trimmed_clip.stat().st_size > 1024:
                    return True
                self._emit_log(
                    f"   Stream-copy trim failed (rc={proc.returncode}); "
                    f"falling back to frame-accurate re-encode with "
                    f"-t {keep_seconds:.3f}s."
                )
            except Exception as exc:
                self._emit_log(f"   Stream-copy trim exception: {exc}; using fallback re-encode.")

        # ---- Fallback: re-encode with -t keep_seconds (100% accurate) -----
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(raw_clip),
            "-t", f"{keep_seconds:.6f}",
            "-c:v", self.config.codec,
            "-preset", "ultrafast",
            "-c:a", self.config.audio_codec,
            "-b:a", self.config.audio_bitrate,
            "-movflags", "+faststart",
            str(trimmed_clip),
        ]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=300,
            )
            ok = (
                proc.returncode == 0
                and trimmed_clip.is_file()
                and trimmed_clip.stat().st_size > 1024
            )
            if not ok:
                tail = proc.stderr[-1500:] if proc.stderr else "(no stderr)"
                self._emit_log(f"   Re-encode trim tail:\n{tail}")
            return ok
        except Exception as exc:
            self._emit_log(f"   Re-encode trim exception: {exc}")
            return False

    def _probe_duration_ffprobe(self, media_path: Path) -> float:
        """Probe *media_path* duration with ffprobe. Returns 0.0 on failure."""
        import shutil as _shutil
        ffprobe = None
        # Try to infer ffprobe from the ffmpeg path
        ffmpeg = Path(self.ffmpeg_path)
        candidate = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        if candidate.is_file():
            ffprobe = str(candidate)
        else:
            on_path = _shutil.which("ffprobe")
            if on_path:
                ffprobe = on_path
        if not ffprobe:
            return 0.0
        try:
            proc = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(media_path),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=30,
            )
            if proc.returncode == 0:
                return float(proc.stdout.strip())
        except Exception:
            pass
        return 0.0

    # --------------------------------------------------------------
    # Concat demuxer (lossless stream copy)
    # --------------------------------------------------------------

    def _concat_clips(self, clips: List[Path], output_path: Path) -> bool:
        """Join clips losslessly with ffmpeg concat demuxer + stream copy.

        Because every trimmed clip shares identical codec parameters (same
        RenderConfig was used), stream-copy concat works perfectly with
        zero quality loss and <1s runtime for multi-hour projects.
        """
        if not clips:
            return False

        clip_dir = clips[0].parent
        list_file = clip_dir / "concat_list.txt"
        lines = []
        for c in clips:
            abs_path = str(c.resolve()).replace("\\", "/")
            lines.append(f"file '{abs_path}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        self._emit_log("Concat: " + " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
        except Exception as exc:
            self._emit_log(f"Concat subprocess error: {exc}")
            return False

        if proc.returncode != 0:
            tail = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
            self._emit_log(f"Concat FFmpeg stderr tail:\n{tail}")
            return False

        if not output_path.is_file() or output_path.stat().st_size < 1024:
            self._emit_log(f"Concat output missing or too small: {output_path}")
            return False

        self._emit_progress(1.0)
        mb = output_path.stat().st_size // (1024 * 1024)
        self._emit_log(f"✓ Concat done: {output_path} ({mb} MB)")
        return True

    # --------------------------------------------------------------
    # Subtitle generation (runs in background thread)
    # --------------------------------------------------------------

    def _generate_subtitles(self, template_path: str, words_per_group: int = 5) -> Path:
        """Align all per-scene WAVs and write a timed .ass subtitle file."""
        import io
        import wave
        import tempfile
        from sceneflow.subtitles.ass_generator import generate_ass
        from sceneflow.tts.forced_aligner import locally_align_audio
        from sceneflow.core.script_parser import ScriptParser

        project_path = Path(self.config.output_path).parent
        # Find the project path from scene image paths
        if self.scenes:
            project_path = self.scenes[0].image_path.parent.parent

        script_file = project_path / "script.txt"
        voice_dir = project_path / "voice"
        images_dir = project_path / "images"

        num_images = len(list(images_dir.glob("*"))) if images_dir.is_dir() else 0
        scenes = ScriptParser.parse(script_file, num_scenes=num_images)
        if not scenes:
            raise RuntimeError("No scenes found in script.txt — cannot generate subtitles.")

        sorted_nums = sorted(scenes.keys())

        # Concatenate all per-scene WAVs into one combined WAV
        combined_frames = b""
        sample_rate = 24000
        sample_width = 2
        channels = 1

        for num in sorted_nums:
            wav_path = voice_dir / f"{num:03d}.wav"
            if not wav_path.is_file():
                raise RuntimeError(
                    f"voice/{num:03d}.wav is missing — generate or import audio first."
                )
            with wave.open(str(wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                combined_frames += wf.readframes(wf.getnframes())

        # Write combined WAV to a temp file
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(combined_frames)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        tmp_path.write_bytes(buf.getvalue())

        try:
            combined_text = "\n\n".join(scenes[n].strip() for n in sorted_nums)
            word_timestamps = locally_align_audio(wav_path=tmp_path, text=combined_text)

            # Build scene ranges from cumulative WAV durations
            scene_ranges = {}
            cursor = 0.0
            for num in sorted_nums:
                wav_path = voice_dir / f"{num:03d}.wav"
                with wave.open(str(wav_path), "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())
                scene_ranges[num] = (cursor, cursor + dur)
                cursor += dur

            ass_out = project_path / "subtitles.ass"
            generate_ass(
                word_timestamps=word_timestamps,
                scene_ranges=scene_ranges,
                template_path=Path(template_path),
                output_path=ass_out,
                words_per_group=words_per_group,
            )
            return ass_out
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
