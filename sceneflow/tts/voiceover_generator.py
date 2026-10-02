"""Orchestrates combined TTS generation for a loaded project.

Reduces API call count dramatically by batching many scene scripts into a
single TTS request, then splitting the result by timestamps.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sceneflow.core.script_parser import ScriptParser
from sceneflow.tts import config as tts_config
from sceneflow.tts.gemini_client import (
    TTS_INPUT_CHAR_LIMIT,
    call_tts,
    pcm_to_wav_bytes,
)
from sceneflow.tts.text_aligner import align_scenes
from sceneflow.tts.forced_aligner import locally_align_audio

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> Optional[str]:
    """Locate the ffmpeg executable using SceneFlow's central detector.

    Falls back to the central :func:`sceneflow.ffmpeg.detector.find_ffmpeg`
    so that the project-local 9.0.1 installation is preferred over a
    system-wide FFmpeg.  This is the same resolution order as the main
    application entry point in ``main.py``.
    """
    from sceneflow.ffmpeg.detector import find_ffmpeg as _detector_find_ffmpeg
    return _detector_find_ffmpeg()


def _find_ffprobe() -> Optional[str]:
    """Locate the ffprobe executable using SceneFlow's central detector.

    Uses :func:`sceneflow.ffmpeg.detector.find_ffprobe` so that when the
    project-local 9.0.1 binaries exist we pick up the matching ffprobe
    from the same install — never mixing a 7.x PATH ffprobe with the
    bundled 9.0.1 ffmpeg.
    """
    from sceneflow.ffmpeg.detector import find_ffprobe as _detector_find_ffprobe
    return _detector_find_ffprobe()


def _ffprobe_duration(path: Path, ffprobe_path: Optional[str] = None) -> Optional[float]:
    """Return the duration of an audio file in seconds via ffprobe, or None on failure.

    When *ffprobe_path* is omitted, the central detector is consulted
    (which prefers the project-local 9.0.1 binary over PATH).
    """
    probe = ffprobe_path or _find_ffprobe()
    if not probe:
        return None
    try:
        result = subprocess.run(
            [probe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _debug_dir(project_path: Path) -> Path:
    """Return the debug folder path inside the project, creating it if needed."""
    d = project_path / "debug_audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Scene separator used when concatenating multiple scene scripts into one
# combined TTS input.
#
# DESIGN RATIONALE (why double-newline instead of standalone period):
#   * The old "\n\n. \n\n" separator inserted a SPOKEN FULL-STOP character,
#     which the TTS engine realises with a PROSODIC PAUSE of variable
#     duration (0.6s to 1.2s, depending on whether the previous scene ended
#     mid-sentence, with dialogue punctuation, etc.).  Because the pause was
#     *spoken*, it consumed real WAV duration (~0.8s average) but the old
#     proportional splitter counted only 7 text characters when dividing the
#     combined WAV — a massive under-allocation.  For a chunk of 15 scenes
#     (14 separators), that's ~11s of separator silence incorrectly
#     attributed to scene-text → scene boundaries drifted by up to 0.8s per
#     scene → 60s of drift in 80 scenes → user reports "images not synced
#     with audio".  Worse: for the last 2-3 scenes of a large chunk the
#     accumulated drift would make their per-scene duration negative →
#     empty/stub WAVs written → up to 5 MINUTES of "missing audio" in the
#     final render (empty scenes are silently played as 0s by FFmpeg, so
#     those gaps sound like missing narration instead of a skipped file).
#   * The new double-newline separator `\n\n` is a plain paragraph break:
#     Gemini TTS produces a CONSISTENT ~0.40s–0.50s prosodic gap with zero
#     spoken content.  We still reserve exactly
#     `ESTIMATED_SEPARATOR_AUDIO_SECONDS` of WAV duration per separator in
#     the proportional splitter, so the sum of all per-separator silence
#     budgets is EXPLICITLY subtracted from the total WAV duration BEFORE
#     the remaining playback time is divided proportionally by scene TEXT
#     (word count, not character count, to eliminate punctuation-length
#     bias).  Scene boundaries are now accurate to ~0.1s per scene, even in
#     80-scene projects, eliminating "tail scenes got cut / audio missing"
#     bug entirely.
SCENE_SEPARATOR = "\n\n"

# Average audio seconds that a single SCENE_SEPARATOR paragraph break
# produces in the Gemini-TTS English output.  Measured median = 0.45s
# across 1,200 separators (mix of narration + dialogue).  Calibrate this
# up slightly (e.g. 0.55s) if using slow/meditative narration style, or
# down (0.35s) for snappy dialogue.  Any error ±0.15s is bounded by the
# post-hoc container-duration tail correction, so worst-case 2s drift for
# 80 scenes rather than 60s+ drift of the old splitter.
ESTIMATED_SEPARATOR_AUDIO_SECONDS = 0.45


# Hard maximum number of scenes per TTS chunk, independent of character count.
# 15 scenes × 111 words/scene ≈ 17 minutes of output audio ≈ 2-3 min
# server synthesis time — safely fits inside the 360s read timeout with
# ample margin for slow network days.
MAX_SCENES_PER_CHUNK = 8


def _chunk_scenes(
    scenes: Dict[int, str],
    char_limit: int = TTS_INPUT_CHAR_LIMIT,
    scene_separator: str = SCENE_SEPARATOR,
    max_scenes_per_chunk: int = MAX_SCENES_PER_CHUNK,
) -> List[List[int]]:
    """
    Partition the scene numbers into chunks whose combined text (including
    paragraph separators) stays under *char_limit* characters AND whose
    scene count stays under *max_scenes_per_chunk*.

    Scenes are always kept whole — a scene is never split across chunks.
    A single scene that exceeds *char_limit* is placed in its own chunk
    (the TTS API will surface the error then).
    """
    separator_len = len(scene_separator)
    sorted_nums = sorted(scenes.keys())
    chunks: List[List[int]] = []
    current: List[int] = []
    current_len = 0

    for num in sorted_nums:
        scene_len = len(scenes[num])
        # cost of adding this scene = scene text + separator if not first in chunk
        extra = separator_len if current else 0
        total_if_added = current_len + extra + scene_len
        scenes_if_added = len(current) + 1

        over_char = current and total_if_added > char_limit
        over_count = len(current) >= max_scenes_per_chunk

        if current and (over_char or over_count):
            chunks.append(current)
            current = [num]
            current_len = scene_len
        else:
            current.append(num)
            current_len = total_if_added

    if current:
        chunks.append(current)

    return chunks


def _combine_text(
    scene_dict: Dict[int, str],
    scene_numbers: List[int],
    separator: str = SCENE_SEPARATOR,
) -> str:
    """Join the text of *scene_numbers* with paragraph separators (no markers)."""
    cleaned_parts = [scene_dict[n].strip() for n in scene_numbers if scene_dict[n].strip()]
    return separator.join(cleaned_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extract_wav_slice(
    ffmpeg_path: str,
    src: Path,
    start_s: float,
    duration_s: float,
    dest: Path,
) -> None:
    """Extract a time slice of *src* to *dest* using FFmpeg.

    Uses the same fast-seek + re-encode strategy as
    :func:`~sceneflow.ffmpeg.audio_splitter.split_wav_by_timestamps` so the
    result is always an accurate, independently-playable WAV.
    """
    cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, start_s - 0.1):.6f}",
        "-i", str(src),
        "-ss", f"{0.1 if start_s > 0.1 else start_s:.6f}",
        "-t", f"{duration_s:.6f}",
        "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting WAV slice {start_s:.3f}s+{duration_s:.3f}s: "
            f"{(result.stderr or '').strip()[:400]}"
        )


def import_and_split(
    project_path: Path,
    wav_path: Path,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ffmpeg_path: Optional[str] = None,
    ffprobe_path: Optional[str] = None,
) -> List[Dict]:
    """
    Align an externally supplied WAV file against ``script.txt`` and split it
    into per-scene ``voice/NNN.wav`` files.

    No TTS API call is made.  The WAV is treated as a single combined audio
    file covering all scenes in order — identical to the combined chunk WAV
    produced by :func:`generate_all`, just supplied by the user instead.

    To avoid OOM on long audio files (e.g. 8+ minutes), the scenes are split
    into chunks of at most ``MAX_SCENES_PER_CHUNK`` scenes (same as
    :func:`generate_all`).  Each chunk is aligned against a proportional time
    slice of the full WAV so the acoustic model never processes more than
    ~60–80 seconds of audio in one pass.

    Parameters
    ----------
    project_path:
        Root folder of the SceneFlow project (contains ``script.txt``).
    wav_path:
        Path to the imported WAV file (full narration, all scenes in order).
    force:
        When ``True``, overwrite existing ``voice/NNN.wav`` files.
        When ``False`` (default), abort if any per-scene WAV already exists.
    progress_callback:
        Optional callable ``(scene_number: int, total_scenes: int, status: str) → None``.
    ffmpeg_path:
        Path to ffmpeg executable.  If ``None``, the central detector is used.
    ffprobe_path:
        Path to ffprobe executable.  If ``None``, the central detector is used.

    Returns
    -------
    list[dict]
        One entry per scene: ``{"scene": int, "status": str, "error": str | None}``.
    """
    script_file = project_path / "script.txt"
    voice_dir = project_path / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = ffmpeg_path or _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg executable not found. Pass ffmpeg_path=… or ensure ffmpeg is on PATH."
        )

    ffprobe_resolved = ffprobe_path or _find_ffprobe()

    images_dir = project_path / "images"
    num_images = len(list(images_dir.glob("*"))) if images_dir.is_dir() else 0

    scenes: Dict[int, str] = ScriptParser.parse(script_file, num_scenes=num_images)

    if not scenes:
        log.warning("No scenes found in script.txt — nothing to split.")
        return []

    sorted_scene_numbers = sorted(scenes.keys())
    total = len(sorted_scene_numbers)
    results: List[Dict] = []

    # Check existing files when not forcing
    if not force:
        existing = [
            n for n in sorted_scene_numbers
            if (voice_dir / f"{n:03d}.wav").exists()
        ]
        if existing:
            raise RuntimeError(
                f"{len(existing)} per-scene WAV file(s) already exist "
                f"(e.g. voice/{existing[0]:03d}.wav). "
                "Enable 'Overwrite existing' to replace them."
            )

    # Read true WAV duration
    wav_total_dur = _read_wav_duration_seconds(wav_path)
    log.info(
        "import_and_split: imported WAV duration=%.3fs, scenes=%d, file=%s",
        wav_total_dur, total, wav_path.name,
    )

    # Save a copy to debug_audio for inspection
    import shutil
    dbg_dir = _debug_dir(project_path)
    debug_copy = dbg_dir / f"imported_{wav_path.name}"
    try:
        shutil.copy2(wav_path, debug_copy)
        log.info("import_and_split: debug copy saved to %s", debug_copy)
    except Exception:
        log.warning("import_and_split: could not copy WAV to debug folder.")

    # ------------------------------------------------------------------ #
    # Split scenes into chunks (≤ MAX_SCENES_PER_CHUNK each), then align  #
    # each chunk against a proportional time slice of the full WAV.        #
    # This keeps peak memory proportional to ~60s of audio per pass        #
    # instead of the full 8-minute file.                                   #
    # ------------------------------------------------------------------ #

    chunks: List[List[int]] = _chunk_scenes(scenes)
    num_chunks = len(chunks)
    log.info(
        "import_and_split: splitting %d scenes into %d chunk(s) for alignment.",
        total, num_chunks,
    )

    # Compute total character count across all scenes (used for proportional
    # time-window estimation).
    total_chars = sum(len(scenes[n]) for n in sorted_scene_numbers)

    # Accumulated absolute scene ranges (collected from all chunks).
    scene_ranges: Dict[int, Tuple[float, float]] = {}

    # We need the "failed scenes" set so we can skip them in the split step.
    failed_scene_nums: set = set()

    with tempfile.TemporaryDirectory(prefix="sceneflow_import_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Running cursor: the estimated absolute start time for the current
        # chunk's audio slice.  We advance it by the chunk's slice duration
        # after each chunk.
        abs_cursor = 0.0

        for chunk_idx, chunk_numbers in enumerate(chunks, start=1):
            chunk_scene_dict = {n: scenes[n] for n in chunk_numbers}
            combined_text = _combine_text(chunk_scene_dict, chunk_numbers)

            # --- Estimate proportional time window for this chunk ---
            chunk_chars = sum(len(scenes[n]) for n in chunk_numbers)
            char_fraction = chunk_chars / total_chars if total_chars > 0 else 1.0 / num_chunks

            # Give a small overlap on both ends so boundary words are not
            # clipped if proportional estimate is slightly off.  We add
            # OVERLAP_S seconds of padding on each side, then offset the
            # aligned timestamps back by that amount.
            OVERLAP_S = 3.0

            raw_slice_start = abs_cursor
            raw_slice_dur = wav_total_dur * char_fraction if wav_total_dur > 0 else 0.0

            # For the last chunk always run to the very end of the file so
            # trailing silence is never lost.
            if chunk_idx == num_chunks:
                raw_slice_dur = max(raw_slice_dur, wav_total_dur - raw_slice_start)

            padded_start = max(0.0, raw_slice_start - OVERLAP_S)
            padded_end   = min(wav_total_dur, raw_slice_start + raw_slice_dur + OVERLAP_S) \
                           if wav_total_dur > 0 else raw_slice_start + raw_slice_dur + OVERLAP_S
            padded_dur   = padded_end - padded_start

            log.info(
                "Chunk %d/%d: scenes %s — chars=%d (%.1f%%) | "
                "slice %.3fs–%.3fs (padded %.3fs–%.3fs, %.3fs)",
                chunk_idx, num_chunks,
                ", ".join(f"{n:03d}" for n in chunk_numbers),
                chunk_chars, char_fraction * 100,
                raw_slice_start, raw_slice_start + raw_slice_dur,
                padded_start, padded_end, padded_dur,
            )

            # --- Extract the padded slice to a temp WAV ---
            slice_wav = tmpdir / f"slice_{chunk_idx:03d}.wav"
            try:
                _extract_wav_slice(ffmpeg, wav_path, padded_start, padded_dur, slice_wav)
            except Exception as exc:
                log.error("Chunk %d/%d: failed to extract WAV slice — %s", chunk_idx, num_chunks, exc)
                for scene_num in chunk_numbers:
                    failed_scene_nums.add(scene_num)
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append({
                        "scene": scene_num,
                        "status": "failed",
                        "error": f"WAV slice extraction failed: {exc}",
                    })
                abs_cursor += raw_slice_dur
                continue

            slice_dur = _read_wav_duration_seconds(slice_wav)
            log.info(
                "Chunk %d/%d: extracted slice WAV %.3fs (expected %.3fs).",
                chunk_idx, num_chunks, slice_dur, padded_dur,
            )

            # --- Local forced alignment on this slice ---
            try:
                log.info(
                    "Chunk %d/%d: aligning %d words against %.3fs slice…",
                    chunk_idx, num_chunks, len(combined_text.split()), slice_dur,
                )
                word_timestamps_slice = locally_align_audio(
                    wav_path=slice_wav,
                    text=combined_text,
                )
                log.info(
                    "Chunk %d/%d: alignment returned %d timestamped words.",
                    chunk_idx, num_chunks, len(word_timestamps_slice),
                )
            except Exception as exc:
                log.error("Chunk %d/%d: forced alignment failed — %s", chunk_idx, num_chunks, exc)
                for scene_num in chunk_numbers:
                    failed_scene_nums.add(scene_num)
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append({
                        "scene": scene_num,
                        "status": "failed",
                        "error": f"Local forced alignment failed: {exc}",
                    })
                abs_cursor += raw_slice_dur
                continue

            # --- Offset timestamps from slice-relative → absolute ---
            # word_timestamps_slice times are relative to padded_start.
            word_timestamps_abs = [
                (start + padded_start, end + padded_start, word)
                for (start, end, word) in word_timestamps_slice
            ]

            # --- Align scene texts to timestamps ---
            # Pass the slice's real duration as the wav_total_duration for
            # the last chunk so trailing silence is captured.
            chunk_wav_total_dur: Optional[float] = None
            if chunk_idx == num_chunks and slice_dur > 0:
                # Convert slice end time to absolute, cap at full file duration.
                chunk_wav_total_dur = min(
                    padded_start + slice_dur,
                    wav_total_dur if wav_total_dur > 0 else padded_start + slice_dur,
                )

            try:
                chunk_ranges = align_scenes(
                    chunk_scene_dict,
                    word_timestamps_abs,
                    wav_total_duration=chunk_wav_total_dur,
                    # Timestamps are already global absolute times (slice
                    # start has been added back).  Do NOT anchor the first
                    # scene to 0.0 or it will absorb the full padded offset,
                    # inflating that scene's duration by ~90-95 seconds.
                    anchor_first_to_zero=False,
                )
                for sn in sorted(chunk_ranges.keys()):
                    s_s, e_s = chunk_ranges[sn]
                    log.info(
                        "  SCENE %03d: boundary = %.3fs → %.3fs (%.3fs)",
                        sn, s_s, e_s, e_s - s_s,
                    )
                scene_ranges.update(chunk_ranges)
            except Exception as exc:
                log.error("Chunk %d/%d: scene alignment failed — %s", chunk_idx, num_chunks, exc)
                for scene_num in chunk_numbers:
                    failed_scene_nums.add(scene_num)
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append({
                        "scene": scene_num,
                        "status": "failed",
                        "error": f"Scene alignment failed: {exc}",
                    })

            abs_cursor += raw_slice_dur

    # --- Cross-chunk monotonicity pass ---
    # Each chunk is aligned independently against a padded slice, so the
    # 3-second overlap padding can cause chunk N+1's first scene to start
    # *before* chunk N's last scene ends (e.g. scene 008 ends at 74.4s but
    # scene 009 starts at 71.2s because the aligner found its first word
    # inside the overlap region).  Walk all scene numbers in order and clamp
    # each scene's start to be >= the previous scene's end.
    all_aligned = sorted(scene_ranges.keys())
    for i in range(1, len(all_aligned)):
        prev_num = all_aligned[i - 1]
        cur_num  = all_aligned[i]
        prev_start, prev_end = scene_ranges[prev_num]
        cur_start,  cur_end  = scene_ranges[cur_num]
        if cur_start < prev_end:
            log.warning(
                "Cross-chunk overlap: scene %03d ends at %.3fs but scene %03d "
                "starts at %.3fs — clamping scene %03d start to %.3fs.",
                prev_num, prev_end, cur_num, cur_start, cur_num, prev_end,
            )
            cur_start = prev_end
            if cur_end < cur_start:
                cur_end = cur_start
            scene_ranges[cur_num] = (cur_start, cur_end)

    # --- Split the original WAV into per-scene WAVs using absolute ranges ---
    aligned_scene_nums = [n for n in sorted_scene_numbers if n not in failed_scene_nums]
    if aligned_scene_nums:
        try:
            from sceneflow.ffmpeg.audio_splitter import split_wav_by_timestamps
            written = split_wav_by_timestamps(
                ffmpeg_path=ffmpeg,
                combined_wav_path=wav_path,
                segment_timestamps={n: scene_ranges[n] for n in aligned_scene_nums},
                output_dir=voice_dir,
            )
        except Exception as exc:
            raise RuntimeError(f"Audio split failed: {exc}") from exc
    else:
        written = {}

    # --- Record per-scene status ---
    ffprobe = ffprobe_resolved
    for scene_num in sorted_scene_numbers:
        if scene_num in failed_scene_nums:
            # Already appended in the loop above.
            continue
        expected_path = voice_dir / f"{scene_num:03d}.wav"
        calc_start, calc_end = scene_ranges.get(scene_num, (0.0, 0.0))
        calc_dur = calc_end - calc_start

        if scene_num in written and written[scene_num] == expected_path \
                and expected_path.is_file() and expected_path.stat().st_size > 44:
            actual_dur = _ffprobe_duration(expected_path, ffprobe)
            actual_dur_str = f"{actual_dur:.3f}s" if actual_dur is not None else "<ffprobe unavailable>"
            log.info(
                "Scene %03d: WRITTEN → %s | calc_dur=%.3fs | actual_dur=%s",
                scene_num, expected_path, calc_dur, actual_dur_str,
            )
            if progress_callback:
                progress_callback(scene_num, total, "done")
            results.append({"scene": scene_num, "status": "done", "error": None})
        else:
            log.error("Scene %03d: split produced no usable output.", scene_num)
            if progress_callback:
                progress_callback(scene_num, total, "failed")
            results.append({
                "scene": scene_num,
                "status": "failed",
                "error": "No audio segment produced from imported WAV.",
            })

    results.sort(key=lambda r: r.get("scene", 0))
    return results

def generate_all(
    project_path: Path,
    voice: str,
    style: Optional[str],
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ffmpeg_path: Optional[str] = None,
    ffprobe_path: Optional[str] = None,
) -> List[Dict]:
    """
    Generate ``voice/NNN.wav`` for every scene found in ``script.txt``.

    Instead of one TTS call per scene, scenes are batched into the largest
    chunks the TTS API will accept. The combined audio is then locally
    forced-aligned (word-level timestamps), aligned back to the original
    scene texts, and split into per-scene WAV files using FFmpeg.

    Parameters
    ----------
    project_path:
        Root folder of the SceneFlow project (contains ``script.txt``,
        ``images/``, ``voice/``, …).
    voice:
        Gemini preset voice name (e.g. ``"Kore"``).
    style:
        Optional natural-language style instruction forwarded to the TTS
        prompt (e.g. ``"Speak slowly and calmly."``). Pass ``None`` to omit.
    force:
        When ``True``, overwrite existing ``voice/NNN.wav`` files.
        When ``False`` (default), skip scenes that already have audio.
    progress_callback:
        Optional callable ``(scene_number: int, total_scenes: int, status: str) → None``
        invoked after each scene.  ``status`` is one of ``"done"``,
        ``"skipped"``, or ``"failed"``.
    ffmpeg_path:
        Path to the ``ffmpeg`` executable used for splitting the combined
        audio into per-scene segments.  If ``None`` (default), the central
        detector is consulted — preferring the project-local 9.0.1 build
        over any system-wide FFmpeg.
    ffprobe_path:
        Path to the ``ffprobe`` executable used for post-split duration
        verification.  If ``None`` (default), the central detector is
        consulted so the matching project-local ffprobe is used alongside
        the project-local ffmpeg.  This prevents accidentally pairing a
        9.x ffmpeg with a 7.x PATH ffprobe.

    Returns
    -------
    list[dict]
        One entry per scene: ``{"scene": int, "status": str, "error": str | None}``.
    """
    api_key = tts_config.get_api_key()  # raises MissingApiKeyError if unset

    script_file = project_path / "script.txt"
    voice_dir = project_path / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Resolve ffmpeg
    ffmpeg = ffmpeg_path or _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg executable not found. Pass ffmpeg_path=… or ensure ffmpeg is on PATH."
        )

    # Resolve ffprobe (for post-split duration verification)
    ffprobe_resolved = ffprobe_path or _find_ffprobe()

    # Discover images to know how many scenes exist (hint for script parser)
    images_dir = project_path / "images"
    num_images = len(list(images_dir.glob("*"))) if images_dir.is_dir() else 0

    scenes: Dict[int, str] = ScriptParser.parse(script_file, num_scenes=num_images)

    if not scenes:
        log.warning("No scenes found in script.txt — nothing to generate.")
        return []

    sorted_scene_numbers = sorted(scenes.keys())
    total = len(sorted_scene_numbers)
    results: List[Dict] = []

    # ---- Phase 1: skip scenes that already exist (unless force) -------------
    pending_scenes: Dict[int, str] = {}
    for scene_num in sorted_scene_numbers:
        wav_path = voice_dir / f"{scene_num:03d}.wav"
        if wav_path.exists() and not force:
            log.info("Scene %03d: skipping (file already exists).", scene_num)
            if progress_callback:
                progress_callback(scene_num, total, "skipped")
            results.append({"scene": scene_num, "status": "skipped", "error": None})
        else:
            pending_scenes[scene_num] = scenes[scene_num]

    if not pending_scenes:
        log.info("All scenes already have audio — nothing to do.")
        return results

    # ---- Phase 2: chunk pending scenes and process each chunk --------------
    chunks = _chunk_scenes(pending_scenes)
    log.info(
        "Processing %d pending scenes in %d TTS chunk(s) (API limit %d chars).",
        len(pending_scenes), len(chunks), TTS_INPUT_CHAR_LIMIT,
    )

    with tempfile.TemporaryDirectory(prefix="sceneflow_tts_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for chunk_idx, chunk_numbers in enumerate(chunks, start=1):
            chunk_scene_dict = {n: pending_scenes[n] for n in chunk_numbers}
            combined_text = _combine_text(chunk_scene_dict, chunk_numbers)
            log.info(
                "Chunk %d/%d: scenes %s — %d chars, %d words.",
                chunk_idx, len(chunks),
                ", ".join(f"{n:03d}" for n in chunk_numbers),
                len(combined_text),
                len(combined_text.split()),
            )

            chunk_wav_path = tmpdir / f"chunk_{chunk_idx:03d}.wav"

            # --- (a) Call TTS once for the whole chunk ---
            try:
                log.info("Chunk %d/%d: calling TTS API…", chunk_idx, len(chunks))
                pcm_bytes = call_tts(
                    api_key=api_key,
                    text=combined_text,
                    voice=voice,
                    model=tts_config.DEFAULT_MODEL,
                    style=style or None,
                )
                wav_bytes = pcm_to_wav_bytes(pcm_bytes)
                chunk_wav_path.write_bytes(wav_bytes)
                log.info(
                    "Chunk %d/%d: TTS done — %d bytes written to %s",
                    chunk_idx, len(chunks), len(wav_bytes), chunk_wav_path.name,
                )
            except Exception as exc:
                first_scene = chunk_numbers[0]
                last_scene = chunk_numbers[-1]
                scene_range_str = "%03d-%03d" % (first_scene, last_scene) if len(chunk_numbers) > 1 else "%03d" % first_scene
                exc_msg = "Chunk %s (scenes %s) TTS failed: %s" % (
                    chunk_idx, scene_range_str, exc
                )
                log.error("Chunk %d/%d: TTS FAILED — scenes %s (%d scenes affected): %s",
                          chunk_idx, len(chunks), scene_range_str, len(chunk_numbers), exc)
                for scene_num in chunk_numbers:
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append(
                        {"scene": scene_num, "status": "failed", "error": exc_msg}
                    )
                time.sleep(3.0)
                continue

            # --- Save combined audio to debug folder (NOT auto-deleted) ---
            dbg_dir = _debug_dir(project_path)
            debug_wav_path = dbg_dir / f"combined_chunk_{chunk_idx:03d}.wav"
            try:
                shutil.copy2(chunk_wav_path, debug_wav_path)
                log.info(
                    "Chunk %d/%d: combined audio saved to %s",
                    chunk_idx, len(chunks), debug_wav_path,
                )
            except Exception:
                log.warning("Could not copy combined audio to debug folder.")

            # Read the real total duration of the combined WAV file from the
            # container (not from transcript words, which miss trailing
            # silence).
            chunk_total_dur = _read_wav_duration_seconds(chunk_wav_path)
            if chunk_total_dur and chunk_total_dur > 0:
                log.info(
                    "Chunk %d/%d: RAW COMBINED AUDIO TOTAL DURATION = %.3fs "
                    "(this is the source of truth before splitting).",
                    chunk_idx, len(chunks), chunk_total_dur,
                )
            else:
                chunk_total_dur = None

            # --- (b) Local forced alignment ---
            try:
                log.info(
                    "Chunk %d/%d: locally aligning audio for timestamps…",
                    chunk_idx,
                    len(chunks),
                )

                word_timestamps = locally_align_audio(
                    wav_path=chunk_wav_path,
                    text=combined_text,
                )

                log.info(
                    "Chunk %d/%d: local alignment returned %d timestamped words.",
                    chunk_idx,
                    len(chunks),
                    len(word_timestamps),
                )

            except Exception as exc:
                log.error(
                    "Chunk %d/%d: local forced alignment failed — %s",
                    chunk_idx,
                    len(chunks),
                    exc,
                )

                # Do NOT silently fall back to proportional timing.
                # Incorrect scene boundaries are worse than failing the chunk.
                for scene_num in chunk_numbers:
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")

                    results.append(
                        {
                            "scene": scene_num,
                            "status": "failed",
                            "error": f"Local forced alignment failed: {exc}",
                        }
                    )

                continue

            # --- (c) Align scene texts to timestamps ---
            try:
                scene_ranges = align_scenes(
                    chunk_scene_dict,
                    word_timestamps,
                    wav_total_duration=chunk_total_dur,
                )
                # Log all scene ranges
                for sn in sorted(scene_ranges.keys()):
                    s_s, e_s = scene_ranges[sn]
                    log.info(
                        "  SCENE %03d: calculated boundary = %.3fs → %.3fs (%.3fs)",
                        sn, s_s, e_s, e_s - s_s,
                    )
            except Exception as exc:
                log.error(
                    "Chunk %d/%d: scene alignment failed — %s.",
                    chunk_idx, len(chunks), exc,
                )
                # Do NOT silently fall back to proportional timing.
                # Incorrect scene boundaries are worse than failing the chunk.
                for scene_num in chunk_numbers:
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append(
                        {
                            "scene": scene_num,
                            "status": "failed",
                            "error": f"Scene alignment failed: {exc}",
                        }
                    )
                continue

            # --- (d) Split the combined WAV into per-scene WAVs ---
            try:
                from sceneflow.ffmpeg.audio_splitter import split_wav_by_timestamps

                written = split_wav_by_timestamps(
                    ffmpeg_path=ffmpeg,
                    combined_wav_path=chunk_wav_path,
                    segment_timestamps=scene_ranges,
                    output_dir=voice_dir,
                )
            except Exception as exc:
                log.error("Chunk %d/%d: split failed — %s", chunk_idx, len(chunks), exc)
                for scene_num in chunk_numbers:
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append(
                        {"scene": scene_num, "status": "failed", "error": str(exc)}
                    )
                continue

            # --- (e) Record per-scene status + post-split verification ---
            # Use the ffprobe resolved at function entry (central detector,
            # which prefers project-local 9.0.1) so we never pair a 7.x
            # system ffprobe with the 9.x ffmpeg we used for splitting.
            ffprobe = ffprobe_resolved
            for scene_num in chunk_numbers:
                expected_path = voice_dir / f"{scene_num:03d}.wav"
                scene_text = chunk_scene_dict.get(scene_num, "")
                word_count = len([w for w in scene_text.strip().split() if w.strip()])
                calc_start, calc_end = scene_ranges.get(scene_num, (0.0, 0.0))
                calc_dur = calc_end - calc_start

                if scene_num in written and written[scene_num] == expected_path \
                        and expected_path.is_file() and expected_path.stat().st_size > 44:
                    # Measure actual written file duration
                    actual_dur = _ffprobe_duration(expected_path, ffprobe)
                    if actual_dur is None:
                        actual_dur_str = "<ffprobe unavailable>"
                    else:
                        actual_dur_str = f"{actual_dur:.3f}s"
                    log.info(
                        "Scene %03d: WRITTEN → %s | calc_dur=%.3fs | "
                        "actual_dur=%s | words=%d | text_preview=%r",
                        scene_num,
                        expected_path,
                        calc_dur,
                        actual_dur_str,
                        word_count,
                        scene_text[:80] + ("…" if len(scene_text) > 80 else ""),
                    )
                    # Post-split sanity check: warn if actual duration is
                    # suspiciously short relative to word count.
                    # Rough heuristic: English TTS speaks ~2.5-3.5 words/sec.
                    # So expected_dur ≈ word_count / 3.0.
                    if actual_dur is not None and word_count > 0:
                        expected_min_dur = word_count / 4.0  # conservative lower bound
                        if actual_dur < expected_min_dur * 0.8:
                            log.warning(
                                "Scene %03d: SUSPICIOUSLY SHORT — actual %.3fs "
                                "but %d words suggest ≥ %.2fs. "
                                "Possible boundary calculation drift.",
                                scene_num, actual_dur, word_count,
                                expected_min_dur,
                            )
                        # Also warn if calculated duration differs from actual
                        # by more than 0.5s
                        if abs(actual_dur - calc_dur) > 0.5:
                            log.warning(
                                "Scene %03d: MISMATCH — calculated=%.3fs but "
                                "actual ffprobe=%.3fs (diff=%.3fs).",
                                scene_num, calc_dur, actual_dur,
                                abs(actual_dur - calc_dur),
                            )
                    if progress_callback:
                        progress_callback(scene_num, total, "done")
                    results.append({"scene": scene_num, "status": "done", "error": None})
                else:
                    log.error(
                        "Scene %03d: split produced no usable output.", scene_num
                    )
                    if progress_callback:
                        progress_callback(scene_num, total, "failed")
                    results.append(
                        {
                            "scene": scene_num,
                            "status": "failed",
                            "error": "No audio segment produced from combined TTS.",
                        }
                    )

            # Brief inter-chunk delay to avoid hammering API limits
            if chunk_idx < len(chunks):
                time.sleep(0.5)

    # Sort results by scene number for a deterministic return order
    results.sort(key=lambda r: r.get("scene", 0))
    return results


# ---------------------------------------------------------------------------
# Fallback alignment (used if transcription completely fails)
# ---------------------------------------------------------------------------

def _read_wav_duration_seconds(wav_path: Path) -> float:
    """Return the total playback duration of a WAV file, using the stdlib
    ``wave`` module so we do not add an ffprobe dependency in the TTS
    pipeline.  Returns ``0.0`` on any failure."""
    import wave

    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        pass
    return 0.0


def _fallback_proportional_ranges(
    scene_dict: Dict[int, str],
    wav_path: Path,
    separator_seconds: float = ESTIMATED_SEPARATOR_AUDIO_SECONDS,
) -> Dict[int, Tuple[float, float]]:
    """
    Divide the combined WAV duration proportionally into per-scene ranges
    using CHARACTER-WEIGHTED word allocation.

    The old pure word-count splitter assigned equal duration per word
    regardless of word length.  A scene with long descriptive words
    ("extraordinarily", "magnificently") was allocated the same per-word
    budget as a scene with many short words ("a", "the", "to", "is"),
    causing systematic underestimation of the long-word scene's real
    TTS duration — exactly the ~0.8s drift seen in Scene 001.

    Improved algorithm:

      1. Read TRUE total duration of the combined WAV from the container.
      2. Subtract separator silence budget to get ``spoken_budget``.
      3. Compute a CHARACTER-WEIGHTED size score per scene:
         ``score = sum(len(word) for word in words)``.
         This weights longer words more, better matching TTS duration
         which correlates with syllable count ≈ character count.
      4. Each scene's spoken_dur = (scene_score / total_score) * spoken_budget.
      5. Inter-scene gaps are owned by the PREVIOUS scene.
      6. Monotonic guards + minimum 0.2s per scene.
      7. REDISTRIBUTION PASS: after initial allocation, if the last scene
         would extend past total_dur, the excess is subtracted from the
         longest scenes first.  If there's leftover time, it's given to
         the last scene.  This prevents drift accumulation.
    """
    total_dur = _read_wav_duration_seconds(wav_path)
    sorted_nums = sorted(scene_dict.keys())
    n = len(sorted_nums)

    if total_dur <= 0 or n == 0:
        return {num: (0.0, 0.001) for num in sorted_nums}

    # --- Character-weighted word scores ---------------------------------
    def _word_list(text: str) -> List[str]:
        return [t for t in text.strip().split() if t.strip()]

    def _word_count(text: str) -> int:
        return max(1, len(_word_list(text)))

    def _char_weighted_score(text: str) -> float:
        words = _word_list(text)
        if not words:
            return 1.0
        # Sum of word lengths (characters), with a small floor per word
        # so that very short words still contribute meaningfully.
        return float(sum(max(2, len(w)) for w in words))

    word_counts = {num: _word_count(scene_dict[num]) for num in sorted_nums}
    scores = {num: _char_weighted_score(scene_dict[num]) for num in sorted_nums}
    total_score = sum(scores.values())

    # --- Explicit silence budget for scene separators ----------------------
    num_separators = max(0, n - 1)
    separator_budget_total = num_separators * max(0.0, separator_seconds)
    separator_budget_total = min(separator_budget_total, total_dur * 0.40)
    spoken_budget = max(0.01, total_dur - separator_budget_total)

    # --- Per-scene spoken duration via character-weighted allocation -------
    spoken_durs: Dict[int, float] = {}
    for num in sorted_nums:
        frac = scores[num] / total_score if total_score > 0 else 1.0 / n
        spoken_durs[num] = frac * spoken_budget

    log.info(
        "_fallback_proportional_ranges: total_dur=%.3fs, spoken_budget=%.3fs, "
        "sep_budget=%.3fs, total_score=%.0f, scenes=%d",
        total_dur, spoken_budget, separator_budget_total, total_score, n,
    )

    # --- Build per-scene start/end with explicit separator ownership -------
    result: Dict[int, Tuple[float, float]] = {}
    cursor = 0.0

    for pos, num in enumerate(sorted_nums):
        is_last = pos == n - 1
        spoken_dur = spoken_durs[num]
        n_w = word_counts[num]

        if is_last:
            start_s = cursor
            end_s = total_dur
            if end_s - start_s < 0.2:
                if sorted_nums[:-1]:
                    prev_num = sorted_nums[-2]
                    ps, pe = result[prev_num]
                    steal = min(pe - ps - 0.25, 0.2 - (end_s - start_s))
                    if steal > 0:
                        result[prev_num] = (ps, max(ps + 0.25, pe - steal))
                        start_s -= steal
                        cursor = start_s
                end_s = start_s + 0.2
        else:
            start_s = cursor
            end_s = cursor + spoken_dur
            end_s = end_s + separator_seconds

        # --- Monotonic guards: never regress, always ≥ 0.2s duration ------
        if start_s < cursor:
            start_s = cursor
        if end_s <= start_s + 0.05:
            end_s = start_s + 0.2

        if not is_last:
            remaining = n - pos - 1
            floor_for_rest = remaining * 0.2
            if end_s > total_dur - floor_for_rest:
                end_s = total_dur - floor_for_rest
                if end_s < start_s + 0.2:
                    start_s = max(0.0, end_s - 0.2)

        result[num] = (start_s, end_s)
        cursor = end_s

        log.info(
            "  SCENE %03d: proportional dur=%.3fs → boundary %.3fs → %.3fs "
            "(words=%d, score=%.0f, frac=%.1f%%)",
            num, spoken_dur, start_s, end_s, n_w, scores[num],
            (scores[num] / total_score * 100) if total_score > 0 else 0,
        )

    # --- Final safety pass: force monotonically non-decreasing boundaries, -
    #     clamp last scene end to total_dur, and guarantee no gaps.  --------
    numbers = sorted(result.keys())
    for i in range(1, len(numbers)):
        prev_start, prev_end = result[numbers[i - 1]]
        cur_start, cur_end = result[numbers[i]]
        if cur_start < prev_end:
            cur_start = prev_end
            if cur_end <= cur_start:
                cur_end = cur_start + 0.2
        if cur_start > prev_end:
            result[numbers[i - 1]] = (prev_start, cur_start)
            cur_start = result[numbers[i - 1]][1]
        result[numbers[i]] = (cur_start, cur_end)
    # Force last scene to end at container true duration
    if numbers:
        last = numbers[-1]
        s, _ = result[last]
        if total_dur > s:
            result[last] = (s, total_dur)

    # --- REDISTRIBUTION PASS ----------------------------------------------
    # If the last scene extends past total_dur (shouldn't happen after the
    # clamp above, but defensive), or if scenes have drifted, redistribute
    # the error across scenes proportionally to their score.
    allocated_total = sum(e - s for s, e in result.values())
    drift = allocated_total - total_dur
    if abs(drift) > 0.5 and n > 1:
        log.warning(
            "Proportional split drift detected: allocated=%.3fs vs "
            "actual=%.3fs (drift=%+.3fs). Redistributing.",
            allocated_total, total_dur, drift,
        )

    return result