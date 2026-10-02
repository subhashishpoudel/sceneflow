"""Align timestamped transcription words to original per-scene scripts."""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Punctuation that gets stripped entirely (intentionally EXCLUDES ASCII
# apostrophe "'" and hyphen "-" so contractions like "don't" and compound
# words like "mother-in-law" survive tokenization).
_PUNCT_RE = re.compile(
    r"[\u2000-\u206F\u2E00-\u2E7F\\!\"#\$%&\(\)\*\+,\./:;<=>\?@\[\]\^_`\{\|\}\~]"
)
_WHITESPACE_RE = re.compile(r"\s+")
_APOSTROPHE_VARIANTS = str.maketrans({
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201B": "'",  # single high-reversed-9 quotation mark
    "\u02BC": "'",  # modifier letter apostrophe
    "\u055A": "'",  # armenian apostrophe
    "\uFF07": "'",  # fullwidth apostrophe
})
_HYPHEN_VARIANTS = str.maketrans({
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
})


def _normalize_word(token: str) -> str:
    """
    Aggressively normalise a single token for matching:
      - strip accents / unicode compose
      - unify apostrophe / hyphen variants to ASCII
      - lowercase
      - strip leading and trailing punctuation characters
      - collapse remaining internal whitespace to a single space each
    Internal apostrophes and hyphens are preserved for contractions like
    ``don't`` and compounds like ``mother-in-law``.
    """
    if not token:
        return ""
    s = unicodedata.normalize("NFKD", token)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(_APOSTROPHE_VARIANTS)
    s = s.translate(_HYPHEN_VARIANTS)
    s = s.lower()
    s = s.strip()
    # Strip leading / trailing punctuation-like chars individually, preserving
    # any internal apostrophes / hyphens.
    s = s.strip("!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~ \t\n\r-'\u2000-\u206F\u2E00-\u2E7F")
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _tokenize(text: str) -> List[str]:
    """Split *text* into normalized word tokens."""
    if not text:
        return []
    tokens = re.split(r"\s+", text.strip())
    normed = [_normalize_word(t) for t in tokens]
    return [t for t in normed if t]


# ---------------------------------------------------------------------------
# Scene-boundary detection
# ---------------------------------------------------------------------------

def _similar(a: str, b: str) -> float:
    """Return similarity ratio in [0,1] between two (normalized) words."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Fast length-based filter
    if abs(len(a) - len(b)) >= max(3, max(len(a), len(b)) // 2 + 1):
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _find_subsequence_match(
    transcript_words: List[str],
    scene_tokens: List[str],
    start_index: int,
    min_similarity: float = 0.75,
) -> Tuple[int, int]:
    """
    Search *transcript_words* (beginning at *start_index*) for a contiguous
    subsequence that best matches *scene_tokens*, tolerant of minor
    transcription differences.

    Returns
    -------
    tuple[int, int]
        ``(transcript_start_idx, transcript_end_idx_exclusive)`` index range
        into ``transcript_words``.  ``(start_index, start_index)`` if no
        reasonable match is found.
    """
    n_scene = len(scene_tokens)
    n_trans = len(transcript_words)

    if n_scene == 0:
        return start_index, start_index
    if start_index >= n_trans:
        return n_trans, n_trans

    # If there are very few words, look ahead by a wide window to allow
    # for a few missing/extra tokens at the boundaries.
    slop = max(3, n_scene // 3)

    best_score = -1.0
    best_start = start_index
    best_end = start_index + n_scene

    # We try every possible start position in a reasonable window, and for
    # each we walk through the scene tokens (consuming transcript tokens
    # greedily, allowing skips on either side) and compute a score.
    search_start = start_index
    search_end_start = min(n_trans - 1, start_index + slop)

    for t_start in range(search_start, search_end_start + 1):
        t_idx = t_start
        score = 0.0
        matched = 0
        # Track the index of the LAST SUCCESSFULLY MATCHED transcript word.
        # This is distinct from t_idx (the walk pointer) which also advances
        # on failed matches.  Using last_matched_t+1 as the exclusive end
        # prevents the boundary from being placed at the walk position when
        # trailing scene tokens fail to match.
        last_matched_t = -1
        for s_idx in range(n_scene):
            s_tok = scene_tokens[s_idx]
            # Look ahead a few transcript words for a decent match
            best_word_sim = 0.0
            best_word_t = t_idx
            look_ahead = min(4, n_trans - t_idx)
            for j in range(look_ahead):
                sim = _similar(s_tok, transcript_words[t_idx + j])
                if sim > best_word_sim:
                    best_word_sim = sim
                    best_word_t = t_idx + j
                if sim >= 0.99:  # perfect match, stop looking
                    break
            if best_word_sim >= min_similarity:
                score += best_word_sim
                matched += 1
                last_matched_t = best_word_t
                t_idx = best_word_t + 1
            else:
                # Scene word not clearly found in transcript: advance through
                # transcript anyway but don't count it as a match.
                t_idx = t_idx + 1

            if t_idx >= n_trans:
                # Diagnostic: log why the walk stopped early.
                if s_idx < n_scene - 1:
                    log.debug(
                        "_find_subsequence_match: walk ran out of transcript "
                        "words at scene_token[%d/%d]=%r (t_idx=%d, n_trans=%d, "
                        "last_matched_t=%d, matched=%d) — remaining %d scene "
                        "tokens unmatched.",
                        s_idx, n_scene - 1, s_tok, t_idx, n_trans,
                        last_matched_t, matched, n_scene - 1 - s_idx,
                    )
                break

        if matched == 0:
            continue

        # Normalise score by the scene length and add a small bonus for
        # tighter alignment (fewer skipped transcript words).
        coverage = matched / n_scene
        avg_sim = score / matched if matched > 0 else 0.0
        final_score = 0.6 * avg_sim + 0.4 * coverage

        if final_score > best_score:
            best_score = final_score
            best_start = t_start
            # best_end is exclusive: one past the last matched transcript word.
            # If nothing matched (shouldn't happen here since matched>0),
            # fall back to the walk position.
            best_end = (last_matched_t + 1) if last_matched_t >= 0 else t_idx

            log.debug(
                "_find_subsequence_match: candidate start=%d → end_excl=%d "
                "(last_matched_t=%d, matched=%d/%d, score=%.3f, coverage=%.2f)",
                best_start, best_end, last_matched_t, matched, n_scene,
                final_score, coverage,
            )

    if best_score < 0.35:
        # Very low confidence — fall back to a proportional-size chunk.
        log.warning(
            "Low-confidence alignment for scene (%d tokens, score=%.2f); "
            "falling back to proportional split.", n_scene, best_score,
        )
        best_start = start_index
        # Proportional guess based on fraction of remaining transcript
        remaining_scenes_tokens_est = n_scene
        # Compute a simple ratio: we don't know total remaining scene tokens,
        # so just take an amount proportional to the scene length.
        best_end = min(n_trans, start_index + max(1, n_scene + slop))
        return best_start, best_end

    log.debug(
        "_find_subsequence_match: BEST start=%d end_excl=%d score=%.3f "
        "(n_scene=%d, n_trans=%d, search_start=%d)",
        best_start, best_end, best_score, n_scene, n_trans, search_start,
    )

    return best_start, best_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

WordTimestamps = List[Tuple[float, float, str]]  # [(start, end, word)]
SceneRange = Tuple[float, float]                 # (start_seconds, end_seconds)


def align_scenes(
    scene_dict: Dict[int, str],
    transcript_timestamps: WordTimestamps,
    wav_total_duration: Optional[float] = None,
    anchor_first_to_zero: bool = True,
) -> Dict[int, SceneRange]:
    """
    Given per-scene script texts and the word-level timestamped transcript
    of the combined audio, determine the start/end timestamp for each scene.

    Scene boundaries work as follows (so images do not end early during
    natural pauses between scenes):

    * Scene ``i`` **starts** at the timestamp of its first matched word.
    * For every scene except the LAST one, scene ``i`` **ends** at the
      START timestamp of the NEXT scene's first matched word.  This means
      the inter-scene silence / breathing gap belongs to the previous
      scene's visual window, so the image stays up during the pause and
      does not appear to "end early".
    * The FINAL scene ends at the greater of (a) its last matched word's
      end-time, or (b) *wav_total_duration* when provided, so any
      trailing silence or reverb tail after the last spoken word is also
      attributed to the last scene.

    Parameters
    ----------
    scene_dict:
        Mapping ``{scene_number: script_text}`` for the scenes that make up
        the combined audio, in ascending numerical order.
    transcript_timestamps:
        List of ``(start, end, word)`` tuples from the transcription step.
    wav_total_duration:
        Optional real total duration (in seconds) of the combined WAV
        file as measured from the container, *not* from the last transcribed
        word.  Used to capture trailing silence at the end of the final
        scene.
    anchor_first_to_zero:
        When ``True`` (default), the first scene's start is clamped to
        ``0.0`` so that any lead-in silence before the first spoken word
        is attributed to the first scene rather than being a gap.  Set to
        ``False`` when *transcript_timestamps* already contain global
        absolute times (e.g. when aligning a chunk whose timestamps have
        been offset by a padded slice start) — otherwise the first scene's
        start would be incorrectly reset to ``0.0`` instead of its real
        absolute position, inflating that scene's duration by the full
        padded offset.

    Returns
    -------
    dict[int, tuple[float, float]]
        Mapping from scene number to ``(audio_start_seconds,
        audio_end_seconds)`` relative to the beginning of the combined audio.
    """
    sorted_numbers = sorted(scene_dict.keys())
    if not sorted_numbers:
        return {}

    # Normalize transcript words (index -> normalized word string)
    trans_words = [_normalize_word(w) for (_, _, w) in transcript_timestamps]
    n_trans = len(trans_words)

    # Build scene token lists
    scene_tokens: Dict[int, List[str]] = {
        num: _tokenize(scene_dict[num]) for num in sorted_numbers
    }

    # ------------------------------------------------------------------
    # Phase 1 — find each scene's word index range in the transcript
    # ------------------------------------------------------------------
    # word_start_idx[num] = transcript index of this scene's first spoken word
    # word_end_last_idx[num] = inclusive transcript index of the LAST spoken word
    #                         that is part of this scene (used to derive the
    #                         last-word boundary used below only for the last
    #                         scene and as a fallback floor for non-last scenes)
    word_start_idx: Dict[int, int] = {}
    word_end_last_idx: Dict[int, int] = {}

    cursor = 0  # transcript word index
    for idx, num in enumerate(sorted_numbers):
        tokens = scene_tokens[num]
        expected_len = len(tokens)
        if not tokens or cursor >= n_trans:
            # Empty scene or we've run out of transcript; anchor to cursor
            if n_trans == 0:
                anchor_idx = 0
            elif cursor >= n_trans:
                anchor_idx = n_trans - 1
            else:
                anchor_idx = cursor
            word_start_idx[num] = anchor_idx
            word_end_last_idx[num] = anchor_idx
            cursor = max(cursor + 1, anchor_idx + 1)
            log.info(
                "SCENE %03d: anchored at transcript[%d] (empty scene or "
                "no transcript left). expected_text_len=%d tokens",
                num, anchor_idx, expected_len,
            )
            continue

        t_start, t_end_excl = _find_subsequence_match(trans_words, tokens, cursor)

        # Clamp to valid range
        t_start = max(0, min(t_start, n_trans - 1 if n_trans > 0 else 0))
        t_end_excl = max(t_start, min(t_end_excl, n_trans))

        word_start_idx[num] = t_start
        # Inclusive index of the last word that belongs to this scene
        word_end_last_idx[num] = max(t_start, t_end_excl - 1) if n_trans > 0 else t_start

        cursor = max(cursor + 1, t_end_excl)

        # --- Diagnostic logging for boundary calculation ---
        first_word = (transcript_timestamps[t_start][2]
                      if 0 <= t_start < n_trans else "<N/A>")
        last_idx_logged = t_end_excl - 1
        last_word = (transcript_timestamps[last_idx_logged][2]
                     if 0 <= last_idx_logged < n_trans else "<N/A>")
        matched_text = " ".join(
            transcript_timestamps[i][2]
            for i in range(t_start, t_end_excl)
            if 0 <= i < n_trans
        )
        stop_word = (transcript_timestamps[t_end_excl][2]
                     if 0 <= t_end_excl < n_trans else "<end of transcript>")
        log.info(
            "SCENE %03d: transcript[%d:%d] %.3f → %.3f | first=%r | last=%r",
            num,
            t_start,
            t_end_excl,
            transcript_timestamps[t_start][0] if 0 <= t_start < n_trans else 0.0,
            transcript_timestamps[last_idx_logged][1] if 0 <= last_idx_logged < n_trans else 0.0,
            first_word,
            last_word,
        )
        log.info(
            "SCENE %03d: expected_text_len=%d tokens | matched_transcript_len=%d words "
            "| matched_text=%r | match_stops_at[%d]=%r | cursor→%d",
            num,
            expected_len,
            t_end_excl - t_start,
            matched_text,
            t_end_excl,
            stop_word,
            cursor,
        )

    # ------------------------------------------------------------------
    # Phase 2 — convert word indices → seconds with correct gap ownership
    # ------------------------------------------------------------------
    # How many seconds after the last spoken word to place the cut point.
    # The gap between scenes is natural silence produced by the TTS pause;
    # cutting 1 second after the last word lands cleanly inside that gap
    # rather than right at the next scene's first word, giving a slightly
    # more natural breath between scenes.  The cut is always capped at the
    # next scene's first word so it never bleeds into the next scene.
    POST_WORD_OFFSET_S = 0.5

    result: Dict[int, SceneRange] = {}

    for pos, num in enumerate(sorted_numbers):
        start_s = transcript_timestamps[word_start_idx[num]][0] if n_trans > 0 else 0.0

        last_idx = word_end_last_idx[num]
        last_word_end_s = (
            transcript_timestamps[last_idx][1]
            if 0 <= last_idx < n_trans
            else start_s
        )

        if pos < len(sorted_numbers) - 1:
            # Non-last scene: END = last_word_end + 1s offset, capped at
            # the next scene's first word so we never bleed into it.
            next_num = sorted_numbers[pos + 1]
            next_start_idx = word_start_idx[next_num]
            if 0 <= next_start_idx < n_trans:
                next_start_s = transcript_timestamps[next_start_idx][0]
                preferred_end_s = last_word_end_s + POST_WORD_OFFSET_S
                end_s = min(preferred_end_s, next_start_s)
                boundary_reason = (
                    f"last_word_end={last_word_end_s:.3f}s +{POST_WORD_OFFSET_S}s offset"
                    f" → {preferred_end_s:.3f}s, capped at "
                    f"next_scene({next_num})_first_word={next_start_s:.3f}s"
                    f" → end={end_s:.3f}s"
                )
            else:
                end_s = last_word_end_s + POST_WORD_OFFSET_S
                boundary_reason = (
                    f"next_scene({next_num})_first_word index {next_start_idx} "
                    f"out of range — using last_word_end={last_word_end_s:.3f}s"
                )
        else:
            # Last scene: include any trailing silence from the actual WAV
            # (beyond the last spoken word — the transcription only reports
            # spoken words, not trailing silence).
            end_s = last_word_end_s
            if wav_total_duration is not None and wav_total_duration > end_s:
                end_s = wav_total_duration
                boundary_reason = (
                    f"last_scene: wav_total_duration={wav_total_duration:.3f}s "
                    f"(last_word_end={last_word_end_s:.3f}s)"
                )
            elif n_trans > 0 and transcript_timestamps[-1][1] > end_s:
                end_s = transcript_timestamps[-1][1]
                boundary_reason = (
                    f"last_scene: transcript_last_word_end="
                    f"{transcript_timestamps[-1][1]:.3f}s "
                    f"(own_last_word_end={last_word_end_s:.3f}s)"
                )
            else:
                boundary_reason = (
                    f"last_scene: last_word_end={last_word_end_s:.3f}s "
                    f"(no wav_total_duration or trailing silence)"
                )

        if end_s < start_s:
            end_s = start_s
            boundary_reason += " [CLAMPED: end_s < start_s]"

        log.info(
            "SCENE %03d: boundary %.3f → %.3f (%.3fs) | reason: %s",
            num, start_s, end_s, end_s - start_s, boundary_reason,
        )

        result[num] = (start_s, end_s)

    # ------------------------------------------------------------------
    # Phase 3 — safety pass: guarantee monotonically non-decreasing
    # contiguous boundaries.  With the Phase 2 rule (scene N ends at
    # scene N+1's start) this should already be true; this pass just
    # clamps any pathological cases (e.g. alignment produced an overlap
    # because the next scene's first word matched before the current
    # scene's last word ended).
    # ------------------------------------------------------------------
    numbers = sorted(result.keys())
    for i in range(1, len(numbers)):
        prev_start, prev_end = result[numbers[i - 1]]
        cur_start, cur_end = result[numbers[i]]

        # If alignment somehow produced an overlap, nudge the current
        # scene's start forward to match previous scene's end.  We never
        # shorten a previous scene's end, because that would drop the
        # silence-gap ownership and reintroduce the "image ends early"
        # bug.
        if cur_start < prev_end:
            cur_start = prev_end
        # If there is still any positive gap (shouldn't happen, Phase 2
        # already ensures prev_end == next_start), expand previous scene's
        # end to absorb it rather than leaving an owned silence hole.
        elif cur_start > prev_end + 1e-3:
            prev_end = cur_start
            result[numbers[i - 1]] = (prev_start, prev_end)

        if cur_end < cur_start:
            cur_end = cur_start
        result[numbers[i]] = (cur_start, cur_end)

    # First scene must start at the very beginning of the audio to avoid
    # an un-owned lead-in gap if transcript's first word starts slightly
    # after 0.0.  Skip this when anchor_first_to_zero=False because the
    # timestamps are global absolute times (e.g. a padded slice offset),
    # and forcing start=0.0 would falsely expand the first scene's duration
    # by the entire padded-slice start offset.
    if anchor_first_to_zero and numbers:
        s, e = result[numbers[0]]
        if s > 0.0:
            result[numbers[0]] = (0.0, e)

    return result
