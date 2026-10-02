"""Local, offline forced alignment for SceneFlow's TTS pipeline.

This module produces word-level timestamps for a WAV file whose spoken
content is *already known* (it is the exact text that was sent to Gemini
TTS). This is classic **forced alignment**: given audio + its transcript,
find where each word starts and ends in the audio. It is not automatic
speech recognition (we never guess what was said) and it is not
proportional/estimated timing (we never divide duration by word or
character count).

Engine
------
We use torchaudio's built-in CTC forced-alignment pipeline
(``torchaudio.pipelines.MMS_FA``), which wraps a wav2vec2-style acoustic
model trained for Meta's "Massively Multilingual Speech" forced-alignment
work. This is the approach documented in torchaudio's official "CTC Forced
Alignment API" / "Forced alignment for multilingual data" tutorials:

    1. Run the known text (as a list of words) and the audio through the
       acoustic model to get a per-frame emission (character probabilities).
    2. Use CTC forced alignment (Viterbi-style) to find the best
       frame-alignment between the KNOWN text and the emission.
    3. Convert the resulting per-word frame spans into seconds.

This is genuine forced alignment against the known script, not a
transcript that we hope matches — the model is never allowed to choose
different words than the ones we give it.

Why this engine
----------------
- Pure PyTorch/torchaudio for the alignment model, plus ``soundfile`` for
  WAV reading (its wheel bundles libsndfile, so no separate FFmpeg
  system install is required on Windows): pip-installable, no conda, no
  external services.
- Ships as a ready-to-use pretrained pipeline (no training or fine-tuning
  needed) and is officially maintained as part of torchaudio.
- Runs reliably on CPU. It does not require CUDA, so it works out of the
  box on a Windows machine with an AMD GPU — the CPU build of
  torch/torchaudio is used and no GPU vendor lock-in is involved.
- Handles long, multi-scene combined audio in a single pass, which matches
  SceneFlow's "one TTS call + one alignment call per chunk" design.

What this module deliberately does NOT do
------------------------------------------
- No calls to Gemini, Google, OpenAI, or any other cloud/network service.
- No API keys.
- No character-count / word-count / equal-time-per-word estimation.
- No silent fallback to proportional timing if alignment fails or looks
  unreliable — this module raises ``ForcedAlignmentError`` instead, so a
  bad alignment always surfaces as a clearly failed chunk.
"""
from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class ForcedAlignmentError(RuntimeError):
    """Raised whenever local forced alignment cannot produce a reliable,
    validated set of word timestamps. Callers must treat this as a hard
    failure for the affected chunk — never substitute estimated timing."""


# ---------------------------------------------------------------------------
# Optional heavy dependencies (imported lazily so simply importing this
# module never requires torch/torchaudio to be installed).
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

try:
    import torchaudio  # type: ignore
except ImportError:
    torchaudio = None  # type: ignore


_PIP_INSTALL_HINT = (
    "Install the CPU build with (PowerShell):\n"
    "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
    "This is a CPU-only build and works on Windows with an AMD GPU — no CUDA "
    "is required or used."
)

# Lazily-initialized, process-wide singletons. Loaded once on first call to
# locally_align_audio() and reused for every subsequent chunk, so we do NOT
# reload the (fairly large) acoustic model for every one of ~15 scenes.
_model = None
_tokenizer = None
_aligner = None
_bundle_sample_rate: Optional[int] = None
_device = None


def _require_torch_stack() -> None:
    """Load torch/torchaudio and the MMS_FA forced-alignment pipeline the
    first time it's needed, raising a clear, actionable error otherwise."""
    global _model, _tokenizer, _aligner, _bundle_sample_rate, _device

    if _model is not None:
        return

    # Make sure the bundled FFmpeg DLL directory (tools/ffmpeg/bin) is on
    # the Windows native DLL search path BEFORE we import torchaudio.  This
    # is normally done by main.py / gui.py entry points, but defensive here
    # for anyone calling locally_align_audio() directly from standalone
    # scripts (e.g. test_alignment_only.py).  Without this, torchaudio's
    # TorchCodec backend may fail to load libavcodec/libavformat even when
    # those DLLs are bundled in the repo.  The call is idempotent so the
    # extra call when entering through main() is harmless.
    try:
        from sceneflow.ffmpeg.detector import register_bundled_ffmpeg_dll_search_path
        register_bundled_ffmpeg_dll_search_path()
    except Exception:  # noqa: BLE001 - never let DLL setup block torch stack init
        pass

    if torch is None:
        raise ForcedAlignmentError(
            "Local forced alignment requires PyTorch, which is not installed.\n"
            + _PIP_INSTALL_HINT
        )

    if torchaudio is None:
        raise ForcedAlignmentError(
            "Local forced alignment requires torchaudio, which is not "
            "installed (it ships alongside torch).\n" + _PIP_INSTALL_HINT
        )

    try:
        from torchaudio.pipelines import MMS_FA as bundle
    except ImportError as exc:
        raise ForcedAlignmentError(
            "Local forced alignment requires a torchaudio version that "
            "includes the MMS_FA forced-alignment pipeline "
            "(torchaudio >= 2.1).\n"
            "Upgrade with (PowerShell):\n"
            "  pip install --upgrade torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "forced_aligner: selected device=%s (torch.cuda.is_available()=%s). "
        "CPU is the supported baseline; CUDA is only used opportunistically "
        "when actually present.",
        device, torch.cuda.is_available(),
    )

    log.info(
        "forced_aligner: loading MMS_FA acoustic model (first run downloads "
        "it once and caches it locally under the torch hub cache directory; "
        "later runs are fully offline)…"
    )
    load_start = time.monotonic()
    try:
        model = bundle.get_model(with_star=False)
        model = model.to(device)
        model.eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, actionable error
        raise ForcedAlignmentError(
            "Failed to load the local MMS_FA forced-alignment model: "
            f"{exc}\n"
            "If this is the very first run, make sure this machine has an "
            "internet connection so the model can be downloaded once. Once "
            "cached locally, alignment works fully offline."
        ) from exc

    _model, _tokenizer, _aligner = model, tokenizer, aligner
    _bundle_sample_rate = bundle.sample_rate
    _device = device
    log.info(
        "forced_aligner: model ready in %.1fs (sample_rate=%dHz, device=%s).",
        time.monotonic() - load_start, _bundle_sample_rate, _device,
    )


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

# Matches a "word" while preserving contractions ("don't") and hyphenated
# compounds ("mother-in-law") as single units. Anything else (standalone
# punctuation, em dashes, quotation marks, ellipses, etc.) is treated as a
# separator and simply does not become a word.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

# Characters the MMS_FA English acoustic model can spell out. Everything
# else is stripped when building the *model-facing* version of a word (the
# word returned to the caller always keeps its original, unstripped form).
_MODEL_CHAR_RE = re.compile(r"[a-z']")


def _extract_original_words(text: str) -> List[str]:
    """Split *text* into the words that will be aligned, preserving each
    word's original punctuation-internal characters (apostrophes, hyphens)
    exactly as written."""
    # Normalize curly quotes to straight quotes so "don't" and "don't" are
    # treated identically regardless of which apostrophe character the
    # script used.
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return _WORD_RE.findall(normalized)


def _model_word(original_word: str) -> str:
    """Reduce *original_word* to the alphabet the acoustic model understands
    (lowercase letters + apostrophe). Hyphens and any other characters are
    dropped for alignment purposes only — the original word text (with its
    hyphen intact) is still what gets returned to the caller."""
    lowered = original_word.lower().replace("\u2019", "'").replace("\u2018", "'")
    # Strip accents/diacritics to their base letters where possible.
    decomposed = unicodedata.normalize("NFKD", lowered)
    decomposed = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(_MODEL_CHAR_RE.findall(decomposed))


# ---------------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------------

def _load_and_resample_audio(wav_path: Path, target_sample_rate: int):
    """Load *wav_path* (read-only) and return a mono waveform tensor
    resampled to *target_sample_rate*. Does not write/modify the source
    file in any way.

    Deliberately does NOT use torchaudio.load(): recent torchaudio versions
    route decoding through TorchCodec, which requires a separately
    installed system FFmpeg (shared-library build) to be on PATH. That's an
    extra, fragile system-level install just to read a plain WAV file we
    generated ourselves. `soundfile` bundles its native decoder
    (libsndfile) directly inside its pip wheel on Windows/macOS/Linux, so
    no separate system install is needed.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ForcedAlignmentError(
            "Local forced alignment requires the 'soundfile' package to "
            "read WAV files.\n"
            "Install it with (PowerShell):\n"
            "  pip install soundfile"
        ) from exc

    log.info(
        "forced_aligner: reading WAV with soundfile decoder (libsndfile) — "
        "bypassing torchaudio.load() / TorchCodec to avoid native DLL issues. "
        "file=%s",
        wav_path.name,
    )
    try:
        # always_2d ensures a consistent (num_samples, num_channels) shape
        # even for mono files.
        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, actionable error
        raise ForcedAlignmentError(
            f"Failed to read WAV file for local alignment: {wav_path} ({exc})"
        ) from exc

    num_samples, num_channels = samples.shape
    duration_s = num_samples / float(sample_rate) if sample_rate > 0 else 0.0
    log.info(
        "forced_aligner: soundfile read OK — duration=%.3fs, "
        "sample_rate=%dHz, channels=%d, samples=%d.",
        duration_s, sample_rate, num_channels, num_samples,
    )

    # (num_samples, num_channels) -> (num_channels, num_samples), matching
    # the (channel, time) convention the rest of this module expects.
    waveform = torch.from_numpy(samples.T).contiguous()

    if waveform.size(0) > 1:
        log.info(
            "forced_aligner: input WAV has %d channels — downmixing to mono.",
            waveform.size(0),
        )
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        log.info(
            "forced_aligner: resampling audio from %dHz to %dHz for alignment.",
            sample_rate, target_sample_rate,
        )
        # Pure tensor-math resampling — does not touch TorchCodec/FFmpeg.
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    return waveform


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def _run_ctc_alignment(waveform, model_words: List[str]):
    """Run the acoustic model + CTC forced alignment. Returns
    (emission, token_spans) where token_spans has one entry per word in
    *model_words*, each a (possibly empty) list of aligned token spans."""
    with torch.inference_mode():
        emission, _ = _model(waveform.to(_device))
        token_spans = _aligner(emission[0], _tokenizer(model_words))
    return emission, token_spans


def _spans_to_seconds(
    token_spans,
    num_samples: int,
    num_frames: int,
    sample_rate: int,
) -> List[Optional[Tuple[float, float]]]:
    """Convert per-word CTC frame spans into (start_seconds, end_seconds)."""
    if num_frames <= 0:
        return [None for _ in token_spans]

    samples_per_frame = num_samples / num_frames
    results: List[Optional[Tuple[float, float]]] = []
    for spans in token_spans:
        if not spans:
            results.append(None)
            continue
        start_frame = spans[0].start
        end_frame = spans[-1].end
        start_s = (start_frame * samples_per_frame) / sample_rate
        end_s = (end_frame * samples_per_frame) / sample_rate
        results.append((start_s, end_s))
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_timestamps(
    timestamps: List[Tuple[float, float, str]],
    wav_duration: float,
    num_input_words: int,
) -> None:
    """Raise ForcedAlignmentError unless *timestamps* look trustworthy.
    Never "fixes up" bad data — a failure here must fail the chunk."""
    if not timestamps:
        raise ForcedAlignmentError(
            "Local forced alignment produced zero aligned words — refusing "
            "to return an empty/fabricated result."
        )

    coverage = len(timestamps) / num_input_words if num_input_words else 0.0
    if coverage < 0.9:
        raise ForcedAlignmentError(
            f"Local forced alignment only aligned {len(timestamps)}/"
            f"{num_input_words} words ({coverage:.0%} coverage), which is "
            "too unreliable to trust. This usually means the WAV audio "
            "does not actually match the supplied text."
        )
    if coverage < 1.0:
        log.warning(
            "forced_aligner: only %d/%d words were aligned (%.0f%% "
            "coverage) — proceeding, but this chunk may be worth "
            "re-checking manually.",
            len(timestamps), num_input_words, coverage * 100,
        )

    prev_end = 0.0
    for idx, (start, end, word) in enumerate(timestamps):
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ForcedAlignmentError(
                f"Non-finite timestamp for word #{idx} ({word!r}): "
                f"start={start}, end={end}."
            )
        if start < 0:
            raise ForcedAlignmentError(
                f"Negative start timestamp for word #{idx} ({word!r}): {start}."
            )
        if end <= start:
            raise ForcedAlignmentError(
                f"Non-positive duration for word #{idx} ({word!r}): "
                f"start={start:.3f}s, end={end:.3f}s."
            )
        if start < prev_end - 0.01:
            raise ForcedAlignmentError(
                f"Timestamps are not monotonically ordered at word #{idx} "
                f"({word!r}): start={start:.3f}s precedes previous "
                f"end={prev_end:.3f}s."
            )
        if end > wav_duration + 0.25:
            raise ForcedAlignmentError(
                f"Timestamp for word #{idx} ({word!r}) ends at {end:.3f}s, "
                f"which is beyond the WAV duration of {wav_duration:.3f}s."
            )
        prev_end = end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locally_align_audio(
    wav_path: Path,
    text: str,
) -> List[Tuple[float, float, str]]:
    """
    Compute word-level timestamps for the audio in *wav_path* against the
    already-known *text*, entirely locally (no network calls, no API keys).

    Parameters
    ----------
    wav_path:
        Path to the combined-chunk WAV file produced by the Gemini TTS call.
        Read-only — this file is never modified.
    text:
        The exact combined text that was sent to TTS for this chunk (i.e.
        SceneFlow's ``combined_text``).

    Returns
    -------
    list[tuple[float, float, str]]
        ``(start_seconds, end_seconds, word)`` for each aligned word, in
        chronological order, using the original word text (contractions and
        hyphenated compounds preserved).

    Raises
    ------
    ForcedAlignmentError
        If required packages/models are missing, if the model fails to run,
        or if the resulting timestamps do not pass validation. Callers must
        treat this as a hard failure — never substitute proportional or
        estimated timing.
    """
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise ForcedAlignmentError(f"WAV file not found for local alignment: {wav_path}")

    _require_torch_stack()

    original_words = _extract_original_words(text)
    if not original_words:
        raise ForcedAlignmentError(
            "No alignable words were found in the supplied text — nothing to align against."
        )

    model_words = [_model_word(w) for w in original_words]
    empty_indices = [i for i, w in enumerate(model_words) if not w]
    if empty_indices:
        offenders = ", ".join(repr(original_words[i]) for i in empty_indices[:5])
        more = "" if len(empty_indices) <= 5 else f" (+{len(empty_indices) - 5} more)"
        raise ForcedAlignmentError(
            "Local forced alignment only supports alphabetic words (plus "
            "apostrophes for contractions). These words contain no "
            f"alignable letters after normalization: {offenders}{more}. "
            "Spell out numbers/symbols as words before sending the script to TTS."
        )

    log.info(
        "forced_aligner: aligning %s against %d input words…",
        wav_path.name, len(original_words),
    )

    waveform = _load_and_resample_audio(wav_path, _bundle_sample_rate)
    num_samples = waveform.size(-1)
    wav_duration = num_samples / _bundle_sample_rate
    log.info(
        "forced_aligner: WAV duration=%.3fs at %dHz.",
        wav_duration, _bundle_sample_rate,
    )

    align_start = time.monotonic()
    try:
        emission, token_spans = _run_ctc_alignment(waveform, model_words)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, actionable error
        raise ForcedAlignmentError(
            f"Local forced alignment failed while running the acoustic model: {exc}"
        ) from exc
    alignment_secs = time.monotonic() - align_start

    num_frames = emission.size(1)
    span_seconds = _spans_to_seconds(token_spans, num_samples, num_frames, _bundle_sample_rate)

    timestamps: List[Tuple[float, float, str]] = []
    for word, span in zip(original_words, span_seconds):
        if span is None:
            continue
        start_s, end_s = span
        timestamps.append((float(start_s), float(end_s), word))

    log.info(
        "forced_aligner: aligned %d/%d words in %.2fs (model=MMS_FA, device=%s).",
        len(timestamps), len(original_words), alignment_secs, _device,
    )

    _validate_timestamps(timestamps, wav_duration, len(original_words))

    return timestamps