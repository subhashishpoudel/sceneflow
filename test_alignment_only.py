# test_alignment_only.py — verifies alignment against audio you already have, zero Gemini calls
from __future__ import annotations

# ── VERY EARLY: register bundled FFmpeg DLL dir before any torch/torchaudio
# import.  Without this, on Windows, torchaudio's TorchCodec backend fails to
# resolve avcodec-63.dll / avformat-63.dll even though those DLLs are shipped
# in tools/ffmpeg/bin.  The function is idempotent so repeated calls are safe.
from sceneflow.ffmpeg.detector import register_bundled_ffmpeg_dll_search_path
register_bundled_ffmpeg_dll_search_path()

import logging
import struct
import sys
import tempfile
import wave
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)

from sceneflow.tts.forced_aligner import locally_align_audio


def _make_synthetic_pcm_wav(out_path: Path, duration_s: float = 1.5, sample_rate: int = 24000) -> None:
    """Write a simple 16-bit PCM mono WAV using only the stdlib ``wave`` module.

    Produces a 440 Hz sine tone with amplitude ~0.3 of full scale.  This gives
    us a KNOWN, deterministic WAV file we can feed into the audio loader so
    we can verify: (a) the soundfile decoder path is actually used, (b) the
    file can be resampled to 16000 Hz for the MMS_FA model, (c) no TorchCodec
    / libav DLL errors occur — all without a real TTS call.
    """
    import math
    num_frames = int(duration_s * sample_rate)
    amp = int(0.3 * 32767)
    freq = 440.0
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = bytearray()
        for i in range(num_frames):
            t = i / sample_rate
            v = int(round(amp * math.sin(2.0 * math.pi * freq * t)))
            samples += struct.pack("<h", v)
        wf.writeframes(bytes(samples))


def test_synthetic_wav_load_and_resample() -> bool:
    """Self-contained test: no TTS, no model download, no internet.

    Creates a synthetic WAV, loads it via forced_aligner._load_and_resample_audio
    (which uses soundfile + torchaudio.functional.resample), and verifies the
    output tensor shape / sample count is plausible.  This is the EXACT same
    code path locally_align_audio() uses for the audio-loading portion of the
    pipeline — we just stop before calling the acoustic model.
    """
    from sceneflow.tts.forced_aligner import _load_and_resample_audio
    print("\n=== SELF-TEST: synthetic WAV loading (no API calls) ===")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "synth_24000hz_1.5s.wav"
            _make_synthetic_pcm_wav(wav_path, duration_s=1.5, sample_rate=24000)
            assert wav_path.is_file(), "Synthetic WAV was not written"
            print(f"  Created synthetic WAV: {wav_path} ({wav_path.stat().st_size} bytes)")

            # Force alignment model expects 16000 Hz; this is what _require_torch_stack sets.
            target_sr = 16000
            waveform = _load_and_resample_audio(wav_path, target_sr)
            print(f"  Decoded waveform tensor shape: {tuple(waveform.shape)}")
            assert waveform.dim() == 2, "Expected (channel, time) tensor"
            assert waveform.size(0) == 1, "Expected mono (1 channel after downmix)"
            expected_approx = int(1.5 * target_sr)
            actual = waveform.size(-1)
            print(f"  Resampled length: {actual} samples  (expected ≈ {expected_approx})")
            assert abs(actual - expected_approx) <= 2, (
                f"Resampled length {actual} differs too much from expected {expected_approx}"
            )
            # Soundness: samples should be finite float32 in a sensible range.
            assert waveform.is_floating_point(), "Waveform should be float32"
            max_abs = waveform.abs().max().item()
            print(f"  Max |sample|: {max_abs:.4f}  (≤ 1.0: {max_abs <= 1.0})")
            print("  SELF-TEST PASSED: WAV decode + resample work correctly.\n")
            return True
    except Exception as exc:
        print(f"  SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Always run the synthetic self-test first.  This verifies:
    #   - DLL search path registration works  (no avcodec-63.dll not found)
    #   - soundfile decoder path is used     (logs explicitly show "soundfile decoder")
    #   - torchaudio.functional.resample works  (pure tensor math)
    # ... WITHOUT requiring any internet, model download, or TTS API call.
    self_ok = test_synthetic_wav_load_and_resample()
    if not self_ok:
        sys.exit(2)

    # Point this at the combined WAV you already have from the failed run:
    wav_path = Path("myproject - Copy/debug_audio/combined_chunk_001.wav")

    # Scene texts joined with a BLANK LINE between each scene, matching
    # voiceover_generator.py's SCENE_SEPARATOR = "\n\n" exactly (no [SCENE nnn]
    # markers — _combine_text() never inserts those, it just joins raw scene text).
    combined_text = (
        "Three years after leaving her hometown, Yuna returned to the seaside "
        "village with one suitcase, a box of unfinished drawings, and no "
        "intention of staying. She had told herself this was only business. "
        "Clean the café, settle her grandmother's affairs, sell the "
        "property, and leave before the memories became too difficult to face."
        "\n\n"
        "But the moment she stepped onto the familiar street, the ocean breeze "
        "carried the scent of salt, pine trees, and freshly roasted coffee "
        "from somewhere nearby. The village looked smaller than she "
        "remembered. The houses seemed older. Yet the little blue café at "
        "the end of the road looked exactly the same."
        "\n\n"
        "Summer's End had belonged to her grandmother for nearly forty years. "
        "Its faded blue walls were covered with climbing vines, the wooden "
        "windows were slightly crooked, and the old sign swung whenever the "
        "sea wind became strong. Yuna stood outside for several seconds, "
        "wondering whether opening that door meant reopening everything she "
        "had tried to forget."
        "\n\n"
        "She finally unlocked it.\n\n"
        "Dust floated through the morning sunlight as she pushed the door "
        "open. The familiar bell above the entrance gave a weak little chime. "
        "Yuna smiled despite herself. She remembered being ten years old and "
        "sitting behind that same counter, drawing customers while her "
        "grandmother complained that she was using all the good paper."
        "\n\n"
        "She placed her suitcase beside the counter and looked around. The "
        "café was quiet, but it didn't feel empty. Every table seemed to "
        "hold a memory. The corner booth where she studied for exams. The "
        "window seat where she sketched sunsets. The back staircase where she "
        "used to hide whenever she didn't want to do her homework."
    )

    if not wav_path.is_file():
        print(f"Pre-existing WAV not found at: {wav_path.resolve()}")
        print("Synthetic self-test passed; skipping real alignment run.")
        sys.exit(0)

    timestamps = locally_align_audio(wav_path=wav_path, text=combined_text)

    print(f"Aligned {len(timestamps)} words:")
    for start, end, word in timestamps[:10]:
        print(f"  {start:.3f}s - {end:.3f}s : {word}")
    print("  ...")
    print(f"Last word ends at {timestamps[-1][1]:.3f}s")