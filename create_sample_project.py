"""
create_sample_project.py — helper script that builds a minimal test project
so you can verify SceneFlow works before you have real assets.

Usage:
    python create_sample_project.py [output_folder]

Requires: Pillow (for generating test images) and any installed audio tool.
If Pillow is not installed it falls back to creating tiny placeholder files.

The generated project works with SceneFlow Video Assembler if you also have
short real WAV files named 001.wav, 002.wav, 003.wav to drop in voice/.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import wave
from pathlib import Path


OUTPUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample_project")


# ─── helpers ───────────────────────────────────────────────────────────────

def make_wav(path: Path, duration: float = 3.0, freq: float = 440.0) -> None:
    """Generate a simple sine-wave WAV file (mono, 44100 Hz)."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(sample_rate)
        data = bytearray()
        for i in range(n_samples):
            val = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            data += struct.pack("<h", val)
        wf.writeframes(bytes(data))


def make_image_pillow(path: Path, scene_num: int, color: tuple) -> None:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1920, 1080), color=color)
    draw = ImageDraw.Draw(img)
    text = f"Scene {scene_num:03d}"
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((1920 - tw) // 2, (1080 - th) // 2), text, fill="white", font=font)
    img.save(str(path))


def make_image_fallback(path: Path) -> None:
    """Write a tiny valid 1x1 PNG (no Pillow needed)."""
    # Minimal valid PNG bytes for a 1×1 red pixel
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"                          # signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90"
        b"wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00"
        b"\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png_bytes)


# ─── main ──────────────────────────────────────────────────────────────────

SCENES = [
    {"num": 1, "duration": 4.0, "tone": 440.0,  "color": (30, 60, 120)},
    {"num": 2, "duration": 6.0, "tone": 550.0,  "color": (120, 30, 60)},
    {"num": 3, "duration": 3.5, "tone": 330.0,  "color": (30, 120, 60)},
]

SCRIPT = """\
[SCENE 001]
The hero stands alone on the mountain,
gazing toward the distant horizon.

[SCENE 002]
A wind sweeps through the valley below,
carrying whispers of change.

[SCENE 003]
Everything changes in an instant —
and nothing will ever be the same.
"""

if __name__ == "__main__":
    images_dir = OUTPUT_DIR / "images"
    voice_dir  = OUTPUT_DIR / "voice"
    images_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)

    try:
        import PIL  # noqa: F401
        has_pillow = True
    except ImportError:
        has_pillow = False
        print("Pillow not installed — creating placeholder images instead.")
        print("  pip install Pillow   for real coloured test images.\n")

    for scene in SCENES:
        num  = scene["num"]
        img_path   = images_dir / f"{num:03d}.png"
        audio_path = voice_dir  / f"{num:03d}.wav"

        if has_pillow:
            make_image_pillow(img_path, num, scene["color"])
            print(f"  Created image: {img_path}")
        else:
            make_image_fallback(img_path)
            print(f"  Created placeholder image: {img_path}")

        make_wav(audio_path, duration=scene["duration"], freq=scene["tone"])
        print(f"  Created audio: {audio_path}  ({scene['duration']}s)")

    (OUTPUT_DIR / "script.txt").write_text(SCRIPT, encoding="utf-8")
    print(f"\n✓ Sample project created at: {OUTPUT_DIR.resolve()}")
    print("\nRun SceneFlow and open this folder to test rendering.")
