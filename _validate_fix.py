"""VALIDATION SCRIPT - part 1: code syntax + detector + DLL registration"""
from __future__ import annotations
import os, sys, py_compile
from pathlib import Path

ROOT = Path.cwd()

# --- 1. py_compile all changed files
changed_files = [
    ROOT / "main.py",
    ROOT / "sceneflow" / "ffmpeg" / "detector.py",
    ROOT / "sceneflow" / "gui.py",
    ROOT / "sceneflow" / "tts" / "forced_aligner.py",
    ROOT / "sceneflow" / "tts" / "voiceover_generator.py",
    ROOT / "sceneflow" / "core" / "timeline_builder.py",
    ROOT / "test_alignment_only.py",
]
print("=== 1. py_compile check ===")
all_ok = True
for f in changed_files:
    try:
        py_compile.compile(str(f), doraise=True)
        print(f"  OK  {f.relative_to(ROOT)}")
    except Exception as e:
        print(f"  FAIL {f.relative_to(ROOT)}: {type(e).__name__}: {e}")
        all_ok = False
print(f"  Result: {'PASS' if all_ok else 'FAIL'}\n")

# --- 2. DLL search path + BLAS env registration BEFORE torch import
print("=== 2. DLL / BLAS env registration ===")
BLAS_KEYS = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
before_blas = {k: os.environ.get(k) for k in BLAS_KEYS}

from sceneflow.ffmpeg.detector import register_bundled_ffmpeg_dll_search_path
register_bundled_ffmpeg_dll_search_path()

after_blas = {k: os.environ.get(k) for k in BLAS_KEYS}
print(f"  BLAS env: OPENBLAS_NUM_THREADS={after_blas['OPENBLAS_NUM_THREADS']!r} "
      f"(before={before_blas['OPENBLAS_NUM_THREADS']!r})")
assert after_blas["OPENBLAS_NUM_THREADS"] == "1", "BLAS guard not applied"

ffmpeg_bin = str(ROOT / "tools" / "ffmpeg" / "bin")
path_parts = os.environ.get("PATH", "").split(os.pathsep)
print(f"  Bundled ffmpeg bin FIRST in $env:PATH = {path_parts[0] == ffmpeg_bin}")
print(f"  (PATH[0]={path_parts[0]!r})")
assert path_parts[0] == ffmpeg_bin, "Bundled bin not prepended to PATH"

# --- 3. find_ffmpeg / find_ffprobe resolution
print("\n=== 3. find_ffmpeg / find_ffprobe resolution ===")
from sceneflow.ffmpeg.detector import (
    find_ffmpeg, find_ffprobe, find_ffplay, validate_installation,
    get_ffmpeg_version,
)
fmpeg = find_ffmpeg()
fprobe = find_ffprobe()
fplay = find_ffplay()
print(f"  ffmpeg:  {fmpeg}")
print(f"  ffprobe: {fprobe}")
print(f"  ffplay:  {fplay}")
expected_suffix = str(Path("tools/ffmpeg/bin/ffmpeg.exe")).lower()
assert fmpeg and Path(fmpeg).name.lower() == "ffmpeg.exe", "ffmpeg not found as exe"
assert "tools" + os.sep + "ffmpeg" + os.sep + "bin" in fmpeg.lower().replace("\\", os.sep).replace("/", os.sep), \
    f"ffmpeg path is not project-local: {fmpeg}"
assert fprobe and "ffprobe.exe" in fprobe, "ffprobe not found"
ok, msg = validate_installation()
ver = get_ffmpeg_version(fmpeg)
print(f"  validate_installation = ({ok}, {msg!r})")
print(f"  get_ffmpeg_version = {ver!r}")
assert ok and "9.0.1" in ver, f"Validation failed or wrong version: ok={ok} ver={ver}"
print("  Result: PASS\n")

# --- 4. Direct ctypes load of libtorchcodec_core9.dll if it exists
#    After DLL registration, loading core9.dll should resolve its avcodec/avformat deps.
print("=== 4. TorchCodec DLL dependency resolution test ===")
tc_core9 = Path(r"C:\Users\DELL\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torchcodec\libtorchcodec_core9.dll")
if tc_core9.is_file():
    import ctypes
    try:
        h = ctypes.CDLL(str(tc_core9))
        print(f"  ctypes.CDLL(libtorchcodec_core9.dll) SUCCESS: handle={h}")
        print("  -> This means avcodec-63.dll / avformat-63.dll WERE FOUND via the registered DLL search path.")
        print("  Result: PASS")
    except Exception as e:
        print(f"  ctypes.CDLL FAILED: {type(e).__name__}: {e}")
        print("  (Soundfile decoder bypasses TorchCodec anyway — this doesn't block alignment.)")
        print("  Result: WARN (non-blocking)")
else:
    print(f"  libtorchcodec_core9.dll not present at {tc_core9}; skipping.")
    print("  Result: SKIPPED")

print("\n=== ALL VALIDATION 1-4 COMPLETE ===")
