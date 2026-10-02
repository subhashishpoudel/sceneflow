"""FFmpeg/FFprobe installation detector."""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Well-known install locations on Windows
_WIN_SEARCH_PATHS = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
]

# Idempotent flag: setup_bundled_ffmpeg_dll_search_path() only acts once.
_DLL_SEARCH_PATH_SETUP_DONE = False
_ADDED_DLL_DIRECTORIES = []  # type: list[object]  # handles from os.add_dll_directory


def _apply_blas_thread_safety_env() -> None:
    """Apply known-safe BLAS/OpenBLAS thread-count environment defaults.

    Some numpy+openblas builds on high-core-count Windows machines raise
    ``OpenBLAS error: Memory allocation still failed after 10 retries``
    because OpenBLAS tries to spawn one thread per logical core *and*
    pre-allocate per-thread stacks.  Pinning thread count to a small,
    conservative value before numpy imports avoids the issue entirely.
    The environment variables only take effect if they are set *before*
    ``import numpy``, so we set them very early in the startup path.
    """
    blas_env_vars = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for key in blas_env_vars:
        if key not in os.environ:
            os.environ[key] = "1"


def register_bundled_ffmpeg_dll_search_path() -> None:
    """Expose the bundled ``tools/ffmpeg/bin`` directory to the Windows
    native DLL search order and prepend it to ``$env:PATH``.

    **Why this exists**

    ``torchaudio`` ≥ 2.11 routes many decoding calls through *TorchCodec*,
    whose native ``libtorchcodec_coreN.dll`` links against the FFmpeg
    shared libraries (``avcodec-63.dll``, ``avformat-63.dll``,
    ``avutil-61.dll``, ``swresample-7.dll``, …).  SceneFlow ships those
    exact DLLs inside ``tools/ffmpeg/bin`` alongside ``ffmpeg.exe``, but
    Windows does **not** search that directory for native DLLs by default.
    The result is a misleading ``FileNotFoundError`` on
    ``libtorchcodec_core9.dll`` — the torchcodec DLL itself exists, the
    problem is that *its* FFmpeg dependencies cannot be resolved.

    **What this function does**

    1. Sets known-safe BLAS/OpenBLAS thread env vars (must happen before
       numpy/torch import; idempotent).
    2. If ``<project_root>/tools/ffmpeg/bin`` exists:

       * Prepends it to ``os.environ["PATH"]`` so:

         - Child subprocesses (``ffmpeg``, ``ffprobe``) inherit it even
           when called without an explicit path.
         - Legacy ``LoadLibrary`` calls find the bundled DLLs.

       * Calls ``os.add_dll_directory(str(bin_dir))`` on Python ≥ 3.8 /
         Windows, which is the official Win32 way to extend the *Default
         DLL Search Order* used by ``ctypes.CDLL``.  This is what actually
         makes ``ctypes.CDLL("libtorchcodec_core9.dll")`` resolve
         ``avcodec-63.dll`` at native-load time.

    3. Logs exactly which directory was registered (or notes it was
       absent) at DEBUG level.

    The function is **idempotent**: calling it more than once per process
    is harmless — only the first invocation has any effect.
    """
    global _DLL_SEARCH_PATH_SETUP_DONE, _ADDED_DLL_DIRECTORIES
    if _DLL_SEARCH_PATH_SETUP_DONE:
        return

    # Step 1: BLAS thread safety env — always, independent of ffmpeg dir.
    _apply_blas_thread_safety_env()

    # Step 2: bundled DLL search path.
    local_bin = _project_local_bin_dir()
    if local_bin is not None:
        local_bin_str = str(local_bin)

        # 2a. Prepend to $PATH (subprocess inheritance + legacy LoadLibrary).
        old_path = os.environ.get("PATH", "")
        path_parts = old_path.split(os.pathsep) if old_path else []
        if local_bin_str not in path_parts:
            os.environ["PATH"] = local_bin_str + (os.pathsep + old_path if old_path else "")

        # 2b. Modern Win32 DLL search order (ctypes.CDLL / Default DLL Search Order).
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                cookie = os.add_dll_directory(local_bin_str)
                _ADDED_DLL_DIRECTORIES.append(cookie)
                log.debug(
                    "detector: registered %s with os.add_dll_directory for native DLL search.",
                    local_bin_str,
                )
            except (OSError, FileNotFoundError) as exc:
                log.warning(
                    "detector: os.add_dll_directory(%r) failed: %s. "
                    "TorchCodec native DLLs may not be able to load their "
                    "FFmpeg dependencies; falling back to PATH prepend only.",
                    local_bin_str, exc,
                )
        log.info(
            "detector: bundled FFmpeg bin added to DLL search path: %s",
            local_bin_str,
        )
    else:
        log.debug(
            "detector: no bundled tools/ffmpeg/bin directory found; "
            "relying on system PATH for native FFmpeg DLL resolution.",
        )

    _DLL_SEARCH_PATH_SETUP_DONE = True


def _project_root() -> Path:
    """Dynamically resolve the project root directory.

    detector.py lives at::

        <project_root>/sceneflow/ffmpeg/detector.py

    so three ``.parent`` hops up from this file land on the project root.
    The result is always an absolute path with no Windows username
    or hard-coded absolute locations baked in.
    """
    return Path(__file__).resolve().parent.parent.parent


def _project_local_bin_dir() -> Optional[Path]:
    """Return ``<project_root>/tools/ffmpeg/bin`` if it exists, else *None*."""
    d = _project_root() / "tools" / "ffmpeg" / "bin"
    return d if d.is_dir() else None


def _run_version(exe: str) -> bool:
    """Return True if `exe -version` exits successfully."""
    try:
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _candidate(name: str) -> Tuple[str, ...]:
    """Return a tuple of candidate filenames for an executable on this OS.

    On Windows, ``name`` and ``name.exe`` are both tried; elsewhere only
    the bare ``name``.  Used so ``find_ffmpeg`` / ``find_ffprobe`` /
    ``find_ffplay`` can probe both variants inside a shared directory.
    """
    if os.name == "nt":
        return (name + ".exe", name)
    return (name,)


def _probe_in_dir(dir_path: Path, exe_name: str) -> Optional[str]:
    """Look for *exe_name* inside *dir_path*, trying platform suffixes."""
    for fname in _candidate(exe_name):
        candidate = dir_path / fname
        if candidate.is_file() and _run_version(str(candidate)):
            return str(candidate)
    return None


def find_ffmpeg() -> Optional[str]:
    """Return absolute path to ``ffmpeg``, or *None* if not found.

    Priority:
      1. Project-local installation: ``<project_root>/tools/ffmpeg/bin/``
      2. Existing known Windows install paths
      3. System ``PATH`` via :func:`shutil.which`
    """
    # 1. Project-local (highest priority — ships with the repo)
    local_bin = _project_local_bin_dir()
    if local_bin is not None:
        found = _probe_in_dir(local_bin, "ffmpeg")
        if found:
            return found

    # 2. Windows-specific known install locations
    for folder in _WIN_SEARCH_PATHS:
        candidate = str(Path(folder) / "ffmpeg.exe")
        if Path(candidate).is_file() and _run_version(candidate):
            return candidate

    # 3. PATH / shutil.which (lowest priority — system-wide install)
    found = shutil.which("ffmpeg")
    if found and _run_version(found):
        return found

    return None


def find_ffprobe() -> Optional[str]:
    """Return absolute path to ``ffprobe``, or *None* if not found.

    Priority:
      1. Project-local installation: ``<project_root>/tools/ffmpeg/bin/``
      2. Same directory as whatever :func:`find_ffmpeg` resolved to
         (they ship together in every FFmpeg distribution)
      3. System ``PATH`` via :func:`shutil.which`
    """
    # 1. Project-local first (guarantees matching 9.0.1 ffprobe when the
    #    repo-local install is present — never mixes a 7.x ffprobe with
    #    a 9.x ffmpeg).
    local_bin = _project_local_bin_dir()
    if local_bin is not None:
        found = _probe_in_dir(local_bin, "ffprobe")
        if found:
            return found

    # 2. Sibling of the resolved ffmpeg binary (they ship together)
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling_dir = Path(ffmpeg).parent
        for fname in _candidate("ffprobe"):
            candidate = sibling_dir / fname
            if candidate.is_file() and _run_version(str(candidate)):
                return str(candidate)

    # 3. System PATH
    found = shutil.which("ffprobe")
    if found and _run_version(found):
        return found

    return None


def find_ffplay() -> Optional[str]:
    """Return absolute path to ``ffplay``, or *None* if not found.

    Mirrors the same priority as :func:`find_ffmpeg` / :func:`find_ffprobe`
    so future ffplay-based features pick up the project-local 9.0.1 build
    automatically.

    Priority:
      1. Project-local installation: ``<project_root>/tools/ffmpeg/bin/``
      2. Same directory as :func:`find_ffmpeg`
      3. System ``PATH`` via :func:`shutil.which`
    """
    local_bin = _project_local_bin_dir()
    if local_bin is not None:
        found = _probe_in_dir(local_bin, "ffplay")
        if found:
            return found

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling_dir = Path(ffmpeg).parent
        for fname in _candidate("ffplay"):
            candidate = sibling_dir / fname
            if candidate.is_file() and _run_version(str(candidate)):
                return str(candidate)

    found = shutil.which("ffplay")
    if found and _run_version(found):
        return found

    return None


def get_ffmpeg_version(ffmpeg_path: str) -> str:
    """Return the FFmpeg version string (e.g. '6.1.1')."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"ffmpeg version (\S+)", result.stdout)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def validate_installation() -> Tuple[bool, str]:
    """
    Return ``(True, version_message)`` when both tools are found,
    ``(False, error_message)`` otherwise.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return (
            False,
            "FFmpeg not found. Install it from https://ffmpeg.org/download.html "
            "and make sure it is on your PATH.",
        )

    ffprobe = find_ffprobe()
    if not ffprobe:
        return (
            False,
            "FFprobe not found. It should be installed alongside FFmpeg. "
            "Please reinstall FFmpeg.",
        )

    version = get_ffmpeg_version(ffmpeg)
    return True, f"FFmpeg {version} found at: {ffmpeg}"
