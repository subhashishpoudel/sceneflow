"""SceneFlow Video Assembler — application entry point."""
from __future__ import annotations

import logging
import sys
import tkinter as tk
from tkinter import messagebox


def _setup_logging() -> str:
    """Set up logging to both stdout and a per-run log file.

    Returns the path of the log file so the GUI can display it.
    """
    import datetime
    import os
    from pathlib import Path

    # Create logs/ directory next to this script
    log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console_handler)

    # File handler (immediately flushed)
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # Force immediate flush on every log message
    file_handler.flush = lambda: file_handler.stream.flush()
    root_logger.addHandler(file_handler)

    return str(log_file)


def main() -> None:
    log_file = _setup_logging()
    log = logging.getLogger("sceneflow.main")

    log.info("Log file: %s", log_file)

    # ── Native DLL search + BLAS thread safety (BEFORE any torch/numpy) ────
    # This MUST happen BEFORE importing sceneflow.gui (which transitively
    # imports forced_aligner → torchaudio → torchcodec → ctypes.CDLL calls
    # that link against the FFmpeg shared DLLs avcodec-63.dll / avformat-63.dll
    # etc.)  Those DLLs live in tools/ffmpeg/bin, and Windows will not find
    # them unless we register the directory first.  Without this step the
    # user sees "Could not load libtorchcodec" even though both torchcodec
    # and ffmpeg are actually installed.
    from sceneflow.ffmpeg.detector import register_bundled_ffmpeg_dll_search_path
    register_bundled_ffmpeg_dll_search_path()

    # ── FFmpeg detection ──────────────────────────────────────────────────
    from sceneflow.ffmpeg.detector import (
        find_ffmpeg,
        find_ffprobe,
        get_ffmpeg_version,
        validate_installation,
    )

    is_ok, msg = validate_installation()

    # Create root window first (needed for messagebox)
    root = tk.Tk()
    root.withdraw()   # hide until fully built

    if not is_ok:
        messagebox.showerror(
            "FFmpeg Not Found",
            f"{msg}\n\n"
            "Please install FFmpeg from https://ffmpeg.org/download.html\n"
            "and make sure 'ffmpeg' and 'ffprobe' are on your PATH.\n\n"
            "SceneFlow cannot run without FFmpeg.",
        )
        root.destroy()
        sys.exit(1)

    ffmpeg_path  = find_ffmpeg()
    ffprobe_path = find_ffprobe()
    version      = get_ffmpeg_version(ffmpeg_path)  # type: ignore[arg-type]
    log.info("FFmpeg: %s  (%s)", ffmpeg_path, version)

    # ── Build and show GUI ────────────────────────────────────────────────
    from sceneflow.gui import SceneFlowApp

    root.deiconify()
    app = SceneFlowApp(root, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path)
    app.set_ffmpeg_label(f"FFmpeg {version}")

    root.mainloop()


if __name__ == "__main__":
    main()
