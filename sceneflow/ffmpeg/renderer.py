"""FFmpeg renderer — runs the process and streams progress."""
from __future__ import annotations

import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


def _parse_ffmpeg_time(line: str) -> Optional[float]:
    """Extract ``time=HH:MM:SS.mm`` from an FFmpeg progress line and return seconds."""
    m = _TIME_RE.search(line)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


class RenderJob:
    """
    Encapsulates a single FFmpeg render.

    Parameters
    ----------
    ffmpeg_path:
        Path to the ``ffmpeg`` executable.
    args:
        Argument list (without the ``ffmpeg`` executable itself) produced by
        :class:`~sceneflow.core.timeline_builder.TimelineBuilder`.
    total_duration:
        Expected output duration in seconds, used to calculate progress.
    on_progress:
        Callback ``(fraction: float) → None`` called on each progress update.
        ``fraction`` is in [0.0, 1.0].
    on_log:
        Callback ``(message: str) → None`` called with log messages.
    on_done:
        Callback ``(success: bool, return_code: int) → None`` called on finish.
    """

    def __init__(
        self,
        ffmpeg_path: str,
        args: List[str],
        total_duration: float,
        on_progress: Optional[Callable[[float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[bool, int], None]] = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.args = args
        self.total_duration = max(total_duration, 0.001)
        self.on_progress = on_progress
        self.on_log = on_log
        self.on_done = on_done

        self._process: Optional[subprocess.Popen] = None
        self._cancelled = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch FFmpeg in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Terminate the running FFmpeg process."""
        self._cancelled = True
        if self._process and self._process.poll() is None:
            self._process.terminate()
            log.info("FFmpeg process terminated by user.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        cmd = [self.ffmpeg_path] + self.args
        log.debug("FFmpeg command: %s", " ".join(cmd))

        self._emit_log(f"Starting FFmpeg…\n{' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            assert self._process.stderr is not None
            for raw_line in self._process.stderr:
                line = raw_line.rstrip()
                if not line:
                    continue

                # Emit full log
                self._emit_log(line)

                # Progress update
                t = _parse_ffmpeg_time(line)
                if t is not None and self.on_progress:
                    frac = min(t / self.total_duration, 1.0)
                    self.on_progress(frac)

            self._process.wait()
            rc = self._process.returncode
        except Exception as exc:
            log.exception("Unexpected error in render thread")
            self._emit_log(f"ERROR: {exc}")
            if self.on_done:
                self.on_done(False, -1)
            return

        if self._cancelled:
            self._emit_log("Render cancelled.")
            if self.on_done:
                self.on_done(False, rc)
            return

        if rc == 0:
            self._emit_log("✓ Render complete.")
        else:
            self._emit_log(f"✗ FFmpeg exited with code {rc}.")

        if self.on_done:
            self.on_done(rc == 0, rc)

    def _emit_log(self, message: str) -> None:
        log.debug("ffmpeg: %s", message)
        if self.on_log:
            self.on_log(message)
