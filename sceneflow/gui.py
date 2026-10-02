"""SceneFlow GUI — built entirely with Tkinter (stdlib only)."""
from __future__ import annotations

import datetime
import logging
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from sceneflow.models.render_config import RenderConfig
from sceneflow.models.project import ProjectData

log = logging.getLogger(__name__)

# ── Early native DLL search + BLAS thread safety ───────────────────────────
# Register tools/ffmpeg/bin on the Windows native DLL search path BEFORE any
# code path imports torch/torchaudio/torchcodec (which happens lazily when the
# user clicks "Generate Voiceover").  Without this, ctypes.CDLL inside
# torchcodec fails to find avcodec-63.dll / avformat-63.dll even though those
# DLLs are bundled in the repo.  This call is idempotent — main() also makes
# it, but we repeat here so gui.py usable as a standalone module import.
try:
    from sceneflow.ffmpeg.detector import register_bundled_ffmpeg_dll_search_path
    register_bundled_ffmpeg_dll_search_path()
except Exception:  # noqa: BLE001 - never block GUI startup on DLL setup
    pass

# ─────────────────────── colour palette ────────────────────────────────────
BG       = "#1e1e2e"   # main background
SURFACE  = "#2a2a3e"   # panel background
BORDER   = "#3d3d5c"   # separator / border
FG       = "#cdd6f4"   # primary text
MUTED    = "#6c7086"   # secondary text
ACCENT   = "#89b4fa"   # blue accent
GREEN    = "#a6e3a1"   # success
YELLOW   = "#f9e2af"   # warning
RED      = "#f38ba8"   # error
BTN_BG   = "#313244"   # button background
BTN_ACTIVE = "#45475a" # button hover


# ─────────────────────── helpers ───────────────────────────────────────────

def _make_label(parent, text, fg=FG, font_weight="normal", size=10, **kw):
    return tk.Label(
        parent, text=text, fg=fg, bg=kw.pop("bg", SURFACE),
        font=("Segoe UI", size, font_weight), **kw
    )

def _make_button(parent, text, command, width=14, **kw):
    options = {
        "bg": BTN_BG,
        "fg": FG,
        "activebackground": BTN_ACTIVE,
        "activeforeground": FG,
        "relief": "flat",
        "bd": 0,
        "padx": 10,
        "pady": 6,
        "font": ("Segoe UI", 10),
        "cursor": "hand2",
        "width": width,
    }

    # Allow individual buttons to override the defaults
    options.update(kw)

    return tk.Button(
        parent,
        text=text,
        command=command,
        **options
    )



def _section_frame(parent, title, **kw):
    """Return a LabelFrame styled to match the dark palette."""
    return tk.LabelFrame(
        parent, text=f"  {title}  ", fg=ACCENT, bg=SURFACE,
        font=("Segoe UI", 9, "bold"), bd=1, relief="groove",
        labelanchor="nw", padx=8, pady=6, **kw
    )


# ─────────────────────── main window ───────────────────────────────────────

class SceneFlowApp:
    """
    Root application window.

    Sections
    --------
    1. Project folder picker
    2. Scene list table + script preview
    3. Render settings
    4. Progress + log
    5. Action bar
    """

    def __init__(self, root: tk.Tk, ffmpeg_path: str, ffprobe_path: str) -> None:
        self.root = root
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

        self._project: Optional[ProjectData] = None
        self._render_job = None
        self._render_start: float = 0.0
        self._tts_queue: queue.Queue = queue.Queue()

        # Try to read the log file path set by main._setup_logging()
        import logging
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                self._log_file_path = h.baseFilename
                break
        else:
            self._log_file_path = None

        self._setup_root()
        self._build_ui()
        self._set_state("empty")

    # ──────────────────────── window setup ─────────────────────────────────

    def _setup_root(self) -> None:
        self.root.title("SceneFlow Video Assembler")
        self.root.geometry("1380x920")
        self.root.minsize(1200, 780)
        self.root.configure(bg=BG)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # Configure ttk styles
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Treeview",
                        background=SURFACE, fieldbackground=SURFACE,
                        foreground=FG, rowheight=26, borderwidth=0,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                        background=BTN_BG, foreground=ACCENT,
                        font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Treeview", background=[("selected", BTN_ACTIVE)])
        style.configure("Horizontal.TProgressbar",
                        background=ACCENT, troughcolor=BTN_BG,
                        borderwidth=0, thickness=20)
        style.configure("TCombobox",
                        fieldbackground=BTN_BG, background=BTN_BG,
                        foreground=FG, selectbackground=BTN_ACTIVE,
                        arrowcolor=FG)
        style.map("TCombobox", fieldbackground=[("readonly", BTN_BG)])

    # ──────────────────────── UI construction ──────────────────────────────

    def _build_ui(self) -> None:
        # ── header bar ──────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=SURFACE, height=48)
        hdr.pack(fill="x", side="top")
        tk.Label(
            hdr, text="⬡  SceneFlow Video Assembler", fg=ACCENT, bg=SURFACE,
            font=("Segoe UI", 14, "bold"), pady=10, padx=16
        ).pack(side="left")
        tk.Label(
            hdr, text="Automatic scene assembly → FFmpeg → MP4",
            fg=MUTED, bg=SURFACE, font=("Segoe UI", 9), pady=10
        ).pack(side="left")

        # ── action bar FIRST so it always claims its space at the bottom ─
        self._build_action_bar()

        # ── main content (left + right columns) fills remaining space ────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        left = tk.Frame(main, bg=BG, width=520)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(main, bg=BG, width=540)
        right.pack(side="right", fill="both", expand=True)

        self._build_project_picker(left)
        self._build_scene_table(left)
        self._build_script_preview(left)
        self._build_render_settings(right)
        self._build_progress_panel(right)

    # ── Project picker ───────────────────────────────────────────────────

    def _build_project_picker(self, parent) -> None:
        frm = _section_frame(parent, "Project Folder")
        frm.pack(fill="x", pady=(0, 8))

        row = tk.Frame(frm, bg=SURFACE)
        row.pack(fill="x")
        _make_label(row, "Folder:").pack(side="left")

        self._project_var = tk.StringVar()
        entry = tk.Entry(
            row, textvariable=self._project_var,
            bg=BTN_BG, fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 9), bd=4
        )
        entry.pack(side="left", fill="x", expand=True, padx=(6, 6))

        _make_button(row, "Browse…", self._browse_project, width=10).pack(side="left", padx=(0, 4))
        _make_button(row, "Load ↻", self._load_project, width=8).pack(side="left")

        self._status_label = _make_label(frm, "No project loaded.", fg=MUTED, size=9)
        self._status_label.pack(anchor="w", pady=(4, 0))

    # ── Scene table ──────────────────────────────────────────────────────

    def _build_scene_table(self, parent) -> None:
        frm = _section_frame(parent, "Detected Scenes")
        frm.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("#", "Duration", "Image", "Audio", "Status")
        self._tree = ttk.Treeview(frm, columns=cols, show="headings", height=10)
        widths = [55, 80, 180, 180, 80]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center" if w < 100 else "w")

        vsb = ttk.Scrollbar(frm, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Tag colours
        self._tree.tag_configure("ok",      foreground=GREEN)
        self._tree.tag_configure("missing", foreground=RED)
        self._tree.tag_configure("warn",    foreground=YELLOW)

        self._tree.bind("<<TreeviewSelect>>", self._on_scene_select)

    # ── Script preview ───────────────────────────────────────────────────

    def _build_script_preview(self, parent) -> None:
        frm = _section_frame(parent, "Script Preview")
        frm.pack(fill="x")

        self._script_text_widget = tk.Text(
            frm, height=5, bg=BTN_BG, fg=FG, insertbackground=FG,
            font=("Segoe UI", 9), relief="flat", wrap="word",
            state="disabled", padx=6, pady=4
        )
        self._script_text_widget.pack(fill="x")

    # ── Render settings ──────────────────────────────────────────────────

    def _build_render_settings(self, parent) -> None:
        frm = _section_frame(parent, "Render Settings")
        frm.pack(fill="x", pady=(0, 8))

        def row(label_text, widget_factory, pady=3):
            r = tk.Frame(frm, bg=SURFACE)
            r.pack(fill="x", pady=pady)
            _make_label(r, label_text, width=20, anchor="w").pack(side="left")
            w = widget_factory(r)
            if w:
                w.pack(side="left", padx=(4, 0))
            return r

        # Resolution
        self._res_var = tk.StringVar(value="1920×1080")
        row("Resolution:", lambda p: ttk.Combobox(
            p, textvariable=self._res_var, width=14, state="readonly",
            values=["1920×1080", "1280×720", "3840×2160", "1080×1920 (portrait)"]
        ))

        # FPS
        self._fps_var = tk.StringVar(value="30")
        row("Frame Rate (fps):", lambda p: ttk.Combobox(
            p, textvariable=self._fps_var, width=8, state="readonly",
            values=["24", "25", "30", "60"]
        ))

        # Transition type
        self._trans_var = tk.StringVar(value="fade")
        row("Transition:", lambda p: ttk.Combobox(
            p, textvariable=self._trans_var, width=14, state="readonly",
            values=["fade", "dissolve", "wipeleft", "wiperight",
                    "slideleft", "slideright", "none"]
        ))

        # Transition duration
        self._trans_dur_var = tk.DoubleVar(value=0.5)
        def _make_trans_dur(p):
            sub = tk.Frame(p, bg=SURFACE)
            self._trans_dur_lbl = _make_label(sub, "0.50 s", size=9, fg=MUTED)
            sc = tk.Scale(
                sub, from_=0.1, to=2.0, resolution=0.05,
                orient="horizontal", variable=self._trans_dur_var,
                bg=SURFACE, fg=FG, highlightthickness=0,
                troughcolor=BTN_BG, sliderrelief="flat", width=12,
                showvalue=False, length=140,
                command=lambda v: self._trans_dur_lbl.config(text=f"{float(v):.2f} s")
            )
            sc.pack(side="left")
            self._trans_dur_lbl.pack(side="left", padx=(6, 0))
            return sub
        row("Transition duration:", _make_trans_dur)

        # Ken Burns intensity
        self._kb_var = tk.DoubleVar(value=0.5)
        kb_labels = ["Off", "Subtle", "Medium", "Strong", "Intense"]
        def _make_kb(p):
            sub = tk.Frame(p, bg=SURFACE)
            self._kb_lbl = _make_label(sub, "Medium", size=9, fg=MUTED)
            def _update_kb_lbl(v):
                idx = round(float(v) * 4)
                self._kb_lbl.config(text=kb_labels[idx])
            sc = tk.Scale(
                sub, from_=0.0, to=1.0, resolution=0.25,
                orient="horizontal", variable=self._kb_var,
                bg=SURFACE, fg=FG, highlightthickness=0,
                troughcolor=BTN_BG, sliderrelief="flat", width=12,
                showvalue=False, length=140,
                command=_update_kb_lbl
            )
            sc.pack(side="left")
            self._kb_lbl.pack(side="left", padx=(6, 0))
            return sub
        row("Ken-Burns intensity:", _make_kb)

        # Codec preset
        self._preset_var = tk.StringVar(value="fast")
        row("Encode preset:", lambda p: ttk.Combobox(
            p, textvariable=self._preset_var, width=12, state="readonly",
            values=["ultrafast", "fast", "medium", "slow"]
        ))

        # Block rendering (low-memory mode)
        self._block_var = tk.IntVar(value=10)
        self._block_enabled_var = tk.BooleanVar(value=True)
        def _make_block(p):
            sub = tk.Frame(p, bg=SURFACE)
            chk = tk.Checkbutton(
                sub, text="Block render (low-memory)",
                variable=self._block_enabled_var,
                bg=SURFACE, fg=FG, selectcolor=BTN_BG,
                activebackground=SURFACE, activeforeground=FG,
                font=("Segoe UI", 9),
                command=lambda: self._block_slider.config(
                    state=("normal" if self._block_enabled_var.get() else "disabled")
                )
            )
            chk.pack(side="left")
            def _update_block_lbl(v):
                n_per = int(float(v))
                if self._project and self._project.scenes:
                    import math
                    n_clips = max(1, math.ceil(len(self._project.scenes) / n_per))
                    self._block_lbl.config(text=f"{n_per} scenes/clip  ({n_clips} clips)")
                else:
                    self._block_lbl.config(text=f"{n_per} scenes/clip")
            self._block_slider = tk.Scale(
                sub, from_=5, to=30, resolution=1, orient="horizontal",
                variable=self._block_var, bg=SURFACE, fg=FG,
                highlightthickness=0, troughcolor=BTN_BG, sliderrelief="flat",
                width=12, showvalue=False, length=120,
                command=_update_block_lbl,
            )
            self._block_slider.pack(side="left", padx=(8, 0))
            self._block_lbl = _make_label(sub, "10 scenes/clip  (8 clips)", size=9, fg=MUTED)
            self._block_lbl.pack(side="left", padx=(6, 0))
            return sub
        row("Block Render:", _make_block)

        # Subtitles
        self._subtitle_var = tk.BooleanVar(value=False)
        self._subtitle_template_var = tk.StringVar()
        self._subtitle_words_var = tk.IntVar(value=5)
        def _make_subtitles(p):
            sub = tk.Frame(p, bg=SURFACE)
            chk = tk.Checkbutton(
                sub, text="Burn subtitles",
                variable=self._subtitle_var,
                bg=SURFACE, fg=FG, selectcolor=BTN_BG,
                activebackground=SURFACE, activeforeground=FG,
                font=("Segoe UI", 9),
            )
            chk.pack(side="left")
            self._subtitle_entry = tk.Entry(
                sub, textvariable=self._subtitle_template_var,
                bg=BTN_BG, fg=FG, insertbackground=FG,
                relief="flat", font=("Segoe UI", 9), bd=4, width=18,
                state="readonly",
            )
            self._subtitle_entry.pack(side="left", padx=(8, 4))
            _make_button(
                sub, "Template…",
                lambda: self._subtitle_template_var.set(
                    filedialog.askopenfilename(
                        title="Select ASS subtitle template",
                        filetypes=[("ASS Subtitles", "*.ass"), ("All Files", "*.*")],
                    ) or self._subtitle_template_var.get()
                ),
                width=10,
            ).pack(side="left")
            _make_label(sub, "Words/caption:", fg=MUTED, size=9).pack(side="left", padx=(10, 2))
            tk.Spinbox(
                sub, from_=1, to=15, width=3,
                textvariable=self._subtitle_words_var,
                bg=BTN_BG, fg=FG, buttonbackground=BTN_BG,
                relief="flat", font=("Segoe UI", 9),
            ).pack(side="left")
            return sub
        row("Subtitles:", _make_subtitles)

        # Output path
        def _make_output(p):
            self._output_var = tk.StringVar()
            sub = tk.Frame(p, bg=SURFACE)
            tk.Entry(
                sub, textvariable=self._output_var,
                bg=BTN_BG, fg=FG, insertbackground=FG,
                relief="flat", font=("Segoe UI", 9), bd=4, width=28
            ).pack(side="left")
            _make_button(sub, "…", self._browse_output, width=3).pack(side="left", padx=(4, 0))
            return sub
        row("Output file:", _make_output)

    # ── Progress panel ───────────────────────────────────────────────────

    def _build_progress_panel(self, parent) -> None:
        frm = _section_frame(parent, "Render Progress")
        frm.pack(fill="both", expand=True, pady=(0, 8))

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            frm, variable=self._progress_var,
            maximum=1.0, style="Horizontal.TProgressbar"
        )
        self._progress_bar.pack(fill="x", pady=(0, 4))

        # Stats row
        stats = tk.Frame(frm, bg=SURFACE)
        stats.pack(fill="x", pady=(0, 4))
        self._pct_label    = _make_label(stats, "0 %", size=12, font_weight="bold", fg=ACCENT)
        self._pct_label.pack(side="left")
        self._eta_label    = _make_label(stats, "", size=9, fg=MUTED)
        self._eta_label.pack(side="right")
        self._elapsed_label = _make_label(stats, "", size=9, fg=MUTED)
        self._elapsed_label.pack(side="right", padx=(0, 20))

        # Log
        self._log = scrolledtext.ScrolledText(
            frm, height=18, bg=BTN_BG, fg=FG, font=("Consolas", 8),
            relief="flat", state="disabled", wrap="word"
        )
        self._log.pack(fill="both", expand=True)

    # ── Action bar ───────────────────────────────────────────────────────

    def _build_action_bar(self) -> None:
        bar = tk.Frame(self.root, bg=SURFACE, height=52)
        bar.pack(fill="x", side="bottom", pady=(6, 0))

        self._render_btn = _make_button(
            bar, "▶  Render", self._start_render,
            bg="#1e6e42", activebackground="#25874f", width=14
        )
        self._render_btn.pack(side="right", padx=(0, 16), pady=8)

        self._cancel_btn = _make_button(
            bar, "✕  Cancel", self._cancel_render,
            bg="#6e1e1e", activebackground="#872525", width=10
        )
        self._cancel_btn.pack(side="right", padx=(0, 6), pady=8)

        self._open_output_btn = _make_button(
            bar, "📂 Open Output", self._open_output_folder, width=14
        )
        self._open_output_btn.pack(side="right", padx=(0, 6), pady=8)

        self._voiceover_btn = _make_button(
            bar, "🎙 Voiceover", self._open_voiceover_dialog,
            bg="#3b3b6e", activebackground="#4e4e8a", width=14
        )
        self._voiceover_btn.pack(side="right", padx=(0, 6), pady=8)

        self._import_audio_btn = _make_button(
            bar, "📥 Import Audio", self._open_import_audio_dialog,
            bg="#3b3b6e", activebackground="#4e4e8a", width=14
        )
        self._import_audio_btn.pack(side="right", padx=(0, 6), pady=8)

        self._imagegen_btn = _make_button(
            bar, "🖼 Gen Images", self._open_imagegen_dialog,
            bg="#3b5e3b", activebackground="#4e8a4e", width=14
        )
        self._imagegen_btn.pack(side="right", padx=(0, 6), pady=8)

        self._open_log_btn = _make_button(
            bar, "📋 Open Log", self._open_log_folder, width=12
        )
        self._open_log_btn.pack(side="right", padx=(0, 6), pady=8)

        self._open_debug_btn = _make_button(
            bar, "🔊 Debug Audio", self._open_debug_folder, width=14
        )
        self._open_debug_btn.pack(side="right", padx=(0, 6), pady=8)

        self._ffmpeg_label = tk.Label(
            bar, text="", fg=MUTED, bg=SURFACE, font=("Segoe UI", 8)
        )
        self._ffmpeg_label.pack(side="left", padx=16)

    # ──────────────────────── state machine ────────────────────────────────

    def _set_state(self, state: str) -> None:
        """
        "empty"     → no project loaded
        "loaded"    → project ready, can render
        "rendering" → FFmpeg running
        "tts"       → TTS generation running
        "done"      → render finished
        "error"     → render failed
        """
        self._state = state
        is_loaded    = state in ("loaded", "done", "error")
        is_rendering = state == "rendering"
        is_tts       = state == "tts"
        can_render   = state == "loaded"

        is_busy = is_rendering or is_tts or state == "imagegen"
        self._render_btn.config(state="normal" if can_render else "disabled")
        self._cancel_btn.config(state="normal" if is_rendering else "disabled")
        self._open_output_btn.config(state="normal" if state == "done" else "disabled")
        self._voiceover_btn.config(state="disabled" if is_busy else "normal")
        self._import_audio_btn.config(state="disabled" if is_busy else "normal")
        self._imagegen_btn.config(state="disabled" if is_busy else "normal")

    # ──────────────────────── actions ──────────────────────────────────────

    def _browse_project(self) -> None:
        folder = filedialog.askdirectory(title="Select Project Folder")
        if folder:
            self._project_var.set(folder)
            self._load_project()

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")],
        )
        if path:
            self._output_var.set(path)

    def _load_project(self) -> None:
        folder = self._project_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder", "Please select a project folder first.")
            return

        path = Path(folder)
        if not path.is_dir():
            messagebox.showerror("Not found", f"Folder does not exist:\n{folder}")
            return

        # Suggest default output path
        if not self._output_var.get():
            self._output_var.set(str(path / "output.mp4"))

        self._log_message("Loading project…")
        self._status_label.config(text="Loading…", fg=MUTED)

        # Run in thread so GUI stays responsive while ffprobe reads durations
        threading.Thread(target=self._do_load, args=(path,), daemon=True).start()

    def _do_load(self, path: Path) -> None:
        from sceneflow.core.project_loader import ProjectLoader

        config = self._build_config()
        try:
            loader = ProjectLoader(self.ffprobe_path)
            project = loader.load(path, config)
        except Exception as exc:
            self.root.after(0, self._on_load_error, str(exc))
            return
        self.root.after(0, self._on_load_done, project)

    def _on_load_done(self, project: ProjectData) -> None:
        self._project = project

        # Populate tree
        for row in self._tree.get_children():
            self._tree.delete(row)

        for scene in project.scenes:
            tag = "ok" if (scene.image_path.is_file() and scene.audio_path.is_file()) else "missing"
            self._tree.insert("", "end", values=(
                f"{scene.scene_number:03d}",
                scene.duration_str,
                scene.image_path.name,
                scene.audio_path.name,
                "✓ OK" if tag == "ok" else "✗ MISSING",
            ), tags=(tag,))

        if project.validation_errors:
            self._log_message("=== Validation Warnings ===")
            for e in project.validation_errors:
                self._log_message(f"  ⚠ {e}", color=YELLOW)

        paired_count = sum(
            1 for s in project.scenes
            if s.audio_path.is_file() and s.image_path.is_file()
        )
        total_count = len(project.scenes)

        if paired_count == 0 and total_count > 0:
            self._status_label.config(
                text=f"⚠  {total_count} image(s) found — click 🎙 Voiceover to generate audio",
                fg=YELLOW
            )
        elif project.validation_errors:
            self._status_label.config(
                text=f"⚠  {paired_count}/{total_count} scenes ready — check log",
                fg=YELLOW
            )
        else:
            self._status_label.config(
                text=f"✓  {total_count} scenes loaded — total {project.total_duration:.1f}s",
                fg=GREEN
            )

        self._log_message(
            f"Loaded {len(project.scenes)} scenes from '{project.project_path.name}'.",
            color=GREEN
        )

        self._set_state("loaded" if project.is_valid else "error")

    def _on_load_error(self, msg: str) -> None:
        self._status_label.config(text=f"✗  {msg}", fg=RED)
        self._log_message(f"ERROR: {msg}", color=RED)
        messagebox.showerror("Load Error", msg)
        self._set_state("empty")

    def _on_scene_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel or not self._project:
            return
        idx = self._tree.index(sel[0])
        if idx < len(self._project.scenes):
            scene = self._project.scenes[idx]
            self._update_script_preview(scene.script_text or "(no script text for this scene)")

    def _update_script_preview(self, text: str) -> None:
        self._script_text_widget.config(state="normal")
        self._script_text_widget.delete("1.0", "end")
        self._script_text_widget.insert("end", text)
        self._script_text_widget.config(state="disabled")

    # ──────────────────────── render control ───────────────────────────────

    def _start_render(self) -> None:
        if not self._project or not self._project.is_valid:
            messagebox.showwarning("Not ready", "Load a valid project first.")
            return

        output = self._output_var.get().strip()
        if not output:
            messagebox.showwarning("No output path", "Please specify an output file path.")
            return

        # Confirm overwrite
        if Path(output).exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"Output file already exists:\n{output}\n\nOverwrite it?"
            ):
                return

        config = self._build_config()

        # Validate subtitle template on main thread (fast — just a file check)
        # Actual alignment runs in the background thread inside BlockRenderJob.
        if self._subtitle_var.get():
            template = self._subtitle_template_var.get().strip()
            if not template:
                messagebox.showwarning(
                    "No Template",
                    "Please select an ASS subtitle template file."
                )
                return
            if not Path(template).is_file():
                messagebox.showwarning(
                    "Template Not Found",
                    f"Cannot find subtitle template:\n{template}"
                )
                return
            # Store template path + words per group in config —
            # subtitle generation happens in the background thread
            config._subtitle_template_path = template
            config._subtitle_words_per_group = max(1, self._subtitle_words_var.get())
        else:
            config._subtitle_template_path = None
            config._subtitle_words_per_group = 5

        # Re-apply Ken-Burns based on current intensity slider
        from sceneflow.ffmpeg.filter_builder import assign_ken_burns
        assign_ken_burns(self._project.scenes, config)

        from sceneflow.ffmpeg.block_renderer import BlockRenderJob

        try:
            # BlockRenderJob is the unified entry-point:
            #   * block_size <= 1 OR block_size >= scene_count
            #       → internally falls back to single monolithic RenderJob
            #   * block_size > 1 AND block_size < scene_count
            #       → block render + concat demuxer (low-memory mode)
            pass   # config validated below by BlockRenderJob constructor
        except Exception as exc:
            messagebox.showerror("Build Error", str(exc))
            return

        self._progress_var.set(0.0)
        self._pct_label.config(text="0 %")
        self._eta_label.config(text="")
        self._elapsed_label.config(text="")
        self._render_start = time.monotonic()
        self._set_state("rendering")
        self._log_message("=== Render started ===", color=ACCENT)

        self._render_job = BlockRenderJob(
            ffmpeg_path=self.ffmpeg_path,
            scenes=self._project.scenes,
            config=config,
            total_duration=self._project.total_duration,
            on_progress=self._on_render_progress,
            on_log=self._on_render_log,
            on_done=self._on_render_done,
        )
        self._render_job.start()

    def _cancel_render(self) -> None:
        if self._render_job:
            self._render_job.cancel()
            self._log_message("Cancelling render…", color=YELLOW)

    def _on_render_progress(self, fraction: float) -> None:
        self.root.after(0, self._update_progress_ui, fraction)

    def _update_progress_ui(self, fraction: float) -> None:
        self._progress_var.set(fraction)
        pct = int(fraction * 100)
        self._pct_label.config(text=f"{pct} %")
        elapsed = time.monotonic() - self._render_start
        self._elapsed_label.config(text=f"Elapsed: {_fmt_dur(elapsed)}")
        if fraction > 0.01:
            eta = elapsed / fraction - elapsed
            self._eta_label.config(text=f"ETA: {_fmt_dur(eta)}")

    def _on_render_log(self, message: str) -> None:
        self.root.after(0, self._log_message, message)

    def _on_render_done(self, success: bool, return_code: int) -> None:
        self.root.after(0, self._finish_render, success, return_code)

    def _finish_render(self, success: bool, return_code: int) -> None:
        if success:
            self._progress_var.set(1.0)
            self._pct_label.config(text="100 %", fg=GREEN)
            elapsed = time.monotonic() - self._render_start
            self._elapsed_label.config(text=f"Elapsed: {_fmt_dur(elapsed)}")
            self._eta_label.config(text="Done")
            output = self._output_var.get()
            size_mb = Path(output).stat().st_size / (1024 * 1024) if Path(output).exists() else 0
            self._log_message(
                f"✓ Render complete in {_fmt_dur(elapsed)}. Output: {output} ({size_mb:.1f} MB)",
                color=GREEN
            )
            self._set_state("done")
            messagebox.showinfo(
                "Render Complete",
                f"Video saved to:\n{output}\n\nFile size: {size_mb:.1f} MB"
            )
        else:
            self._log_message(
                f"✗ Render failed (exit code {return_code}).", color=RED
            )
            self._set_state("loaded")
            messagebox.showerror(
                "Render Failed",
                f"FFmpeg exited with code {return_code}.\n"
                "Check the log panel for details."
            )

    # ──────────────────────── utilities ────────────────────────────────────

    def _build_config(self) -> RenderConfig:
        res_map = {
            "1920×1080": (1920, 1080),
            "1280×720": (1280, 720),
            "3840×2160": (3840, 2160),
            "1080×1920 (portrait)": (1080, 1920),
        }
        resolution = res_map.get(self._res_var.get(), (1920, 1080))
        fps = int(self._fps_var.get())

        block_size = 0  # 0 or 1 = monolithic (no block render)
        if self._block_enabled_var.get():
            block_size = max(2, int(self._block_var.get()))

        return RenderConfig(
            output_path=self._output_var.get().strip(),
            resolution=resolution,
            fps=fps,
            transition_type=self._trans_var.get(),
            transition_duration=self._trans_dur_var.get(),
            ken_burns_intensity=self._kb_var.get(),
            preset=self._preset_var.get(),
            block_size=block_size,
        )

    def _log_message(self, message: str, color: str = FG) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{timestamp}] {message}\n", ("colored",))
        # Apply colour tag
        tag_name = f"color_{color.replace('#', '')}"
        self._log.tag_configure(tag_name, foreground=color)
        # Retag the last inserted line
        last_line = int(self._log.index("end-2l").split(".")[0])
        self._log.tag_add(tag_name, f"{last_line}.0", f"{last_line}.end")
        self._log.config(state="disabled")
        self._log.see("end")

    def _open_output_folder(self) -> None:
        output = self._output_var.get().strip()
        if output:
            folder = str(Path(output).parent)
            try:
                if os.name == "nt":
                    os.startfile(folder)
                elif os.uname().sysname == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

    def set_ffmpeg_label(self, text: str) -> None:
        self._ffmpeg_label.config(text=text)

    def _open_log_folder(self) -> None:
        """Open the folder containing the per-run log files."""
        import os
        import subprocess
        from pathlib import Path

        log_path = getattr(self, "_log_file_path", None)
        if log_path:
            folder = str(Path(log_path).parent)
        else:
            # Fallback: logs/ next to main.py
            folder = str(Path(__file__).resolve().parent.parent / "logs")
            Path(folder).mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(folder)
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            pass

    def _open_debug_folder(self) -> None:
        """Open the debug_audio folder inside the current project."""
        import os
        import subprocess
        from pathlib import Path

        if not self._project:
            messagebox.showwarning("No Project", "Load a project first.")
            return
        folder = self._project.project_path / "debug_audio"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            elif os.uname().sysname == "Darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass


    # ──────────────────────── voiceover generation ─────────────────────────

    def _open_voiceover_dialog(self) -> None:
        """Open the voiceover settings dialog."""
        if not self._project:
            messagebox.showwarning(
                "No Project",
                "Load a project with a script.txt file first."
            )
            return

        from sceneflow.tts.config import SUGGESTED_VOICES

        dlg = tk.Toplevel(self.root)
        dlg.title("Generate Voiceover")
        dlg.configure(bg=SURFACE)
        dlg.resizable(False, False)
        dlg.grab_set()

        pad = {"padx": 12, "pady": 6}

        # Voice selection
        voice_row = tk.Frame(dlg, bg=SURFACE)
        voice_row.pack(fill="x", **pad)
        _make_label(voice_row, "Voice:", width=18, anchor="w").pack(side="left")
        voice_var = tk.StringVar(value="Kore")
        voice_cb = ttk.Combobox(
            voice_row, textvariable=voice_var, width=16,
            values=SUGGESTED_VOICES
        )
        voice_cb.pack(side="left", padx=(4, 0))

        # Style / tone instruction
        style_row = tk.Frame(dlg, bg=SURFACE)
        style_row.pack(fill="x", **pad)
        _make_label(style_row, "Style (optional):", width=18, anchor="w").pack(side="left")
        style_var = tk.StringVar()
        tk.Entry(
            style_row, textvariable=style_var,
            bg=BTN_BG, fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 9), bd=4, width=32
        ).pack(side="left", padx=(4, 0))

        # Regenerate checkbox
        force_var = tk.BooleanVar(value=False)
        force_row = tk.Frame(dlg, bg=SURFACE)
        force_row.pack(fill="x", **pad)
        tk.Checkbutton(
            force_row, text="Regenerate existing voice files",
            variable=force_var, bg=SURFACE, fg=FG,
            selectcolor=BTN_BG, activebackground=SURFACE,
            font=("Segoe UI", 9)
        ).pack(side="left")

        # Buttons
        btn_row = tk.Frame(dlg, bg=SURFACE)
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        _make_button(
            btn_row, "Generate",
            lambda: self._start_voiceover(
                dlg, voice_var.get().strip(), style_var.get().strip() or None, force_var.get()
            ),
            bg="#3b3b6e", activebackground="#4e4e8a", width=12
        ).pack(side="right")
        _make_button(btn_row, "Cancel", dlg.destroy, width=8).pack(side="right", padx=(0, 6))

    def _start_voiceover(
        self,
        dialog: tk.Toplevel,
        voice: str,
        style: Optional[str],
        force: bool,
    ) -> None:
        """Close the dialog and start TTS generation in a background thread."""
        if not voice:
            messagebox.showwarning("No Voice", "Please enter a voice name.", parent=dialog)
            return

        dialog.destroy()

        self._set_state("tts")
        self._log_message("=== Voiceover generation started ===", color=ACCENT)
        self._log_message(f"  Voice: {voice}  |  Force: {force}", color=MUTED)

        # Drain any stale queue items
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

        project_path = self._project.project_path  # type: ignore[union-attr]

        def _worker():
            from sceneflow.tts.voiceover_generator import generate_all
            from sceneflow.tts.config import MissingApiKeyError

            def _on_progress(scene_num: int, total: int, status: str):
                self._tts_queue.put(("progress", scene_num, total, status))

            try:
                results = generate_all(
                    project_path=project_path,
                    voice=voice,
                    style=style,
                    force=force,
                    progress_callback=_on_progress,
                    ffmpeg_path=self.ffmpeg_path,
                    ffprobe_path=self.ffprobe_path,
                )
                self._tts_queue.put(("done", results))
            except MissingApiKeyError as exc:
                self._tts_queue.put(("error", str(exc)))
            except Exception as exc:
                self._tts_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_tts_queue()

    def _poll_tts_queue(self) -> None:
        """Poll the TTS progress queue and update the UI via root.after()."""
        try:
            while True:
                item = self._tts_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, scene_num, total, status = item
                    color = GREEN if status == "done" else (YELLOW if status == "skipped" else RED)
                    self._log_message(
                        f"  Scene {scene_num:03d} / {total}: {status}", color=color
                    )
                    # Update progress bar
                    self._progress_var.set(scene_num / total)
                    self._pct_label.config(text=f"{int(scene_num / total * 100)} %")

                elif kind == "done":
                    results = item[1]
                    self._on_tts_done(results)
                    return  # stop polling

                elif kind == "error":
                    self._on_tts_error(item[1])
                    return  # stop polling

        except queue.Empty:
            pass

        # Schedule next poll
        self.root.after(100, self._poll_tts_queue)

    def _on_tts_done(self, results: list) -> None:
        """Called in the main thread after TTS generation completes."""
        done    = [r for r in results if r["status"] == "done"]
        skipped = [r for r in results if r["status"] == "skipped"]
        failed  = [r for r in results if r["status"] == "failed"]

        self._progress_var.set(1.0)
        self._pct_label.config(text="100 %", fg=GREEN)

        summary_parts = [
            f"✓ Generated: {len(done)}",
            f"↷ Skipped: {len(skipped)}",
        ]
        if failed:
            failed_nums = ", ".join(f"{r['scene']:03d}" for r in failed)
            summary_parts.append(f"✗ Failed: {len(failed)} (scenes {failed_nums})")

        summary = "\n".join(summary_parts)
        self._log_message("=== Voiceover generation complete ===", color=GREEN)
        for part in summary_parts:
            color = RED if part.startswith("✗") else GREEN
            self._log_message(f"  {part}", color=color)

        if failed:
            for r in failed:
                self._log_message(f"    Scene {r['scene']:03d}: {r['error']}", color=RED)

        # Restore state
        self._set_state("loaded" if self._project and self._project.is_valid else "error")

        # Refresh scene table so newly written voice files appear immediately
        self._load_project()

        messagebox.showinfo("Voiceover Complete", summary)

    def _on_tts_error(self, msg: str) -> None:
        """Called in the main thread if the TTS worker raises an unhandled exception."""
        self._log_message(f"TTS ERROR: {msg}", color=RED)
        self._set_state("loaded" if self._project and self._project.is_valid else "error")
        messagebox.showerror("Voiceover Error", msg)

    # ──────────────────────── import audio & split ──────────────────────────

    def _open_import_audio_dialog(self) -> None:
        """Open the Import Audio & Split dialog."""
        if not self._project:
            messagebox.showwarning(
                "No Project",
                "Load a project with a script.txt file first."
            )
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Import Audio & Split")
        dlg.configure(bg=SURFACE)
        dlg.resizable(False, False)
        dlg.grab_set()

        pad = {"padx": 12, "pady": 6}

        # Info label
        _make_label(
            dlg,
            "Select a WAV file containing the full narration\n"
            "for all scenes in order. SceneFlow will align\n"
            "and split it into per-scene voice files.",
            fg=MUTED, size=9,
        ).pack(fill="x", padx=12, pady=(10, 2))

        # WAV file picker
        file_row = tk.Frame(dlg, bg=SURFACE)
        file_row.pack(fill="x", **pad)
        _make_label(file_row, "WAV file:", width=14, anchor="w").pack(side="left")
        wav_var = tk.StringVar()
        wav_entry = tk.Entry(
            file_row, textvariable=wav_var,
            bg=BTN_BG, fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 9), bd=4, width=28,
            state="readonly",
        )
        wav_entry.pack(side="left", padx=(4, 4))
        _make_button(
            file_row, "Browse…",
            lambda: wav_var.set(
                filedialog.askopenfilename(
                    title="Select WAV file",
                    filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")],
                ) or wav_var.get()
            ),
            width=8,
        ).pack(side="left")

        # Overwrite checkbox
        force_var = tk.BooleanVar(value=False)
        force_row = tk.Frame(dlg, bg=SURFACE)
        force_row.pack(fill="x", **pad)
        tk.Checkbutton(
            force_row, text="Overwrite existing voice files",
            variable=force_var, bg=SURFACE, fg=FG,
            selectcolor=BTN_BG, activebackground=SURFACE,
            font=("Segoe UI", 9)
        ).pack(side="left")

        # Buttons
        btn_row = tk.Frame(dlg, bg=SURFACE)
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        _make_button(
            btn_row, "Import & Split",
            lambda: self._start_import_audio(dlg, wav_var.get().strip(), force_var.get()),
            bg="#3b3b6e", activebackground="#4e4e8a", width=14,
        ).pack(side="right")
        _make_button(btn_row, "Cancel", dlg.destroy, width=8).pack(side="right", padx=(0, 6))

    def _start_import_audio(
        self,
        dialog: tk.Toplevel,
        wav_file: str,
        force: bool,
    ) -> None:
        """Close the dialog and run align+split in a background thread."""
        if not wav_file:
            messagebox.showwarning("No File", "Please select a WAV file.", parent=dialog)
            return
        if not Path(wav_file).is_file():
            messagebox.showwarning("File Not Found", f"Cannot find:\n{wav_file}", parent=dialog)
            return

        dialog.destroy()

        self._set_state("tts")
        self._log_message("=== Import Audio & Split started ===", color=ACCENT)
        self._log_message(f"  File: {wav_file}  |  Force: {force}", color=MUTED)

        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

        project_path = self._project.project_path  # type: ignore[union-attr]

        def _worker():
            from sceneflow.tts.voiceover_generator import import_and_split

            def _on_progress(scene_num: int, total: int, status: str):
                self._tts_queue.put(("progress", scene_num, total, status))

            try:
                results = import_and_split(
                    project_path=project_path,
                    wav_path=Path(wav_file),
                    force=force,
                    progress_callback=_on_progress,
                    ffmpeg_path=self.ffmpeg_path,
                    ffprobe_path=self.ffprobe_path,
                )
                self._tts_queue.put(("done", results))
            except Exception as exc:
                self._tts_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_tts_queue()

    # ──────────────────────── image generation ─────────────────────────────

    def _open_imagegen_dialog(self) -> None:
        """Open the image generation settings dialog."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Generate Images")
        dlg.configure(bg=SURFACE)
        dlg.resizable(False, False)
        dlg.grab_set()

        pad = {"padx": 12, "pady": 6}

        # Info label
        info = tk.Frame(dlg, bg=SURFACE)
        info.pack(fill="x", **pad)
        _make_label(
            info,
            "Reads scene.txt from the project folder.\n"
            "Format:  001 - A mountain at sunrise",
            fg=MUTED, size=9
        ).pack(anchor="w")

        # Force regenerate
        force_var = tk.BooleanVar(value=False)
        force_row = tk.Frame(dlg, bg=SURFACE)
        force_row.pack(fill="x", **pad)
        tk.Checkbutton(
            force_row, text="Regenerate existing images",
            variable=force_var, bg=SURFACE, fg=FG,
            selectcolor=BTN_BG, activebackground=SURFACE,
            font=("Segoe UI", 9)
        ).pack(side="left")

        # Buttons
        btn_row = tk.Frame(dlg, bg=SURFACE)
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        _make_button(
            btn_row, "Generate",
            lambda: self._start_imagegen(dlg, force_var.get()),
            bg="#3b5e3b", activebackground="#4e8a4e", width=12
        ).pack(side="right")
        _make_button(btn_row, "Cancel", dlg.destroy, width=8).pack(side="right", padx=(0, 6))

    def _start_imagegen(self, dialog: tk.Toplevel, force: bool) -> None:
        """Close dialog and start image generation in a background thread."""
        dialog.destroy()

        self._set_state("imagegen")
        self._log_message("=== Image generation started ===", color=ACCENT)
        self._log_message(f"  Force: {force}", color=MUTED)

        # Reuse TTS queue for progress messages
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break

        project_path = self._project.project_path if self._project else None
        if not project_path:
            messagebox.showwarning("No Project", "Load a project first.")
            self._set_state("empty")
            return

        def _worker():
            from sceneflow.imagen.image_generator import generate_all_images
            from sceneflow.tts.config import MissingApiKeyError

            def _on_progress(scene_num: int, total: int, status: str, error: str = ""):
                self._tts_queue.put(("progress", scene_num, total, status, error))

            try:
                results = generate_all_images(
                    project_path=project_path,
                    force=force,
                    progress_callback=_on_progress,
                )
                self._tts_queue.put(("imgdone", results))
            except MissingApiKeyError as exc:
                self._tts_queue.put(("error", str(exc)))
            except Exception as exc:
                self._tts_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_imagegen_queue()

    def _poll_imagegen_queue(self) -> None:
        """Poll image generation progress queue."""
        try:
            while True:
                item = self._tts_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, scene_num, total, status, *rest = item
                    error = rest[0] if rest else ""
                    color = GREEN if status == "done" else (YELLOW if status == "skipped" else RED)
                    self._log_message(f"  Scene {scene_num:03d} / {total}: {status}", color=color)
                    if error:
                        self._log_message(f"    ↳ {error}", color=RED)
                    self._progress_var.set(scene_num / total)
                    self._pct_label.config(text=f"{int(scene_num / total * 100)} %")

                elif kind == "imgdone":
                    self._on_imagegen_done(item[1])
                    return

                elif kind == "error":
                    self._on_imagegen_error(item[1])
                    return

        except queue.Empty:
            pass

        self.root.after(100, self._poll_imagegen_queue)

    def _on_imagegen_done(self, results: list) -> None:
        done    = [r for r in results if r["status"] == "done"]
        skipped = [r for r in results if r["status"] == "skipped"]
        failed  = [r for r in results if r["status"] == "failed"]

        self._progress_var.set(1.0)
        self._pct_label.config(text="100 %", fg=GREEN)

        summary_parts = [f"✓ Generated: {len(done)}", f"↷ Skipped: {len(skipped)}"]
        if failed:
            failed_nums = ", ".join(f"{r['scene']:03d}" for r in failed)
            summary_parts.append(f"✗ Failed: {len(failed)} (scenes {failed_nums})")

        self._log_message("=== Image generation complete ===", color=GREEN)
        for part in summary_parts:
            self._log_message(f"  {part}", color=RED if part.startswith("✗") else GREEN)
        if failed:
            for r in failed:
                self._log_message(f"    Scene {r['scene']:03d}: {r['error']}", color=RED)

        self._set_state("loaded" if self._project and self._project.is_valid else "empty")
        self._load_project()
        messagebox.showinfo("Image Generation Complete", "\n".join(summary_parts))

    def _on_imagegen_error(self, msg: str) -> None:
        self._log_message(f"IMAGE GEN ERROR: {msg}", color=RED)
        self._set_state("loaded" if self._project and self._project.is_valid else "empty")
        messagebox.showerror("Image Generation Error", msg)


# ─────────────────────── formatting helpers ────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
