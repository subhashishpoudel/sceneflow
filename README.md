# SceneFlow Video Assembler

Automatically turn externally-generated scene assets (images + voice audio) into one finished MP4 video using FFmpeg.

---

## What it does

Given a project folder containing:

```
MyProject/
├── script.txt          (optional – for script preview inside the app)
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── voice/
    ├── 001.wav
    ├── 002.wav
    └── ...
```

SceneFlow will:

1. **Detect scenes** — match images and audio files by their scene number prefix.
2. **Validate** — report any missing image or audio file before rendering.
3. **Read exact audio duration** — using `ffprobe`; the audio length drives the scene duration.
4. **Apply Ken-Burns effect** — smooth zoom/pan on each still image (adjustable intensity, or off).
5. **Add crossfade transitions** — `fade`, `dissolve`, `wipe`, `slide`, or hard cut.
6. **Render to MP4** — a single `ffmpeg` command does everything; real-time progress shown.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.9 or newer | `tkinter` must be included (standard on Windows/Mac) |
| FFmpeg ≥ 4.3 | Includes `ffprobe`. Download from <https://ffmpeg.org/download.html> |

**No pip packages are required.** The application uses only the Python standard library.

---

## Installation

1. **Download or clone** this repository.
2. **Install FFmpeg** and make sure `ffmpeg` and `ffprobe` are on your `PATH`:
   - Windows: <https://ffmpeg.org/download.html> → "Windows builds" → extract and add `bin/` to PATH.
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` / `sudo dnf install ffmpeg`
3. **Run** the application:
   ```
   python main.py
   ```

---

## Naming convention for assets

Files inside `images/` and `voice/` are matched by the **leading integer in the filename stem**.

| Image file | Audio file | Scene |
|---|---|---|
| `001.png` | `001.wav` | 1 |
| `002.jpg` | `002.mp3` | 2 |
| `003_final.webp` | `003_v2.flac` | 3 |

### Supported formats

| Type | Extensions |
|---|---|
| Images | `.png` `.jpg` `.jpeg` `.webp` |
| Audio | `.wav` `.mp3` `.m4a` `.aac` `.flac` |

---

## Script file (optional)

Place a `script.txt` in your project root. Three formats are auto-detected:

### 1. Marker-based (recommended for multi-line scenes)
```
[SCENE 001]
The hero stands alone on the mountain,
gazing toward the distant horizon.

[SCENE 002]
A wind sweeps through the valley below.

[SCENE 003]
Everything changes in an instant.
```

### 2. Blank-line-separated
Each paragraph maps to one scene in order.

### 3. One line per scene
Each non-empty line maps to one scene in order.

---

## GUI walkthrough

| Section | What to do |
|---|---|
| **Project Folder** | Click **Browse…** to pick your project folder, then **Load ↻** |
| **Detected Scenes** | Review matched pairs; missing files shown in red |
| **Script Preview** | Click a scene row to preview its script text |
| **Render Settings** | Choose resolution, FPS, transition style, Ken-Burns intensity, and output file |
| **Action bar** | Click **▶ Render** to start; **✕ Cancel** to abort |

---

## Render settings

| Setting | Options | Default |
|---|---|---|
| Resolution | 1920×1080 · 1280×720 · 3840×2160 · 1080×1920 | 1920×1080 |
| Frame Rate | 24 · 25 · 30 · 60 fps | 30 |
| Transition | fade · dissolve · wipeleft · wiperight · slideleft · slideright · none | fade |
| Transition duration | 0.1 – 2.0 s | 0.5 s |
| Ken-Burns intensity | Off · Subtle · Medium · Strong · Intense | Medium |
| Encode preset | ultrafast · fast · medium · slow | fast |

---

## How the FFmpeg pipeline works

```
Image 1 ──┐              ┌── xfade ──┐
Image 2 ──┤  Ken-Burns   │           ├── xfade ──▶ final video
Image N ──┘  (zoompan)   └──────── ...
Audio 1 ──┐
Audio 2 ──┤  concat (audio)          ──▶ final audio
Audio N ──┘
```

Each still image is converted into a video clip using the FFmpeg `zoompan` filter (Ken-Burns). The clips are then chained together with `xfade` transitions. All audio files are concatenated in order. The final video and audio streams are muxed into an MP4 with H.264/AAC encoding.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "FFmpeg Not Found" at startup | Install FFmpeg and add its `bin/` folder to your PATH. Restart the terminal / the app. |
| Scene is red in the table | A matching image or audio file is missing. Add the missing file and reload. |
| Render fails with filter_complex error | Reduce transition duration — it must be shorter than the shortest scene's audio. |
| Video is black | Make sure images are in a supported format and not corrupted. |
| Ken-Burns looks choppy | Increase the encode preset (medium/slow) or reduce intensity. |

---

## Project structure (source)

```
sceneflow_video_assembler/
├── main.py                          Entry point
├── requirements.txt
├── sceneflow/
│   ├── gui.py                       Tkinter application window
│   ├── core/
│   │   ├── project_loader.py        Loads a project folder
│   │   ├── asset_manager.py         Image/audio discovery & pairing
│   │   ├── script_parser.py         Parses script.txt
│   │   └── timeline_builder.py      Builds FFmpeg filter_complex
│   ├── ffmpeg/
│   │   ├── detector.py              Finds ffmpeg/ffprobe on the system
│   │   ├── audio_analyzer.py        Reads audio duration via ffprobe
│   │   ├── filter_builder.py        Ken-Burns filter expressions
│   │   └── renderer.py              Runs FFmpeg subprocess + progress
│   └── models/
│       ├── scene.py                 SceneData dataclass
│       ├── project.py               ProjectData dataclass
│       └── render_config.py         RenderConfig dataclass
```

---

## License

MIT
