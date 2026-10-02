# SceneFlow Video Assembler

Turn scene assets (images + voiceover audio) into a finished MP4 video — with Ken-Burns effects, crossfade transitions, AI-generated voiceovers via Gemini TTS, and AI-generated images via Imagen.

---

## Features

- **Scene assembly** — match images and audio by scene number, validate before render
- **Ken-Burns effect** — smooth zoom/pan on still images (Off / Subtle / Medium / Strong / Intense)
- **Crossfade transitions** — fade, dissolve, wipe, slide, or hard cut
- **Gemini TTS voiceover** — generate per-scene voiceover audio from your script using Google Gemini
- **Imagen image generation** — generate scene images from text prompts using Google Imagen
- **Forced alignment** — align generated audio to script words using local CTC alignment (PyTorch)
- **Subtitles** — auto-generate `.ass` subtitle files
- **Real-time progress** — live FFmpeg render progress inside the GUI

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.9 or newer | `tkinter` must be included (standard on Windows/macOS) |
| FFmpeg = 4.3 | Must include `ffprobe`. See installation below. |
| `requests` | HTTP calls to Gemini and Imagen APIs |
| `soundfile` | WAV decoding for forced alignment |
| `torch` + `torchaudio` | **Optional** — only needed for local forced alignment |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/subhashishpoudel/sceneflow.git
cd sceneflow
```

### 2. Install Python dependencies

```bash
pip install requests soundfile
```

If you want **local forced alignment** (word-level audio timestamps without a network call):

```bash
# CPU-only build (recommended unless you have a CUDA GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install FFmpeg

FFmpeg must be installed and accessible. Choose one:

**Windows**
- Download from https://ffmpeg.org/download.html ? "Windows builds"
- Extract the archive and add the `bin/` folder to your system `PATH`
- Or place `ffmpeg.exe`, `ffprobe.exe`, and the `.dll` files into `tools/ffmpeg/bin/` inside this repo (SceneFlow checks there automatically)

**macOS**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt install ffmpeg        # Debian/Ubuntu
sudo dnf install ffmpeg        # Fedora/RHEL
```

### 4. Set your Gemini API key

SceneFlow uses Google Gemini for TTS voiceover generation and Google Imagen for image generation.
You need a Gemini API key from https://aistudio.google.com/app/apikey.

**Windows (persistent — recommended)**
```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key-here", "User")
```
Then restart your terminal.

**Windows (current session only)**
```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

**macOS / Linux**
```bash
export GEMINI_API_KEY="your-key-here"
```
Add that line to `~/.bashrc` or `~/.zshrc` to make it permanent.

> **Optional:** If you want to use a separate key for image generation, also set `GEMINI_IMAGE_API_KEY`. If not set, it falls back to `GEMINI_API_KEY`.

> **Security note:** Never paste your API key into source code or commit it to git. The `.gitignore` in this repo already blocks `.env` files.

---

## Running the app

```bash
python main.py
```

On Windows you can also double-click **`run.bat`** — it reads `GEMINI_API_KEY` from your user environment automatically and launches the app.

---

## Project folder structure

The app works with a project folder you create. Expected layout:

```
MyProject/
+-- script.txt          (optional — for script preview and TTS generation)
+-- images/
¦   +-- 001.png
¦   +-- 002.png
¦   +-- ...
+-- voice/
    +-- 001.wav
    +-- 002.wav
    +-- ...
```

### File naming convention

Images and audio files are matched by the **leading integer** in the filename.

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

Place a `script.txt` in your project folder. Three formats are auto-detected:

### 1. Marker-based (recommended)
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
| **Project Folder** | Click **Browse…** to pick your project folder, then **Load ?** |
| **Detected Scenes** | Review matched pairs — missing files are shown in red |
| **Script Preview** | Click a scene row to preview its script text |
| **Voiceover** | Generate AI voiceover for all scenes using Gemini TTS |
| **Images** | Generate scene images from script text using Imagen |
| **Render Settings** | Choose resolution, FPS, transition style, Ken-Burns intensity, and output file |
| **Action bar** | Click **? Render** to start; **? Cancel** to abort at any time |

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
Image 1 --+              +-- xfade --+
Image 2 --¦  Ken-Burns   ¦           +-- xfade --? final video
Image N --+  (zoompan)   +-------- ...
Audio 1 --+
Audio 2 --¦  concat (audio)          --? final audio
Audio N --+
```

Each image is converted to a video clip with the FFmpeg `zoompan` filter (Ken-Burns). Clips are chained with `xfade` transitions. All audio is concatenated. The final streams are muxed into MP4 with H.264/AAC encoding.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "FFmpeg Not Found" at startup | Install FFmpeg and add its `bin/` folder to your PATH, then restart the terminal and the app |
| TTS / image generation fails with "API key" error | Set `GEMINI_API_KEY` as an environment variable (see step 4 above) |
| Scene is red in the table | A matching image or audio file is missing — add it and reload |
| Render fails with `filter_complex` error | Reduce transition duration — it must be shorter than the shortest scene audio |
| Video is black | Check that images are in a supported format and not corrupted |
| Ken-Burns looks choppy | Use `medium` or `slow` encode preset, or reduce intensity |
| `ModuleNotFoundError: requests` | Run `pip install requests soundfile` |
| Forced alignment not working | Run `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |

---

## Source structure

```
sceneflow/
+-- main.py                          Entry point
+-- run.bat                          Windows launcher (reads API key from user env)
+-- requirements.txt                 Python dependencies
+-- sceneflow/
¦   +-- gui.py                       Tkinter application window
¦   +-- core/
¦   ¦   +-- project_loader.py        Loads a project folder
¦   ¦   +-- asset_manager.py         Image/audio discovery & pairing
¦   ¦   +-- scene_parser.py          Scene detection logic
¦   ¦   +-- script_parser.py         Parses script.txt
¦   ¦   +-- timeline_builder.py      Builds FFmpeg filter_complex
¦   +-- ffmpeg/
¦   ¦   +-- detector.py              Finds ffmpeg/ffprobe on the system
¦   ¦   +-- audio_analyzer.py        Reads audio duration via ffprobe
¦   ¦   +-- audio_splitter.py        Splits audio chunks
¦   ¦   +-- filter_builder.py        Ken-Burns filter expressions
¦   ¦   +-- block_renderer.py        Per-block render logic
¦   ¦   +-- renderer.py              Runs FFmpeg subprocess + progress
¦   +-- tts/
¦   ¦   +-- config.py                API key helpers, model config
¦   ¦   +-- gemini_client.py         Gemini TTS + transcription API wrapper
¦   ¦   +-- voiceover_generator.py   Orchestrates TTS generation per scene
¦   ¦   +-- forced_aligner.py        Local CTC forced alignment (torchaudio)
¦   ¦   +-- text_aligner.py          Text-level alignment utilities
¦   +-- imagen/
¦   ¦   +-- imagen_client.py         Google Imagen API wrapper
¦   ¦   +-- image_generator.py       Per-scene image generation
¦   +-- subtitles/
¦   ¦   +-- ass_generator.py         ASS subtitle file generator
¦   +-- models/
¦       +-- scene.py                 SceneData dataclass
¦       +-- project.py               ProjectData dataclass
¦       +-- render_config.py         RenderConfig dataclass
```

---

## License

MIT
