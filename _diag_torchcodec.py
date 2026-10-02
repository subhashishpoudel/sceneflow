import sys, os
from pathlib import Path

print('Python:', sys.version)

# 1. Check soundfile availability
print('\n--- soundfile check ---')
try:
    import soundfile as sf
    print('soundfile version:', sf.__version__)
    print('libsndfile version:', sf.__libsndfile_version__)
except Exception as e:
    print('soundfile FAILED:', type(e).__name__, e)

# 2. Check torch/torchaudio/torchcodec
print('\n--- torch stack check ---')
try:
    import torch
    print('torch version:', torch.__version__)
except Exception as e:
    print('torch FAILED:', type(e).__name__, e)

try:
    import torchaudio
    print('torchaudio version:', torchaudio.__version__)
except Exception as e:
    print('torchaudio FAILED:', type(e).__name__, e)

try:
    import torchcodec
    print('torchcodec found. torchcodec file:', torchcodec.__file__)
except Exception as e:
    print('torchcodec import failed:', type(e).__name__, e)

# 3. Look in torchcodec dir for core dlls
print('\n--- torchcodec DLL dir check ---')
tc_dir = r'C:\Users\DELL\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torchcodec'
if os.path.isdir(tc_dir):
    dlls = [f for f in os.listdir(tc_dir) if f.lower().endswith('.dll')]
    print('DLLs in torchcodec dir:', dlls)
    for f in ['libtorchcodec_core4.dll','libtorchcodec_core5.dll','libtorchcodec_core6.dll',
              'libtorchcodec_core7.dll','libtorchcodec_core8.dll','libtorchcodec_core9.dll']:
        p = os.path.join(tc_dir, f)
        print(f'  {f} exists={os.path.exists(p)} size={os.path.getsize(p) if os.path.exists(p) else "N/A"}')
else:
    print('torchcodec dir not found at expected path')
    tc_dir2 = None
    try:
        import torchcodec
        tc_dir2 = os.path.dirname(torchcodec.__file__)
    except: pass
    if tc_dir2 and os.path.isdir(tc_dir2):
        dlls = [f for f in os.listdir(tc_dir2) if f.lower().endswith('.dll')]
        print(f'DLLs at torchcodec real dir ({tc_dir2}): {dlls}')

# 4. Check bundled FFmpeg bin directory for DLLs
print('\n--- Bundled FFmpeg bin DLL check ---')
project_root = Path.cwd()
ffmpeg_bin = project_root / 'tools' / 'ffmpeg' / 'bin'
print('Project root:', project_root)
print('Bundled ffmpeg bin exists:', ffmpeg_bin.is_dir())
if ffmpeg_bin.is_dir():
    items = sorted([p.name for p in ffmpeg_bin.iterdir()])
    dlls = [n for n in items if n.lower().endswith('.dll')]
    exes = [n for n in items if n.lower().endswith('.exe')]
    print(f'DLLs ({len(dlls)}):', dlls[:20])
    print(f'EXEs ({len(exes)}):', exes)
    avcodec = [n for n in dlls if n.lower().startswith('avcodec')]
    avformat = [n for n in dlls if n.lower().startswith('avformat')]
    avutil = [n for n in dlls if n.lower().startswith('avutil')]
    swresample = [n for n in dlls if n.lower().startswith('swresample')]
    print(f'avcodec DLLs: {avcodec}')
    print(f'avformat DLLs: {avformat}')
    print(f'avutil DLLs: {avutil}')
    print(f'swresample DLLs: {swresample}')

# 5. Check torchcodec dependency loader - what happens when we try to load core9.dll manually?
print('\n--- Try to load libtorchcodec_core9.dll via ctypes (wrapped try) ---')
import ctypes
core9_path = os.path.join(tc_dir, 'libtorchcodec_core9.dll') if os.path.isdir(tc_dir) else None
if core9_path and os.path.exists(core9_path):
    # Try adding bundled ffmpeg bin to PATH first for DLL search
    prev_path = os.environ.get('PATH', '')
    if ffmpeg_bin.is_dir():
        os.environ['PATH'] = str(ffmpeg_bin) + os.pathsep + prev_path
    try:
        h = ctypes.CDLL(core9_path)
        print(f'  SUCCESS: loaded {core9_path} -> handle={h}')
    except Exception as e:
        print(f'  FAILED (w/ bundled ffmpeg on PATH): {type(e).__name__}: {e}')
    finally:
        os.environ['PATH'] = prev_path
    # Now try WITHOUT bundled ffmpeg
    try:
        h2 = ctypes.CDLL(core9_path)
        print(f'  Loaded WITHOUT prepending ffmpeg bin -> handle={h2}')
    except Exception as e:
        print(f'  WITHOUT ffmpeg bin prepend: {type(e).__name__}: {e}')

# 6. Check OpenBLAS warning
print('\n--- numpy / openblas check ---')
try:
    import numpy as np
    print('numpy version:', np.__version__)
    a = np.random.rand(100, 100)
    b = np.dot(a, a.T)
    print('numpy dot() test OK, shape:', b.shape)
except Exception as e:
    print('numpy FAILED:', type(e).__name__, e)
