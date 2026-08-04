import os
import site
import shutil
from pathlib import Path

# Find torch package
site_packages = site.getsitepackages()[0]

torch_lib = os.path.join(site_packages, "torch", "lib")
ffmpeg_dir = os.environ.get("VOICE_STUDIO_FFMPEG_DIR")
ffmpeg_bin = Path(ffmpeg_dir) if ffmpeg_dir else None
if ffmpeg_bin is None:
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_bin = Path(ffmpeg_path).resolve().parent if ffmpeg_path else None

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(torch_lib)
    if ffmpeg_bin is not None:
        os.add_dll_directory(str(ffmpeg_bin))