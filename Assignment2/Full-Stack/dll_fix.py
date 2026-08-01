import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Torch DLLs
torch_lib = ROOT / "venv" / "Lib" / "site-packages" / "torch" / "lib"

# FFmpeg DLLs
ffmpeg_bin = Path(r"H:\ffmpeg\ffmpeg-8.1.2-full_build-shared\bin")

if os.name == "nt":
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    if ffmpeg_bin.exists():
        os.add_dll_directory(str(ffmpeg_bin))