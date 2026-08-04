import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

torch_lib = ROOT / "venv" / "Lib" / "site-packages" / "torch" / "lib"
ffmpeg_dir = os.environ.get("VOICE_STUDIO_FFMPEG_DIR")
ffmpeg_bin = Path(ffmpeg_dir) if ffmpeg_dir else None
if ffmpeg_bin is None:
    ffmpeg_path = shutil.which("ffmpeg")
    ffmpeg_bin = Path(ffmpeg_path).resolve().parent if ffmpeg_path else None

if os.name == "nt":
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))
    if ffmpeg_bin and ffmpeg_bin.exists():
        os.add_dll_directory(str(ffmpeg_bin))