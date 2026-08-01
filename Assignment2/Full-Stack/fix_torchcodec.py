import os
import site

# Find torch package
site_packages = site.getsitepackages()[0]

torch_lib = os.path.join(site_packages, "torch", "lib")
ffmpeg_bin = r"H:\ffmpeg\ffmpeg-8.1.2-full_build-shared\bin"

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(torch_lib)
    os.add_dll_directory(ffmpeg_bin)