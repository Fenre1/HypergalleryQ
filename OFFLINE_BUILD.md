# Offline PyInstaller Build

This app is configured to run without model downloads at runtime. Model files are loaded from `models/` beside the source tree during development, or beside `HyperGalleryDash.exe` in a PyInstaller one-folder build.

You can override the model folder with:

```powershell
$env:HYPERGALLERY_MODEL_DIR = "D:\HyperGalleryDashModels"
```

## Required model files

Place these files exactly as shown:

```text
models/
  swinv2_large_window12to24_192to384.ms_in22k_ft_in1k/
    model.safetensors
  CLIP-ViT-B-32-laion2B-s34B-b79K/
    open_clip_model.safetensors
```

Download URLs:

- https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k/resolve/main/model.safetensors
- https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_model.safetensors

On an internet-connected machine you can also run:

```powershell
python scripts\download_models.py
```

## Build

The build uses `icon.ico` for the Windows executable icon and bundles `icon.png` for the Qt task-bar/window icon.

For the distributable build, use a clean CPU-only virtual environment. The current app code can still run from your normal Python environment, but CUDA-enabled Torch wheels are fragile in PyInstaller and require the target machine to have a compatible NVIDIA driver/CUDA runtime stack. CPU-only Torch makes the offline package much smaller and avoids startup failures such as `WinError 1114` while loading `torch\lib\c10.dll`.

```powershell
python -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install --upgrade pip
.\.venv-build\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-build\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv-build\Scripts\python.exe -m pip install -r requirements-build-cpu.txt
```

Then build from that CPU-only environment:

```powershell
.\.venv-build\Scripts\pyinstaller.exe --clean --noconfirm hypergallery_offline.spec
```

After building, copy the external `models/` folder into:

```text
dist/HyperGalleryDash/models/
```

The runtime sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. If a required model file is missing, the app raises a local-file error instead of downloading it.
