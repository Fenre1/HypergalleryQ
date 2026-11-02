# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


block_cipher = None

hiddenimports = []
for package in ("open_clip",):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

hiddenimports += [
    "fcmeans",
    "timm.models.swin_transformer_v2",
    "sklearn.decomposition",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "numba.core",
    "safetensors.torch",
]

datas = []
datas += [("icon.png", ".")]
for package in (
    "timm",
    "open_clip_torch",
    "torch",
    "torchvision",
    "scikit-learn",
    "scipy",
    "numba",
    "fuzzy-c-means",
    "h5py",
    "Pillow",
    "safetensors",
):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

for package in ("open_clip", "timm"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass


a = Analysis(
    ["HypergalleryQ.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyinstaller_runtime_hook.py"],
    excludes=[
        "tensorflow",
        "tensorboard",
        "jax",
        "jaxlib",
        "paddle",
        "mlx",
        "IPython",
        "scipy._lib.array_api_compat.torch",
        "sklearn.externals.array_api_compat.torch",
        "torchvision.datasets",
        "torchvision.io",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HyperGalleryDash",
    icon="icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HyperGalleryDash",
)
