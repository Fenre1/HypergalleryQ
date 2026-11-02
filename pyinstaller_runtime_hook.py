from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_dll_directory(path: Path) -> None:
    if not path.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path))
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


if sys.platform == "win32":
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    torch_lib = bundle_root / "torch" / "lib"
    _add_dll_directory(torch_lib)
    _add_dll_directory(bundle_root)

    if torch_lib.is_dir():
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p
        for dll in sorted(torch_lib.glob("*.dll")):
            ctypes.set_last_error(0)
            handle = kernel32.LoadLibraryExW(str(dll), None, 0x00001100)
            error = ctypes.get_last_error()
            if handle is None and error != 126:
                raise ctypes.WinError(error)
