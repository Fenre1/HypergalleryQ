from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Return the project root in development or the executable folder in PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def model_dir() -> Path:
    override = os.environ.get("HYPERGALLERY_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return app_root() / "models"


def bundled_resource_path(relative_path: str | os.PathLike[str]) -> Path:
    base = Path(getattr(sys, "_MEIPASS", app_root()))
    return base / relative_path
