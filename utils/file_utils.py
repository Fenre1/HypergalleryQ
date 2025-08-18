from __future__ import annotations
from pathlib import Path
from typing import List, Iterable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile
import os
import warnings

# Be lenient with slightly truncated files (common on the web)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# If you have some very large images, you may want to relax this warning:
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

IMG_EXTS: Set[str] = {".jpg",".jpeg",".gif",".png",".tif",".tiff",".bmp",".webp",".jfif"}

def _check_image(path: Path, mode: str) -> bool:
    try:
        with Image.open(path) as img:
            if mode == "none":
                return True
            elif mode == "header":
                # Touch header-only fields; no full decode
                _ = img.size  # forces header parse
                return (img.size[0] > 0 and img.size[1] > 0)
            elif mode == "load":
                # Decode once; tolerant to some minor corruption
                img.load()
                return True
            elif mode == "verify":
                # Slow & strict; reads entire file to check consistency
                img.verify()
                return True
            else:
                raise ValueError(f"Unknown check mode: {mode}")
    except Exception:
        return False

def get_image_files(
    directory: str,
    check: str = "header",          
    workers: int | None = None,     
    follow_symlinks: bool = False,
) -> List[str]:
    """
    Return image file paths quickly with optional lightweight validation.
    Preserves the original filesystem order.
    """
    base = Path(directory)
    paths: List[Path] = [
        p for p in base.rglob("*")
        if p.is_file() and (follow_symlinks or not p.is_symlink()) and p.suffix.lower() in IMG_EXTS
    ]
    if check == "none":
        return [str(p) for p in paths]

    if workers is None:
        # IO-bound; threads help. Keep bounded to avoid thrashing disks.
        workers = min(32, (os.cpu_count() or 8) * 4)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_check_image, p, check): p for p in paths}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    # Preserve order
    good = [str(p) for p in paths if results.get(p, False)]

    # Dedupe while preserving order (extremely rare you'd have duplicate paths)
    seen: Set[str] = set()
    return [p for p in good if not (p in seen or seen.add(p))]
