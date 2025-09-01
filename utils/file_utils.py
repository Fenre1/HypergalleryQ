from __future__ import annotations
from pathlib import Path
from typing import List, Iterable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFile
import os
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

IMG_EXTS: Set[str] = {".jpg",".jpeg",".gif",".png",".tif",".tiff",".bmp",".webp",".jfif"}

def _check_image(path: Path, mode: str) -> bool:
    try:
        with Image.open(path) as img:
            if mode == "none":
                return True
            elif mode == "header":
                _ = img.size  
                return (img.size[0] > 0 and img.size[1] > 0)
            elif mode == "load":
                img.load()
                return True
            elif mode == "verify":
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
    base = Path(directory)
    paths: List[Path] = [
        p for p in base.rglob("*")
        if p.is_file() and (follow_symlinks or not p.is_symlink()) and p.suffix.lower() in IMG_EXTS
    ]
    if check == "none":
        return [str(p) for p in paths]

    if workers is None:
        workers = min(32, (os.cpu_count() or 8) * 4)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_check_image, p, check): p for p in paths}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    good = [str(p) for p in paths if results.get(p, False)]

    seen: Set[str] = set()
    return [p for p in good if not (p in seen or seen.add(p))]
