from pathlib import Path
from typing import List
from PIL import Image

def get_image_files(directory: str) -> List[str]:
    exts = {".jpg",".jpeg",".gif",".png",".tif",".tiff",".bmp",".webp"}
    file_paths: List[str] = []

    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                with Image.open(p) as img:
                    img.verify()
                file_paths.append(str(p))
            except Exception as e:
                print(f"Skipping corrupted image {p}: {e}")

    # dedupe while preserving order
    seen = set()
    return [p for p in file_paths if not (p in seen or seen.add(p))]