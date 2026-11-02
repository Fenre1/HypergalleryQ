from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from utils.model_registry import REQUIRED_MODELS


REPOS = {
    "swinv2_large_window12to24_192to384": (
        "timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        "model.safetensors",
    ),
    "openclip_vit_b_32_laion2b_s34b_b79k": (
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "open_clip_model.safetensors",
    ),
}


def main() -> None:
    for asset in REQUIRED_MODELS:
        repo_id, filename = REPOS[asset.key]
        destination = asset.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {asset.display_name}")
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached_path, destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
