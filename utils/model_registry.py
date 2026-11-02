from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime_paths import model_dir


@dataclass(frozen=True)
class ModelAsset:
    key: str
    display_name: str
    relative_path: Path
    url: str

    @property
    def path(self) -> Path:
        return model_dir() / self.relative_path


SWINV2_LARGE = ModelAsset(
    key="swinv2_large_window12to24_192to384",
    display_name="SwinV2 Large 192-to-384, ImageNet-22k fine-tuned on ImageNet-1k",
    relative_path=Path("swinv2_large_window12to24_192to384.ms_in22k_ft_in1k") / "model.safetensors",
    url=(
        "https://huggingface.co/timm/"
        "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k/resolve/main/model.safetensors"
    ),
)

OPENCLIP_VIT_B_32 = ModelAsset(
    key="openclip_vit_b_32_laion2b_s34b_b79k",
    display_name="OpenCLIP ViT-B-32 LAION-2B s34B b79K",
    relative_path=Path("CLIP-ViT-B-32-laion2B-s34B-b79K") / "open_clip_model.safetensors",
    url=(
        "https://huggingface.co/laion/"
        "CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_model.safetensors"
    ),
)

REQUIRED_MODELS = (SWINV2_LARGE, OPENCLIP_VIT_B_32)


class MissingModelError(FileNotFoundError):
    pass


def require_model(asset: ModelAsset) -> Path:
    path = asset.path
    if path.is_file():
        return path
    raise MissingModelError(
        f"{asset.display_name} is missing.\n\n"
        f"Expected file:\n{path}\n\n"
        f"Download it from:\n{asset.url}\n\n"
        "Set HYPERGALLERY_MODEL_DIR to use a different model directory."
    )
