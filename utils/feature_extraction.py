from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import open_clip
import timm
import timm.data
import torch
from PIL import Image, ImageFile
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset

from .model_registry import OPENCLIP_VIT_B_32, SWINV2_LARGE, require_model
from .runtime_paths import model_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = model_dir()
MODEL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR))


def _load_swinv2_checkpoint(model, checkpoint_path) -> None:
    state_dict = load_safetensors(str(checkpoint_path), device="cpu")
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith(".attn_mask")
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.endswith(".attn_mask")
    ]
    if unexpected:
        print(f"Warning: unexpected keys loading {checkpoint_path}: {unexpected}")
    if incompatible.missing_keys:
        print(f"Warning: missing keys loading {checkpoint_path}: {incompatible.missing_keys}")


class ImageFileDataset(Dataset):
    """Load image files and apply a model transform."""

    def __init__(self, file_list, transform, image_size=None):
        self.file_list = file_list
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to load {file_path}: {e}")
            if self.image_size:
                image = Image.new("RGB", self.image_size)
            else:
                raise
        return self.transform(image)


class FeatureExtractor:
    """Extract features from images using a timm model."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size

        checkpoint_path = require_model(SWINV2_LARGE)
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=1000,
        )
        _load_swinv2_checkpoint(self.model, checkpoint_path)
        if hasattr(self.model, "reset_classifier"):
            self.model.reset_classifier(0)
        self.model.to(self.device)
        self.model.eval()

        self.transform = timm.data.transforms_factory.create_transform(
            input_size=self.model.default_cfg["input_size"]
        )

        _, h, w = self.model.default_cfg["input_size"]
        self.image_size = (w, h)

    def extract_features(self, file_list: list, progress_callback=None) -> np.ndarray:
        dataset = ImageFileDataset(file_list, self.transform, self.image_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_features = []
        processed = 0
        total = len(file_list)
        for images in loader:
            images = images.to(self.device)
            with torch.no_grad():
                features = self.model(images)
            all_features.append(features.cpu().numpy())
            processed += images.size(0)
            if progress_callback:
                progress_callback(processed, total)

        return np.vstack(all_features)


class Swinv2LargeFeatureExtractor(FeatureExtractor):
    def __init__(self, batch_size: int = 32):
        super().__init__("swinv2_large_window12to24_192to384.ms_in22k_ft_in1k", batch_size)


class OpenClipFeatureExtractor:
    """Extract image and text features using an OpenCLIP model."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        *,
        pretrained: str = "laion2b_s34b_b79k",
        batch_size: int = 32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        checkpoint_path = require_model(OPENCLIP_VIT_B_32)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(checkpoint_path)
        )
        self.model.to(self.device)
        self.model.eval()
        self.image_size = (224, 224)

    @property
    def transform(self):
        return self.preprocess

    def extract_features(self, file_list: list, progress_callback=None) -> np.ndarray:
        dataset = ImageFileDataset(file_list, self.preprocess, self.image_size)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        all_features = []
        processed = 0
        total = len(file_list)
        for images in loader:
            images = images.to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(images)
            all_features.append(feats.cpu().numpy())
            processed += images.size(0)
            if progress_callback:
                progress_callback(processed, total)
        return np.vstack(all_features)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = open_clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.cpu().numpy()
