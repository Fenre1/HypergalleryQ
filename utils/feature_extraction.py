from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
import timm.data
from torch.utils.data import Dataset, DataLoader
import open_clip
import urllib.request
from typing import List
import re



MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(MODEL_DIR))

#'swinv2_large_window12to24_192to384'
class ImageFileDataset(Dataset):
    """ 
    A PyTorch Dataset for loading and transforming images given a list of file paths.
    """
    def __init__(self, file_list, transform):
        """
        Args:
            file_list (List[str]): List of image file paths.
            transform: A transformation function to apply to each image.
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        # Load the image and convert to RGB
        image = Image.open(file_path).convert('RGB')
        # Apply the provided transformation
        image = self.transform(image)
        return image

class FeatureExtractor:
    """
    Extracts features from images using a timm model.
    """
    def __init__(self, model_name: str, batch_size: int = 32):
        """
        Initialize the feature extractor.

        Args:
            model_name (str): The name of the model to be used from timm.
            batch_size (int, optional): The batch size for processing images.
                                        Default is 32.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.batch_size = batch_size

        # Load the model from timm; using num_classes=0 removes the final classification layer.
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        # Set up transformation based on the model's default configuration.
        self.transform = timm.data.transforms_factory.create_transform(
            input_size=self.model.default_cfg['input_size']
        )




    def extract_features(self, file_list: list) -> np.ndarray:
        """
        Extract features for the given list of image file paths.

        Args:
            file_list (list): List of image file paths.

        Returns:
            np.ndarray: Features extracted from all images.
        """
        # Create a dataset instance and a DataLoader to handle batching.
        dataset = ImageFileDataset(file_list, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Adjust num_workers based on your system
        )
 


        all_features = []
        for images in loader:
            images = images.to(self.device)
            with torch.no_grad():
                features = self.model(images)
            # Append the features (moved to CPU and converted to numpy)
            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)

class Swinv2LargeFeatureExtractor(FeatureExtractor):
    """Convenience wrapper for the SwinV2 large model."""

    def __init__(self, batch_size: int = 32):
        super().__init__("swinv2_large_window12to24_192to384", batch_size)

class OpenClipFeatureExtractor:
    """Extract features using an OpenCLIP model."""

    def __init__(self, model_name: str = "ViT-B-32", *, pretrained: str = "laion2b_s34b_b79k", batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

    @property
    def transform(self):
        return self.preprocess

    def extract_features(self, file_list: list) -> np.ndarray:
        dataset = ImageFileDataset(file_list, self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        all_features = []
        for images in loader:
            images = images.to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(images)
            all_features.append(feats.cpu().numpy())
        return np.vstack(all_features)
    
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Return OpenCLIP embeddings for the given text strings."""
        tokens = open_clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.cpu().numpy()
    

_PLACES_URL = (
    "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar"
)

def _download_places365(checkpoint_dir: str | Path = MODEL_DIR / "places365") -> Path:
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "densenet161_places365.pth.tar"
    if not ckpt_path.exists():
        print("Downloading DenseNet_161 Places365 weights …")
        urllib.request.urlretrieve(_PLACES_URL, ckpt_path)
    return ckpt_path




_LEGACY_PATTERN = re.compile(r"\.(norm|relu|conv)\.(\d+)")     # ".norm.1" → ".norm1"

def _remap_legacy_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert PyTorch‑0.2 DenseNet key style to modern style."""
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")                           # DataParallel prefix
        k = _LEGACY_PATTERN.sub(lambda m: f".{m.group(1)}{m.group(2)}", k)
        new_state[k] = v
    return new_state

class DenseNet161Places365FeatureExtractor:
    """Return 2208‑D global‑pooled features of DenseNet‑161 Places365."""

    def __init__(
        self,
        batch_size: int = 32,
        checkpoint_path: str | Path | None = None,
        *,
        num_workers: int = 0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size, self.num_workers = batch_size, num_workers

        # 1. Architecture **without** classifier
        self.model = timm.create_model("densenet161", pretrained=False, num_classes=0)
        self.model.to(self.device).eval()

        # 2. Load & remap legacy weights
        ckpt_path = (
            Path(checkpoint_path) if checkpoint_path else _download_places365()
        )
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", encoding="latin1")
        state = raw_ckpt.get("state_dict", raw_ckpt)
        state = _remap_legacy_keys(state)

        # Drop the original classifier (fully‑connected) layer
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected}"     # must be empty
        # 'missing' only contains 'classifier.weight/bias' – expected

        # 3. Default Places365 transform (256 → centre‑crop 224)
        self.transform = timm.data.transforms_factory.create_transform(
            input_size=(3, 224, 224), interpolation="bicubic", crop_pct=224 / 256
        )

    # ------------------------------------------------------------------
    # 4.  Public API (same signature as your other extractors)
    # ------------------------------------------------------------------
    def extract_features(self, file_list: List[str]) -> np.ndarray:
        loader = DataLoader(
            ImageFileDataset(file_list, self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        feats = []
        with torch.no_grad():
            for imgs in loader:
                feats.append(self.model(imgs.to(self.device)).cpu().numpy())
        return np.vstack(feats)

