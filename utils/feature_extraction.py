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
    "http://places2.csail.mit.edu/models_places365/resnet152_places365.pth.tar"
)

def _download_places365(checkpoint_dir: str | Path = MODEL_DIR / "places365") -> Path:
    """Download the official PyTorch checkpoint if it is not present."""
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / "resnet152_places365.pth.tar"
    if not ckpt_path.exists():
        print("Downloading ResNet‑152 Places365 weights …")
        urllib.request.urlretrieve(_PLACES_URL, ckpt_path)
    return ckpt_path


class ResNet152Places365FeatureExtractor:

    def __init__(self, batch_size: int = 32, checkpoint_path: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # 1. Build an **architecture‑only** ResNet‑152 *without* the FC layer
        self.model = timm.create_model("resnet152", pretrained=False, num_classes=0)
        self.model.to(self.device)

        # 2. Load Places365 weights
        ckpt_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else _download_places365()
        )
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", encoding="latin1"  # <- legacy pickle
        )
        state = checkpoint.get("state_dict", checkpoint)
        # Strip `module.` that existed when DataParallel was standard
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # Drop the original FC weights because we removed the FC layer
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected  # should be empty
        self.model.eval()

        # 3. Use the transform that MIT’s demo uses (Resize 256 → CenterCrop 224)
        #    You can swap this for timm’s convenience factory if you like.
        self.transform = timm.data.transforms_factory.create_transform(
            input_size=(3, 224, 224),  # 224×224 crop
            interpolation="bicubic",
            crop_pct=224 / 256,
        )

    # -------- identical public API to your other extractors --------
    def extract_features(self, file_list: list[str]) -> np.ndarray:
        dataset = ImageFileDataset(file_list, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        feats = []
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                feats.append(self.model(x).cpu().numpy())
        return np.vstack(feats)
