import numpy as np
import torch
import timm
from PIL import Image
import timm.data
from torch.utils.data import Dataset, DataLoader
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
