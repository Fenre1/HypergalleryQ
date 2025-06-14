import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import model_builders  # Your module for loading embeddings
from load_embeds import load_embeds

class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, dataset, datapath='./data/embeddings', train=True, arch=None):
        super().__init__()
        # Define the dataset path, e.g. "./data/embeddings/mydataset"
        dataset_path = Path(datapath) / 'embeddings' /dataset
        # Load the embeddings and labels using your custom loader.
        self.emb, self.targets = load_embeds(dataset_path)
    
    def __getitem__(self, index):
        return self.emb[index], self.targets[index]
    
    def __len__(self):
        return len(self.emb)

class EmbedNN(Dataset):
    def __init__(self,
                 dataset,
                 knn_filename='knn.pt',
                 transform=None,
                 k=10,
                 datapath='./data/',
                 precompute_arch=None):
        super().__init__()
        self.transform = transform

        dataset_path = Path(datapath) / 'embeddings' / dataset
        self.knn_path = dataset_path / knn_filename

        self.complete_neighbors = torch.load(self.knn_path)
        if k < 0:
            k = self.complete_neighbors.size(1)
        self.k = k
        self.neighbors = self.complete_neighbors[:, :k]
        
        # Load the dataset using the precomputed embeddings.
        self.dataset = PrecomputedEmbeddingDataset(
            dataset=dataset,
            datapath=datapath,
            train=True,
            arch=precompute_arch)
        
    def get_transformed_imgs(self, idx, *idcs):
        img, label = self.dataset[idx]
        # Retrieve neighbor images (assumes each neighbor index is valid).
        rest_imgs = (self.dataset[i][0] for i in idcs)
        return self.transform(img, *rest_imgs), label

    def __getitem__(self, idx):
        # Choose a random neighbor from the precomputed list.
        neighbor_indices = self.neighbors[idx]
        # Convert to a Python list if necessary.
        if isinstance(neighbor_indices, torch.Tensor):
            neighbor_indices = neighbor_indices.tolist()
        pair_idx = random.choice(neighbor_indices)
        
        # If a transform function is provided, use it to process the images.
        if self.transform:
            return self.get_transformed_imgs(idx, pair_idx)
        else:
            # Otherwise, just return the raw anchor and neighbor images along with the label.
            anchor, label = self.dataset[idx]
            neighbor, _ = self.dataset[pair_idx]
            return (anchor, neighbor), label

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset, datapath='./data', train=True, transform=None, download=True, precompute_arch=None):
    return PrecomputedEmbeddingDataset(
        dataset=dataset,
        arch=precompute_arch,
        datapath='./data/', # assumes embeddings are saved in the ./data folder
        train=train)
    
