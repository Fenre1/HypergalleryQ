import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional


def _sim_weight(p1: torch.Tensor, p2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Similarity weight used by TEMI."""
    return (p1 * p2).pow(gamma).sum(dim=-1)


def _beta_mi(p1: torch.Tensor, p2: torch.Tensor, pk: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    beta_emi = (((p1 * p2) ** beta) / pk).sum(dim=-1)
    return -beta_emi.log()


class _FeaturePairDataset(Dataset):
    """Dataset yielding pairs of features using k-NN indices."""

    def __init__(self, features: torch.Tensor, knn_idx: torch.Tensor):
        self.features = features
        self.knn_idx = knn_idx

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        neigh_list = self.knn_idx[idx]
        if isinstance(neigh_list, torch.Tensor):
            neigh_list = neigh_list.tolist()
        pair_idx = neigh_list[torch.randint(0, len(neigh_list), (1,)).item()]
        return self.features[idx], self.features[pair_idx]


class TemiClustering:
    """Minimal implementation of TEMI clustering for feature embeddings."""

    def __init__(
        self,
        k: int,
        epochs: int = 100,
        lr: float = 1e-4,
        batch_size: int = 256,
        k_neighbors: int = 50,
        teacher_momentum: float = 0.996,
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.k_neighbors = k_neighbors
        self.momentum = teacher_momentum
        self.threshold = threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student: Optional[nn.Linear] = None
        self.teacher: Optional[nn.Linear] = None

    @staticmethod
    def _compute_knn(features: torch.Tensor, k: int) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        dists = features @ features.t()
        dists.fill_diagonal_(-float("inf"))
        knn = dists.topk(k, dim=-1).indices
        return knn

    def _build_heads(self, dim: int) -> None:
        self.student = nn.Linear(dim, self.k, bias=False).to(self.device)
        self.teacher = nn.Linear(dim, self.k, bias=False).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def fit(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.device)
        knn_idx = self._compute_knn(features, self.k_neighbors)
        dataset = _FeaturePairDataset(features, knn_idx)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self._build_heads(features.size(1))
        optim = torch.optim.AdamW(self.student.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            for anchor, neighbor in loader:
                anchor = anchor.to(self.device)
                neighbor = neighbor.to(self.device)
                s1 = self.student(anchor)
                s2 = self.student(neighbor)
                with torch.no_grad():
                    t1 = self.teacher(anchor)
                    t2 = self.teacher(neighbor)
                ps1 = F.softmax(s1 / 0.1, dim=-1)
                ps2 = F.softmax(s2 / 0.1, dim=-1)
                pt1 = F.softmax(t1 / 0.04, dim=-1)
                pt2 = F.softmax(t2 / 0.04, dim=-1)
                pk = pt1.mean(dim=0)
                weight = _sim_weight(pt1, pt2)
                loss = (_beta_mi(ps1, pt2, pk) * weight + _beta_mi(ps2, pt1, pk) * weight).mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

                with torch.no_grad():
                    for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
                        tp.data.mul_(self.momentum).add_((1 - self.momentum) * sp.data)

        return self.predict(features)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(self.device)
        with torch.no_grad():
            logits = self.teacher(features)
            probs = F.softmax(logits, dim=-1)
        if self.threshold is None:
            assign = probs.argmax(dim=1)
        else:
            assign = (probs >= self.threshold).int()
            max_idx = probs.argmax(dim=1)
            for i in range(assign.size(0)):
                if assign[i].sum() == 0:
                    assign[i, max_idx[i]] = 1
        return assign.cpu()