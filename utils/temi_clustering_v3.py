import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional


@torch.no_grad()
def compute_neighbors(embedding: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k nearest neighbors using cosine similarity."""
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    dists = embedding @ embedding.t()
    dists.fill_diagonal_(-float('inf'))
    return dists.topk(k, dim=-1).indices


class PairDataset(Dataset):
    """Dataset yielding a feature and one of its neighbors."""

    def __init__(self, features: torch.Tensor, neighbors: torch.Tensor):
        self.features = features
        self.neighbors = neighbors

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int):
        neigh_idx = np.random.choice(self.neighbors[index].cpu().numpy())
        return self.features[index], self.features[neigh_idx]


class SimpleHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TEMILoss(nn.Module):
    """Simplified TEMI loss for two views."""

    def __init__(self, out_dim: int, beta: float = 0.6, momentum: float = 0.9,
                 student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.beta = beta
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.momentum = momentum
        self.register_buffer("pk", torch.ones(1, out_dim) / out_dim)

    @staticmethod
    def sim_weight(p1: torch.Tensor, p2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        return (p1 * p2).pow(gamma).sum(dim=-1)

    @staticmethod
    def beta_mi(p1: torch.Tensor, p2: torch.Tensor, pk: torch.Tensor, beta: float,
                 clip_min: float = -float('inf')) -> torch.Tensor:
        beta_emi = (((p1 * p2) ** beta) / pk).sum(dim=-1)
        beta_pmi = beta_emi.log().clamp(min=clip_min)
        return -beta_pmi

    def forward(self, s1: torch.Tensor, s2: torch.Tensor,
                t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        sp1 = torch.softmax(s1 / self.student_temp, dim=-1)
        sp2 = torch.softmax(s2 / self.student_temp, dim=-1)
        tp1 = torch.softmax(t1 / self.teacher_temp, dim=-1)
        tp2 = torch.softmax(t2 / self.teacher_temp, dim=-1)

        with torch.no_grad():
            batch_pk = torch.cat([tp1, tp2]).mean(dim=0, keepdim=True)
            self.pk = self.pk * self.momentum + batch_pk * (1 - self.momentum)

        weight = self.sim_weight(tp1, tp2)
        loss1 = self.beta_mi(sp1, tp2, self.pk, self.beta)
        loss2 = self.beta_mi(sp2, tp1, self.pk, self.beta)
        return (weight * (loss1 + loss2) / 2).mean()


class TEMIHypergraphClusterer:
    """Cluster features using a small TEMI model."""

    def __init__(self, n_clusters: int, *, epochs: int = 100, batch_size: int = 256,
                 lr: float = 1e-3, knn: int = 50, threshold: float = 0.5,
                 device: Optional[torch.device] = None):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.knn = knn
        self.threshold = threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student: Optional[SimpleHead] = None
        self.teacher: Optional[SimpleHead] = None
        self.loss_fn: Optional[TEMILoss] = None

    def fit(self, features: np.ndarray) -> None:
        feats = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        neighbors = compute_neighbors(feats, self.knn)
        dataset = PairDataset(feats, neighbors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.student = SimpleHead(feats.size(1), self.n_clusters).to(self.device)
        self.teacher = SimpleHead(feats.size(1), self.n_clusters).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        self.loss_fn = TEMILoss(self.n_clusters).to(self.device)
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr)

        momentum = 0.996
        for _ in range(self.epochs):
            for x1, x2 in loader:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                v1 = x1 + torch.randn_like(x1) * 0.01
                v2 = x2 + torch.randn_like(x2) * 0.01

                t1 = self.teacher(v1)
                t2 = self.teacher(v2)
                s1 = self.student(v1)
                s2 = self.student(v2)

                loss = self.loss_fn(s1, s2, t1, t2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
                        p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.teacher is None:
            raise RuntimeError("Model has not been fitted")
        feats = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        logits = self.teacher(feats)
        probs = torch.softmax(logits / 0.04, dim=-1)
        matrix = (probs >= self.threshold).cpu().numpy().astype(np.int32)
        max_idx = probs.argmax(dim=1).cpu().numpy()
        for i, row in enumerate(matrix):
            if not row.any():
                row[max_idx[i]] = 1
        return matrix

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.predict(features)