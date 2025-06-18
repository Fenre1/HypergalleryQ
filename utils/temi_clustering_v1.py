import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TEMIClusterer:
    """Minimal TEMI-style clustering with EMA teacher.

    Parameters
    ----------
    k : int
        Number of clusters.
    threshold : float, optional
        Probability threshold for hyperedge assignment. Defaults to 0.5.
    epochs : int, optional
        Number of training epochs. Defaults to 100.
    batch_size : int, optional
        Mini-batch size. Defaults to 256.
    lr : float, optional
        Learning rate. Defaults to 1e-3.
    momentum : float, optional
        Exponential moving average factor for the teacher. Defaults to 0.99.
    device : str or torch.device, optional
        Device to run the optimisation on.
    """

    def __init__(self, k: int, threshold: float = 0.5, *, epochs: int = 100,
                 batch_size: int = 256, lr: float = 1e-4, momentum: float = 0.996,
                 device: Optional[str] = None) -> None:
        self.k = k
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.student: Optional[nn.Linear] = None
        self.teacher: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    def fit(self, features: torch.Tensor) -> "TEMIClusterer":
        """Train the clusterer on the provided features."""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)
        dim = features.shape[1]

        # simple linear heads for student/teacher
        self.student = nn.Linear(dim, self.k, bias=True).to(self.device)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        optimiser = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        dataset = DataLoader(TensorDataset(features), batch_size=self.batch_size,
                             shuffle=True)

        for _ in range(self.epochs):
            for (x,) in dataset:
                x = x.to(self.device)
                with torch.no_grad():
                    teacher_logits = self.teacher(x)
                    teacher_probs = F.softmax(teacher_logits / 0.1, dim=1)

                student_logits = self.student(x)
                student_log_probs = F.log_softmax(student_logits / 0.1, dim=1)
                loss = -(teacher_probs * student_log_probs).sum(dim=1).mean()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # EMA update for teacher
                with torch.no_grad():
                    for t, s in zip(self.teacher.parameters(),
                                     self.student.parameters()):
                        t.data.mul_(self.momentum).add_(s.data, alpha=1 - self.momentum)

        return self

    # ------------------------------------------------------------------
    def _predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)
        with torch.no_grad():
            logits = self.teacher(features)
            probs = F.softmax(logits, dim=1)
        return probs.cpu()

    # ------------------------------------------------------------------
    def labels(self, features) -> torch.Tensor:
        """Return argmax cluster assignment."""
        probs = self._predict_proba(features)
        return probs.argmax(dim=1)

    # ------------------------------------------------------------------
    def hyperedges(self, features) -> torch.Tensor:
        """Return binary hyperedge matrix."""
        probs = self._predict_proba(features)
        edges = (probs >= self.threshold).int()
        max_idx = probs.argmax(dim=1)
        for i in range(edges.size(0)):
            if edges[i].sum() == 0:
                edges[i, max_idx[i]] = 1
        return edges