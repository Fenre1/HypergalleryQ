# coding: utf-8
"""Minimal TEMI clustering using in-memory features.

This module implements a small wrapper around the TEMI weighted mutual
information objective.  It only requires the features in memory and
produces a clustering matrix that mirrors the ``clustering_results``
layout used in the session files.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Utility functions taken from the original implementation
# ---------------------------------------------------------------------------

def compute_neighbors(embeds: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the k nearest neighbors for each embedding."""
    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
    num = embeds.shape[0]
    dists = embeds @ embeds.t()
    dists.fill_diagonal_(-torch.inf)
    _, nn_idx = dists.topk(k, dim=-1)
    return nn_idx


def sim_weight(p1: torch.Tensor, p2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return (p1 * p2).pow(gamma).sum(dim=-1)


def beta_mi(p1: torch.Tensor, p2: torch.Tensor, pk: torch.Tensor,
             beta: float = 1.0, clip_min: float = -torch.inf) -> torch.Tensor:
    beta_emi = (((p1 * p2) ** beta) / pk).sum(dim=-1)
    beta_pmi = beta_emi.log().clamp(min=clip_min)
    return -beta_pmi


class TEMILoss(nn.Module):
    """Single-head TEMI objective."""

    def __init__(self, out_dim: int, batchsize: int = 256, epochs: int = 100,
                 beta: float = 0.6, student_temp: float = 0.1,
                 teacher_temp: float = 0.04, probs_momentum: float = 0.996):
        super().__init__()
        self.out_dim = out_dim
        self.batchsize = batchsize
        self.beta = beta
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.probs_momentum = probs_momentum
        self.register_buffer("pk", torch.ones(1, out_dim) / out_dim)
        self.epochs = epochs

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor, epoch: int) -> torch.Tensor:
        """Compute TEMI loss for a batch."""
        n_views = len(student_out) // self.batchsize
        s_out = student_out.chunk(n_views)
        t_probs = torch.softmax(teacher_out / self.teacher_temp, dim=-1).detach().chunk(n_views)

        with torch.no_grad():
            batch_center = torch.cat(t_probs).sum(dim=0, keepdim=True) / (len(t_probs) * self.batchsize)
            self.pk.mul_(self.probs_momentum).add_(batch_center * (1 - self.probs_momentum))
            pk = self.pk

        weight = 0.0
        pairs = 0
        for i in range(n_views):
            for j in range(i + 1, n_views):
                weight += sim_weight(t_probs[i], t_probs[j])
                pairs += 1
        weight = weight / max(pairs, 1)

        total_loss = 0.0
        count = 0
        for w in range(n_views):
            for v in range(n_views):
                if v == w:
                    continue
                loss = weight * beta_mi(torch.softmax(s_out[v], dim=-1), t_probs[w], pk, beta=self.beta)
                total_loss += loss.mean()
                count += 1
        return total_loss / count


# ---------------------------------------------------------------------------
# Simple network definitions
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class StudentTeacher(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.student = ProjectionHead(in_dim, out_dim)
        self.teacher = ProjectionHead(in_dim, out_dim)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, views):
        # views is list/tuple of tensors
        cat = torch.cat(views)
        t_out = self.teacher(cat)
        s_out = self.student(cat)
        return s_out, t_out


# ---------------------------------------------------------------------------
# Dataset working purely in memory
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, embeds: torch.Tensor, neighbors: torch.Tensor):
        self.embeds = embeds
        self.neighbors = neighbors

    def __len__(self) -> int:
        return self.embeds.shape[0]

    def __getitem__(self, idx: int):
        pair_idx = np.random.choice(self.neighbors[idx].cpu().numpy())
        return self.embeds[idx], self.embeds[pair_idx]


# ---------------------------------------------------------------------------
# Main clustering class
# ---------------------------------------------------------------------------

class TEMIClusterer:
    def __init__(self, n_clusters: int, epochs: int = 100,
                 k_nn: int = 50, threshold: float | None = None,
                 batch_size: int = 256, lr: float = 1e-4):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.k_nn = k_nn
        self.threshold = threshold
        self.batch_size = batch_size
        self.lr = lr
        self.model: StudentTeacher | None = None
        self.loss_fn: TEMILoss | None = None

    def fit(self, features: np.ndarray | torch.Tensor) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feats = torch.as_tensor(features, dtype=torch.float32)
        neighbors = compute_neighbors(feats, self.k_nn)
        dataset = EmbeddingDataset(feats, neighbors)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = StudentTeacher(feats.shape[1], self.n_clusters).to(device)
        self.loss_fn = TEMILoss(self.n_clusters, self.batch_size, self.epochs).to(device)
        optimizer = torch.optim.Adam(self.model.student.parameters(), lr=self.lr)
        momentum = 0.996
        for epoch in range(self.epochs):
            for x1, x2 in loader:
                v1 = x1.to(device)
                v2 = x2.to(device)
                student_out, teacher_out = self.model([v1, v2])
                loss = self.loss_fn(student_out, teacher_out, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for p_s, p_t in zip(self.model.student.parameters(), self.model.teacher.parameters()):
                        p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)

    def predict(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet")
        feats = torch.as_tensor(features, dtype=torch.float32).cuda()
        with torch.no_grad():
            logits = self.model.teacher(feats).cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        if self.threshold is None:
            out = np.zeros_like(probs, dtype=np.int64)
            out[np.arange(len(probs)), probs.argmax(axis=1)] = 1
            return out
        else:
            out = (probs >= self.threshold).astype(np.int64)
            empty = out.sum(axis=1) == 0
            if np.any(empty):
                argmax = probs[empty].argmax(axis=1)
                out[empty, :] = 0
                out[np.where(empty)[0], argmax] = 1
            return out

    def fit_predict(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        self.fit(features)
        return self.predict(features)