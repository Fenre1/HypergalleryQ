from __future__ import annotations

from typing import Dict, List

import numpy as np
import networkx as nx

try:
    from pyforceatlas2 import ForceAtlas2
except Exception as exc:  # pragma: no cover - library might not be installed
    ForceAtlas2 = None

from .session_model import SessionModel

class HyperedgeForceAtlas2:
    """Compute a ForceAtlas2 layout for hyperedges.

    Parameters
    ----------
    overlap_data:
        Mapping of hyperedge names to a mapping of overlap counts with all
        other hyperedges. This has the same structure as the hyperedge
        matrix displayed in :mod:`hyperedge_matrix`.
    session:
        :class:`SessionModel` providing access to hyperedge membership.
    """

    def __init__(self, overlap_data: Dict[str, Dict[str, int]], session: SessionModel) -> None:
        if ForceAtlas2 is None:
            raise ImportError("pyforceatlas2 is required to compute the layout")
        self.overlap_data = overlap_data
        self.session = session
        self.names: List[str] = list(overlap_data)
        self.node_sizes = self._compute_node_sizes()
        self.graph = self._build_graph()
        self.fa2 = ForceAtlas2()
        # Initialize random positions for step-by-step updates
        self.positions: Dict[str, np.ndarray] = {
            name: np.random.randn(2) for name in self.names
        }

    # ------------------------------------------------------------------
    def _compute_node_sizes(self) -> np.ndarray:
        sizes = [len(self.session.hyperedges.get(name, [])) for name in self.names]
        # return np.sqrt(np.asarray(sizes, dtype=float))
        return np.asarray(sizes, dtype=float)

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for name, size in zip(self.names, self.node_sizes):
            G.add_node(name, size=float(size))

        for i, rk in enumerate(self.names):
            for j, ck in enumerate(self.names):
                if j <= i:
                    continue
                overlap = self.overlap_data[rk][ck]
                len_rk = len(self.session.hyperedges[rk])
                len_ck = len(self.session.hyperedges[ck])
                p1 = overlap / len_rk if len_rk else 0.0
                p2 = overlap / len_ck if len_ck else 0.0
                score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0.0
                if score > 0:
                    G.add_edge(rk, ck, weight=float(score))
        return G

    # ------------------------------------------------------------------
    def step(self, iterations: int = 1) -> Dict[str, np.ndarray]:
        """Advance the layout by ``iterations`` steps."""
        self.positions = self.fa2.forceatlas2_networkx_layout(
            self.graph,
            pos=self.positions,
            iterations=iterations,
        )
        return self.positions

    def compute_layout(self, iterations: int = 500) -> Dict[str, np.ndarray]:
        """Compute the layout in ``iterations`` steps and return positions."""
        self.step(iterations)
        return self.positions