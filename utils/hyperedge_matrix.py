from __future__ import annotations

from typing import List
from functools import lru_cache

from PySide6.QtWidgets import QDockWidget, QTableWidget, QTableWidgetItem
from PySide6.QtGui import QIcon, QPixmap, QColor
from PySide6.QtCore import Qt, QSize

from .selection_bus import SelectionBus

from .session_model import SessionModel


class HyperedgeMatrixDock(QDockWidget):
    """Dock widget showing overlap between hyperedges."""

    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 64):
        super().__init__("Hyperedge Overlap", parent)
        self.bus = bus
        self.thumb_size = thumb_size
        self.session: SessionModel | None = None
        self.table = QTableWidget()
        self.table.verticalHeader().setDefaultSectionSize(self.thumb_size)
        self.table.horizontalHeader().setDefaultSectionSize(self.thumb_size)
        self.setWidget(self.table)
        self.table.cellClicked.connect(self._on_cell_clicked)

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        # Clear cached thumbnails so icons match the currently loaded session
        self._load_thumb.cache_clear()
        self.update_matrix()

    # ------------------------------------------------------------------
    def update_matrix(self):
        if not self.session:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self._data = {}
            return

        edges = list(self.session.hyperedges.keys())
        self.table.setRowCount(len(edges))
        self.table.setColumnCount(len(edges))

        # Set headers with thumbnails
        for i, name in enumerate(edges):
            icon = QIcon(self._load_thumb(name))
            h_item = QTableWidgetItem()
            h_item.setIcon(icon)
            h_item.setToolTip(name)
            v_item = QTableWidgetItem()
            v_item.setIcon(icon)
            v_item.setToolTip(name)
            self.table.setHorizontalHeaderItem(i, h_item)
            self.table.setVerticalHeaderItem(i, v_item)

        # Fill matrix and store overlap counts
        self._data: dict[str, dict[str, int]] = {}
        for r, r_name in enumerate(edges):
            self._data[r_name] = {}
            r_imgs = self.session.hyperedges[r_name]
            for c, c_name in enumerate(edges):
                c_imgs = self.session.hyperedges[c_name]
                overlap = len(r_imgs & c_imgs)
                item = QTableWidgetItem(str(overlap))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
                self._data[r_name][c_name] = overlap

        self.apply_heatmap_coloring()

    # ------------------------------------------------------------------
    def _on_cell_clicked(self, row: int, col: int):
        """Send the intersection of the two selected hyperedges via the bus."""
        if not self.session:
            return
        edges = list(self.session.hyperedges.keys())
        if not (0 <= row < len(edges) and 0 <= col < len(edges)):
            return
        r_name = edges[row]
        c_name = edges[col]
        idxs = sorted(self.session.hyperedges[r_name] &
                      self.session.hyperedges[c_name])
        self.bus.set_images(idxs)

    # ------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def _load_thumb(self, edge_name: str) -> QPixmap:
        idxs = sorted(self.session.hyperedges.get(edge_name, [])) if self.session else []
        if not idxs:
            return QPixmap()
        path = self.session.im_list[idxs[0]]
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    # ------------------------------------------------------------------
    def apply_heatmap_coloring(self):
        if not self.session:
            return
        hyperedge_keys = list(self._data)
        max_score = 0.0

        # First pass: compute max score excluding diagonal
        for i, rk in enumerate(hyperedge_keys):
            for j, ck in enumerate(hyperedge_keys):
                if i == j:
                    continue
                overlap = self._data[rk][ck]
            for j, ck in enumerate(hyperedge_keys):
                item = self.table.item(i, j)
                if item is None:
                    continue
                if i == j:
                    item.setBackground(QColor('#5555FF'))
                else:
                    overlap = self._data[rk][ck]
                    p1 = overlap / len(self.session.hyperedges[rk]) if self.session.hyperedges[rk] else 0
                    p2 = overlap / len(self.session.hyperedges[ck]) if self.session.hyperedges[ck] else 0
                    score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0
                    color = QColor(self.get_heatmap_color(score, max_score))
                    item.setBackground(color)

    # ------------------------------------------------------------------
    @staticmethod
    def get_heatmap_color(score: float, max_score: float) -> str:
        if max_score == 0:
            return '#555555'
        start_color = (0x55, 0x55, 0x55)
        end_color = (0xFF, 0x55, 0x55)
        t = max(0.0, min(score, max_score)) / max_score
        r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        return f'#{r:02x}{g:02x}{b:02x}'