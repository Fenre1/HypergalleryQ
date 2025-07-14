from __future__ import annotations

from typing import Dict, List, Iterable

from PyQt5.QtWidgets import (
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt

from .session_model import SessionModel
from .selection_bus import SelectionBus
from .image_grid import ImageGridDock


class OverlapListDock(QDockWidget):
    """Dock widget listing hyperedges overlapping with the current grid."""

    THUMB_SIZE = 32

    def __init__(self, bus: SelectionBus, grid: ImageGridDock, parent=None):
        super().__init__("Related Hyperedges", parent)
        self.bus = bus
        self.grid = grid
        self.session: SessionModel | None = None
        self._last_indices: set[int] = set()

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_double_clicked)
        self.setWidget(self.list_widget)

        self.bus.imagesChanged.connect(self._on_images_changed)

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        self.list_widget.clear()

    def _on_images_changed(self, idxs: List[int]):
        if self.session is None:
            self.list_widget.clear()
            return
        self._update_list(idxs)

    # ------------------------------------------------------------------
    def _update_list(self, idxs: Iterable[int]):
        self.list_widget.clear()
        indices = set(idxs)
        if not indices:
            self._last_indices = set()
            return

        edge_counts: Dict[str, int] = {}
        for idx in indices:
            for edge in self.session.image_mapping.get(idx, set()):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        if not edge_counts:
            self._last_indices = indices
            return

        triplets = self.session.compute_overview_triplets()
        for name, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, name)
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(2, 2, 2, 2)
            layout.addWidget(QLabel(f"{name} ({count})"))
            imgs = triplets.get(name, ())
            for idx in imgs[:6]:
                lbl = QLabel()
                lbl.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
                if idx is not None:
                    pix = QPixmap(self.session.im_list[idx])
                    if not pix.isNull():
                        pix = pix.scaled(
                            self.THUMB_SIZE,
                            self.THUMB_SIZE,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                        lbl.setPixmap(pix)
                layout.addWidget(lbl)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        self._last_indices = indices

    # ------------------------------------------------------------------
    def _on_double_clicked(self, item: QListWidgetItem):
        name = item.data(Qt.UserRole)
        if not name or self.session is None:
            return
        overlap = set(self.session.hyperedges.get(name, set())) & self._last_indices
        highlight = {i: QColor(221, 160, 221) for i in overlap}  # plum / light violet
        self.grid._highlight_next = highlight
        self.bus.set_edges([name])