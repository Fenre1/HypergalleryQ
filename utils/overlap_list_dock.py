from __future__ import annotations

from typing import Dict, List, Iterable

from PyQt5.QtWidgets import (
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QStyle,
)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFontMetrics, QPalette, QPen
from PyQt5.QtCore import Qt, QModelIndex, QSize
from functools import lru_cache
import time
from .session_model import SessionModel
from .selection_bus import SelectionBus
from .image_grid import ImageGridDock


NAME_ROLE = Qt.UserRole
IMAGES_ROLE = Qt.UserRole + 1
COUNT_ROLE = Qt.UserRole + 2


@lru_cache(maxsize=1024)
def _scaled_thumb(path: str, size: int) -> QPixmap:
    pix = QPixmap(path)
    if pix.isNull():
        return QPixmap()
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _OverlapDelegate(QStyledItemDelegate):
    """Paint hyperedge name with small thumbnails."""

    def __init__(self, session: SessionModel | None, thumb: int, parent=None):
        super().__init__(parent)
        self._session = session
        self._thumb = thumb
        self._margin = 2

    def set_session(self, session: SessionModel | None) -> None:
        self._session = session

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # type: ignore[override]
        painter.save()

        # background & selection
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())

        name = index.data(NAME_ROLE)
        count = index.data(COUNT_ROLE)
        imgs = index.data(IMAGES_ROLE)
        if imgs is None and self._session:
            imgs = self._session.overview_triplet_for(name)
            index.model().setData(index, imgs, IMAGES_ROLE)

        fm: QFontMetrics = option.fontMetrics
        color = option.palette.highlightedText().color() if option.state & QStyle.State_Selected else option.palette.text().color()

        x = option.rect.left() + self._margin
        y = option.rect.top() + (option.rect.height() + fm.ascent() - fm.descent()) // 2

        text = f"{name} ({count})"
        painter.setPen(color)
        painter.drawText(x, y, text)
        x += fm.horizontalAdvance(text) + self._margin

        if self._session:
            for idx in imgs[:6]:
                if idx is None:
                    x += self._thumb + self._margin
                    continue
                path = self._session.im_list[idx]
                pix = _scaled_thumb(path, self._thumb)
                top = option.rect.top() + (option.rect.height() - pix.height()) // 2
                painter.drawPixmap(x, top, pix)
                x += self._thumb + self._margin

        if option.state & QStyle.State_HasFocus:
            option_rect = option.rect
            option_rect.adjust(0, 0, -1, -1)
            painter.setPen(QPen(option.palette.highlight().color()))
            painter.drawRect(option_rect)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # type: ignore[override]
        fm: QFontMetrics = option.fontMetrics
        name = index.data(NAME_ROLE)
        count = index.data(COUNT_ROLE)
        text = f"{name} ({count})"
        width = fm.horizontalAdvance(text) + self._margin + (self._thumb + self._margin) * 6
        height = max(fm.height(), self._thumb) + self._margin * 2
        return QSize(width, height)


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
        self.list_widget.setUniformItemSizes(True)
        self.delegate = _OverlapDelegate(None, self.THUMB_SIZE, self.list_widget)
        self.list_widget.setItemDelegate(self.delegate)
        self.list_widget.itemDoubleClicked.connect(self._on_double_clicked)
        self.setWidget(self.list_widget)

        self.bus.imagesChanged.connect(self._on_images_changed)

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        self.delegate.set_session(session)
        self.list_widget.clear()

    def _on_images_changed(self, idxs: List[int]):
        if self.session is None:
            self.list_widget.clear()
            return
        self._update_list(idxs)

    # ------------------------------------------------------------------
    def _update_list(self, idxs: Iterable[int]):
        start_timer17 = time.perf_counter()
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
            item = QListWidgetItem(f"{name} ({count})")
            item.setData(NAME_ROLE, name)
            item.setData(COUNT_ROLE, count)
            item.setData(IMAGES_ROLE, triplets.get(name, ())[:6])
            self.list_widget.addItem(item)

        self._last_indices = indices
        print('_update_list',time.perf_counter() - start_timer17)
    # ------------------------------------------------------------------
    def _on_double_clicked(self, item: QListWidgetItem):
        name = item.data(NAME_ROLE)
        if not name or self.session is None:
            return
        overlap = set(self.session.hyperedges.get(name, set())) & self._last_indices
        highlight = {i: QColor(221, 160, 221) for i in overlap}  # plum / light violet
        self.grid._highlight_next = highlight
        self.bus.set_edges([name])