from __future__ import annotations

from functools import lru_cache
from PySide6.QtWidgets import (
    QListView, QDockWidget
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QSize

from .selection_bus import SelectionBus
from .session_model import SessionModel


class ImageListModel(QAbstractListModel):
    """Lightweight model providing thumbnails for a set of image indices."""

    def __init__(self, session: SessionModel, idxs: list[int], thumb_size: int = 128, parent=None):
        super().__init__(parent)
        self._session = session
        self._indexes = idxs
        self._thumb = thumb_size

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self._indexes)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._indexes)):
            return None
        if role == Qt.DecorationRole:
            idx = self._indexes[index.row()]
            return QIcon(self._load_thumb(idx))
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        fl = super().flags(index)
        if index.isValid():
            fl |= Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return fl

    @lru_cache(maxsize=256)
    def _load_thumb(self, idx: int) -> QPixmap:
        path = self._session.im_list[idx]
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(self._thumb, self._thumb, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix


class ImageGridDock(QDockWidget):
    """Dock widget that displays selected images using a QListView."""

    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 128):
        super().__init__("Images", parent)
        self.bus = bus
        self.thumb_size = thumb_size
        self.session: SessionModel | None = None

        self.view = QListView()
        self.view.setViewMode(QListView.IconMode)
        self.view.setIconSize(QSize(self.thumb_size, self.thumb_size))
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSelectionMode(QListView.ExtendedSelection)
        self.view.setSpacing(4)
        self.setWidget(self.view)

        self.bus.imagesChanged.connect(self.update_images)

    def set_model(self, model: SessionModel):
        self.session = model
        self.update_images([])

    # ------------------------------------------------------------------
    def update_images(self, idxs: list[int]):
        if self.session is None:
            self.view.setModel(None)
            return
        model = ImageListModel(self.session, idxs, self.thumb_size, self)
        self.view.setModel(model)
