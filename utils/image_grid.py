from __future__ import annotations

# from PySide6.QtWidgets import (
#     QListView, QDockWidget
# )

from PyQt5.QtWidgets import (
    QListView, QDockWidget
)

# from PySide6.QtGui import QPixmap, QIcon, QImage
# from PySide6.QtCore import (
#     Qt, QAbstractListModel, QModelIndex, QSize, QObject, QThread, Signal
# )
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import (
    Qt, QAbstractListModel, QModelIndex, QSize, QObject, QThread, pyqtSignal as Signal
)

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .similarity import SIM_METRIC


class _ThumbWorker(QObject):
    """Worker object living in a QThread that loads thumbnails."""

    thumbReady = Signal(int, QImage)        # idx, QImage

    def __init__(self, im_list: list[str], thumb: int):
        super().__init__()
        self._im_list = im_list
        self._thumb = thumb
    def load(self, idx: int):
        img = QImage(self._im_list[idx])
        if not img.isNull():
            img = img.scaled(self._thumb, self._thumb,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumbReady.emit(idx, img)


class ImageListModel(QAbstractListModel):
    """Model that provides thumbnails lazily using a background thread."""

    requestThumb = Signal(int)              # idx → worker

    def __init__(self, session: SessionModel, idxs: list[int], thumb_size: int = 128, parent=None):
        super().__init__(parent)
        self._session = session
        self._indexes = idxs
        self._thumb = thumb_size
        self._preload = 64

        self._pixmaps: dict[int, QPixmap] = {}
        self._index_map = {idx: row for row, idx in enumerate(self._indexes)}
        self._placeholder = QPixmap(self._thumb, self._thumb)
        self._placeholder.fill(Qt.gray)
        self._requested: set[int] = set()

        # background worker for loading thumbnails
        self._thread = QThread(self)
        self._worker = _ThumbWorker(self._session.im_list, self._thumb)
        self._worker.moveToThread(self._thread)
        self.requestThumb.connect(self._worker.load)
        self._worker.thumbReady.connect(self._on_thumb_ready)
        self._thread.start()

    def __del__(self):
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self._indexes)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._indexes)):
            return None
        if role == Qt.DecorationRole:
            idx = self._indexes[index.row()]
            pix = self._pixmaps.get(idx)
            if pix is None:
                self._request_range(index.row())
                pix = self._placeholder
            return QIcon(pix)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # type: ignore[override]
        fl = super().flags(index)
        if index.isValid():
            fl |= Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return fl

    def _on_thumb_ready(self, idx: int, img: QImage):
        self._pixmaps[idx] = QPixmap.fromImage(img)
        self._requested.discard(idx)
        row = self._index_map.get(idx)
        if row is not None:
            i = self.index(row)
            self.dataChanged.emit(i, i, [Qt.DecorationRole])

    def _request_range(self, row: int):
        start = max(0, row - self._preload)
        end = min(len(self._indexes), row + self._preload + 1)
        for r in range(start, end):
            idx = self._indexes[r]
            if idx not in self._pixmaps and idx not in self._requested:
                self._requested.add(idx)
                self.requestThumb.emit(idx)


class ImageGridDock(QDockWidget):
    """Dock widget that displays selected images using a QListView."""

    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 128):
        super().__init__("Images", parent)
        self.bus = bus
        self.thumb_size = thumb_size
        self.session: SessionModel | None = None
        self._selected_edges: list[str] = []

        self.view = QListView()
        self.view.setViewMode(QListView.IconMode)
        self.view.setIconSize(QSize(self.thumb_size, self.thumb_size))
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSelectionMode(QListView.ExtendedSelection)
        self.view.setSpacing(4)
        self.view.setUniformItemSizes(True)
        self.view.setLayoutMode(QListView.Batched)
        self.view.setBatchSize(64)
        self.setWidget(self.view)

        self.bus.imagesChanged.connect(self.update_images)
        self.bus.edgesChanged.connect(self._remember_edges)

    def set_model(self, model: SessionModel):
        self.session = model
        self.update_images([])

    # ------------------------------------------------------------------
    def update_images(self, idxs: list[int]):
        if self.session is None:
            self.view.setModel(None)
            return
        if idxs and self._selected_edges:
            vecs = [self.session.hyperedge_avg_features[e]
                    for e in self._selected_edges
                    if e in self.session.hyperedge_avg_features]
            if vecs:
                ref = sum(vecs) / len(vecs)
                feats = self.session.features[idxs]
                sims = SIM_METRIC(ref.reshape(1, -1), feats)[0]
                idxs = [i for _, i in sorted(zip(sims, idxs), reverse=True)]

        model = ImageListModel(self.session, idxs, self.thumb_size, self)
        self.view.setModel(model)

    def _remember_edges(self, names: list[str]):
        self._selected_edges = names