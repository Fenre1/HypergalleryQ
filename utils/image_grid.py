from __future__ import annotations

from PyQt5.QtWidgets import (
    QListView, QDockWidget, QWidget, QLabel, QGridLayout, QHBoxLayout,
    QVBoxLayout, QFrame, QScrollArea
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import (
    Qt, QAbstractListModel, QModelIndex, QSize, QObject, QThread,
    pyqtSignal as Signal, QEvent
)
from pathlib import Path

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .similarity import SIM_METRIC
from .image_popup import show_image_metadata



class _ClickableLabel(QLabel):
    """Label used for emitting a signal on double click."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)

class _ThumbWorker(QObject):
    """Worker object living in a QThread that loads thumbnails or full images."""

    thumbReady = Signal(int, QImage)  # idx, QImage

    def __init__(self, session: SessionModel, thumb: int, use_full_images: bool):
        super().__init__()
        self._session = session
        self._thumb = thumb
        self._use_full = use_full_images

    def set_mode(self, use_full: bool) -> None:
        self._use_full = use_full

    def load(self, idx: int):
        if not self._use_full and self._session.thumbnail_data:
            if self._session.thumbnails_are_embedded:
                data = self._session.thumbnail_data[idx]
                img = QImage.fromData(data)
            else:
                path = Path(self._session.h5_path).parent / self._session.thumbnail_data[idx]
                img = QImage(str(path))
        else:
            img = QImage(self._session.im_list[idx])

        if not img.isNull():
            img = img.scaled(
                self._thumb,
                self._thumb,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.thumbReady.emit(idx, img)



class ImageListModel(QAbstractListModel):
    """Model that provides thumbnails lazily using a background thread."""

    requestThumb = Signal(int)              # idx → worker

    def __init__(self, session: SessionModel, idxs: list[int], thumb_size: int = 128, parent=None, highlight: list[int] | None = None, use_full_images: bool = False):
        super().__init__(parent)
        self._session = session
        self._indexes = idxs
        self._thumb = thumb_size
        self._preload = 64
        self._highlight = set(highlight or [])
        self._use_full = use_full_images

        self._pixmaps: dict[int, QPixmap] = {}
        self._index_map = {idx: row for row, idx in enumerate(self._indexes)}
        self._placeholder = QPixmap(self._thumb, self._thumb)
        self._placeholder.fill(Qt.gray)
        self._requested: set[int] = set()

        # background worker for loading thumbnails
        self._thread = QThread(self)
        self._worker = _ThumbWorker(self._session, self._thumb, self._use_full)
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
            if idx in self._highlight:
                pix = self._add_border(pix)
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

    def _add_border(self, pix: QPixmap, color: QColor = QColor("red"), width: int = 4) -> QPixmap:
        if pix.isNull():
            return pix
        bordered = QPixmap(pix.size())
        bordered.fill(Qt.transparent)
        painter = QPainter(bordered)
        painter.drawPixmap(0, 0, pix)
        pen = QPen(color)
        pen.setWidth(width)
        painter.setPen(pen)
        painter.drawRect(width // 2, width // 2, pix.width() - width, pix.height() - width)
        painter.end()
        return bordered

    def _request_range(self, row: int):
        start = max(0, row - self._preload)
        end = min(len(self._indexes), row + self._preload + 1)
        for r in range(start, end):
            idx = self._indexes[r]
            if idx not in self._pixmaps and idx not in self._requested:
                self._requested.add(idx)
                self.requestThumb.emit(idx)


class ImageGridDock(QDockWidget):
    """Dock widget that displays selected images or an overview of triplets."""

    labelDoubleClicked = Signal(str)  # hyperedge name
    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 128):
        super().__init__("Images", parent)
        self.bus = bus
        self.thumb_size = thumb_size
        self.session: SessionModel | None = None
        self._selected_edges: list[str] = []
        self.use_full_images: bool = False
        self._current_indices: list[int] = []

        self.view = QListView()
        self.view.setViewMode(QListView.IconMode)
        self.view.setIconSize(QSize(self.thumb_size, self.thumb_size))
        self.view.setResizeMode(QListView.Adjust)
        self.view.setSelectionMode(QListView.ExtendedSelection)
        self.view.setSpacing(4)
        self.view.setUniformItemSizes(True)
        self.view.setLayoutMode(QListView.Batched)
        self.view.setBatchSize(64)

        self._overview_widget: QScrollArea | None = None
        self._mode = "grid"  # or "overview"

        self.setWidget(self.view)

        self.view.doubleClicked.connect(self._on_double_clicked)

        self.bus.imagesChanged.connect(self.update_images)
        self.bus.edgesChanged.connect(self._remember_edges)


    def set_use_full_images(self, flag: bool) -> None:
        self.use_full_images = flag
        model = self.view.model()
        if isinstance(model, ImageListModel):
            model._worker.set_mode(flag)
        if self._current_indices:
            self.update_images(self._current_indices, sort=False)


    def set_model(self, model: SessionModel):
        self.session = model
        self.update_images([])

    # ------------------------------------------------------------------
    def update_images(self, idxs: list[int], highlight: list[int] | None = None, sort: bool = True):
        if self.session is None:
            self.view.setModel(None)
            return
        if self._mode == "overview":
            # switching back to normal grid mode
            self.show_grid()

        if sort and idxs and self._selected_edges:
            vecs = [self.session.hyperedge_avg_features[e]
                    for e in self._selected_edges
                    if e in self.session.hyperedge_avg_features]
            if vecs:
                ref = sum(vecs) / len(vecs)
                feats = self.session.features[idxs]
                sims = SIM_METRIC(ref.reshape(1, -1), feats)[0]
                idxs = [i for _, i in sorted(zip(sims, idxs), reverse=True)]

        self._current_indices = list(idxs)
        model = ImageListModel(
            self.session,
            idxs,
            self.thumb_size,
            self,
            highlight=highlight,
            use_full_images=self.use_full_images,
        )
        self.view.setModel(model)

    def _remember_edges(self, names: list[str]):
        self._selected_edges = names
        if len(names) == 1:
            title = f"Images of {names[0]}"
        elif len(names) > 1:
            title = f"Images of {len(names)} hyperedges"
        else:
            title = "Images"
        self.setWindowTitle(title)

    def _on_double_clicked(self, index: QModelIndex):
        if not self.session:
            return
        model = self.view.model()
        if not isinstance(model, ImageListModel):
            return
        row = index.row()
        if not (0 <= row < len(model._indexes)):
            return
        img_idx = model._indexes[row]
        show_image_metadata(self.session, img_idx, self)

            # ------------------------------------------------------------------
    def show_overview(self, triplets: dict[str, tuple[int | None, ...]], session: SessionModel):
        """Display an overview of 6-image sets for each hyperedge."""
        self.session = session
        self._mode = "overview"

        content = QWidget()
        layout = QGridLayout(content)
        layout.setSpacing(10)

        col = row = 0
        max_cols = 3
        for name, imgs in triplets.items():
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)
            frame.setLineWidth(2)
            v = QVBoxLayout(frame)
            lbl = _ClickableLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            v.addWidget(lbl)
            top = QHBoxLayout()
            bottom = QHBoxLayout()
            for pos, idx in enumerate(imgs[:6]):
                container = top if pos < 3 else bottom
                lbl_img = QLabel()
                lbl_img.setFixedSize(self.thumb_size, self.thumb_size)
                if idx is not None:
                    pix = QPixmap(session.im_list[idx])
                    if not pix.isNull():
                        pix = pix.scaled(
                            self.thumb_size,
                            self.thumb_size,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                    lbl_img.setPixmap(pix)

                style = ""
                if pos < 3:
                    style = "background-color: #ccffcc; border: 2px solid green;"
                elif pos < 5:
                    style = "background-color: orange;"
                elif pos == 5:
                    style = "background-color: red;"
                if style:
                    lbl_img.setStyleSheet(style)

                container.addWidget(lbl_img)
            v.addLayout(top)
            v.addLayout(bottom)
            layout.addWidget(frame, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

            lbl.installEventFilter(self)

        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self._overview_widget = scroll
        self.setWidget(scroll)
        
    def show_grid(self):
        if self._mode != "grid":
            self._mode = "grid"
            self.setWidget(self.view)
            if self._overview_widget is not None:
                self._overview_widget.deleteLater()
                self._overview_widget = None

    # --------------------------------------------------------------
    def eventFilter(self, obj, event):
        if isinstance(obj, _ClickableLabel) and event.type() == QEvent.MouseButtonDblClick:
            self.labelDoubleClicked.emit(obj.text())
            return True
        return super().eventFilter(obj, event)