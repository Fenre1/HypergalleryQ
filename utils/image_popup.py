from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QTableWidget, QTableWidgetItem, QLineEdit, QLabel,
    QPushButton
)
from PyQt5.QtGui import QPixmap, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, pyqtSignal as Signal, QEvent
from PIL import Image, ImageOps
import numpy as np
import torch

from .similarity import SIM_METRIC
from .feature_extraction import Swinv2LargeFeatureExtractor
from .image_utils import pixmap_from_file


class ZoomPanGraphicsView(QGraphicsView):
    """Graphics view supporting zooming and shift-drag rectangle selection."""

    selectionChanged = Signal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self._drag_start = None
        self._rect_item = None

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    # --------------------------------------------------------------
    def mousePressEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ShiftModifier
        ):
            self._drag_start = self.mapToScene(event.pos())
            if self._rect_item is None:
                pen = QPen(QColor("red"))
                pen.setWidth(2)
                self._rect_item = self.scene().addRect(QRectF(), pen)
                self._rect_item.setZValue(10)
            self._rect_item.setRect(QRectF(self._drag_start, self._drag_start))
            self._rect_item.show()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            pos = self.mapToScene(event.pos())
            rect = QRectF(self._drag_start, pos).normalized()
            self._rect_item.setRect(rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_start is not None and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            rect = QRectF(self._drag_start, pos).normalized()
            self._drag_start = None
            self.selectionChanged.emit(rect)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def clear_selection(self):
        if self._rect_item is not None:
            self._rect_item.hide()
        self._drag_start = None


class ImageMetadataDialog(QDialog):
    """Dialog showing a larger image with metadata and ranking tools."""

    _extractor: Swinv2LargeFeatureExtractor | None = None

    def __init__(
        self,
        session,
        idx: int,
        parent=None,
        indices: list[int] | None = None,
        ):

        super().__init__(parent)

        self._session = session
        self._indices = indices or list(range(len(session.im_list)))
        self._pos = self._indices.index(idx) if idx in self._indices else 0
        self._image_path = ""
        self._sel_rect: QRectF | None = None

        self.view = ZoomPanGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.rank_btn = QPushButton("Rank Selection")
        self.rank_btn.setEnabled(False)
        self.rank_btn.clicked.connect(self._on_rank_selection)

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, 2)
        side = QVBoxLayout()
        side.addWidget(QLabel("Metadata"))
        side.addWidget(self.filter_edit)
        side.addWidget(self.table)
        side.addWidget(self.rank_btn)
        layout.addLayout(side, 1)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.view.selectionChanged.connect(self._on_selection_changed)
        
        self.installEventFilter(self)
        self._load_image(self._indices[self._pos])




    def _populate_table(self, metadata: Mapping[str, Any]):
        self.table.setRowCount(0)
        for key, value in metadata.items():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(key)))
            self.table.setItem(r, 1, QTableWidgetItem(str(value)))

    def _apply_filter(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            combined = f"{key_item.text()} {val_item.text()}".lower()
            match = text in combined
            self.table.setRowHidden(row, not match)

    # ------------------------------------------------------------------
    def _on_selection_changed(self, rect: QRectF):
        """Handle selection rectangle updates from the graphics view."""
        if rect is None or rect.isNull():
            self._sel_rect = None
        else:
            self._sel_rect = rect

        self.rank_btn.setEnabled(self._session is not None and self._sel_rect is not None)

    def _load_image(self, idx: int) -> None:
        """Load image and metadata for the given session index."""
        self._sel_rect = None
        self.rank_btn.setEnabled(False)
        self.view.clear_selection()

        self._image_path = self._session.im_list[idx]
        self.setWindowTitle(Path(self._image_path).name)
        pix = pixmap_from_file(self._image_path)
        self.pix_item.setPixmap(pix)
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
        self.view.setFocus()

        row = self._session.metadata[self._session.metadata["image_path"] == self._image_path]
        meta = row.iloc[0].to_dict() if not row.empty else {}
        self._populate_table(meta)
        self.filter_edit.clear()

    def _show_prev(self) -> None:
        if self._pos > 0:
            self._pos -= 1
            self._load_image(self._indices[self._pos])

    def _show_next(self) -> None:
        if self._pos + 1 < len(self._indices):
            self._pos += 1
            self._load_image(self._indices[self._pos])

    def eventFilter(self, obj, event):  
        if event.type() == QEvent.KeyPress:
            print('event')
            if event.key() == Qt.Key_Left:
                print('event lkeft')
                self._show_prev()
                return True
            if event.key() == Qt.Key_Right:
                print('event right')
                self._show_next()
                return True
        return super().eventFilter(obj, event)

    def _on_rank_selection(self):
        if not (self._session and self._sel_rect):
            return
        rect = self._sel_rect.intersected(self.pix_item.boundingRect())
        if rect.width() <= 2 or rect.height() <= 2:
            return

        if ImageMetadataDialog._extractor is None:
            ImageMetadataDialog._extractor = Swinv2LargeFeatureExtractor(batch_size=1)

        img = ImageOps.exif_transpose(Image.open(self._image_path)).convert("RGB")
        crop = img.crop((int(rect.x()), int(rect.y()), int(rect.x()+rect.width()), int(rect.y()+rect.height())))
        tensor = ImageMetadataDialog._extractor.transform(crop).unsqueeze(0).to(ImageMetadataDialog._extractor.device)
        with torch.no_grad():
            vec = ImageMetadataDialog._extractor.model(tensor).cpu().numpy()[0]

        feats = self._session.features
        sims = SIM_METRIC(vec.reshape(1, -1), feats)[0]
        ranked = np.argsort(sims)[::-1][:500]

        win = self.parent()
        while win and not hasattr(win, "image_grid"):
            win = win.parent()
        if win and hasattr(win, "image_grid"):
            win.image_grid.update_images(list(ranked), sort=False, query=True)


_open_dialogs: list[ImageMetadataDialog] = []

def show_image_metadata(session, idx: int, parent=None):
    """Display the metadata dialog for the given image index."""
    indices = getattr(parent, "_current_indices", None)
    dlg = ImageMetadataDialog(session, idx, parent=parent, indices=indices)
    _open_dialogs.append(dlg)
    dlg.finished.connect(lambda *_: _open_dialogs.remove(dlg))
    dlg.show()