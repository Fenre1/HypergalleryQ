from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QTableWidget, QTableWidgetItem, QLineEdit, QLabel
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class ZoomPanGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        event.accept()


class ImageMetadataDialog(QDialog):
    def __init__(self, image_path: str, metadata: Mapping[str, Any] | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(image_path).name)

        self.view = ZoomPanGraphicsView()
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        pix = QPixmap(image_path)
        self.pix_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pix_item)
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, 2)
        side = QVBoxLayout()
        side.addWidget(QLabel("Metadata"))
        side.addWidget(self.filter_edit)
        side.addWidget(self.table)
        layout.addLayout(side, 1)

        self._populate_table(metadata or {})
        self.filter_edit.textChanged.connect(self._apply_filter)

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


def show_image_metadata(session, idx: int, parent=None):
    path = session.im_list[idx]
    row = session.metadata[session.metadata["image_path"] == path]
    meta = row.iloc[0].to_dict() if not row.empty else {}
    dlg = ImageMetadataDialog(path, meta, parent)
    dlg.exec_()