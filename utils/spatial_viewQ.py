from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QGraphicsPixmapItem, QGraphicsRectItem
from PySide6.QtGui import QPalette, QPixmap, QPen, QColor, QPainterPath
from PySide6.QtCore import Qt, QPointF, Signal
from matplotlib.path import Path as MplPath

from .selection_bus import SelectionBus
from .session_model import SessionModel

import umap


class LassoViewBox(pg.ViewBox):
    """ViewBox that emits a polygon drawn with the left mouse button."""

    sigLassoFinished = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drawing = False
        self._path = QPainterPath()
        self._item = None
        self._panning = False
        self._pan_start = None
        self._range_start = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drawing = True
            self._path = QPainterPath(self.mapToView(ev.pos()))
            pen = QPen(pg.mkColor('y'))
            self._item = pg.QtWidgets.QGraphicsPathItem()
            self._item.setPen(pen)
            self._item.setFlag(
                pg.QtWidgets.QGraphicsPathItem.ItemIgnoresTransformations
            )
            self.addItem(self._item)
            ev.accept()
            return
        if ev.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = ev.pos()
            self._range_start = self.viewRange()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._path.lineTo(self.mapToView(ev.pos()))
            self._item.setPath(self._path)
            ev.accept()
            return
        if self._panning:
            if self._pan_start is None or self._range_start is None:
                return
            dx = ev.pos().x() - self._pan_start.x()
            dy = ev.pos().y() - self._pan_start.y()
            (x0, x1), (y0, y1) = self._range_start
            w = self.width()
            h = self.height()
            dx_data = -dx * (x1 - x0) / w
            dy_data = -dy * (y1 - y0) / h
            self.setXRange(x0 + dx_data, x1 + dx_data, padding=0)
            self.setYRange(y0 + dy_data, y1 + dy_data, padding=0)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drawing and ev.button() == Qt.LeftButton:
            self._drawing = False
            item = self._item
            path = self._path
            self.removeItem(item)
            self._item = None
            self._path = QPainterPath()
            pts = []
            for i in range(path.elementCount()):
                el = path.elementAt(i)
                pts.append(QPointF(el.x, el.y))
            if len(pts) > 2:
                self.sigLassoFinished.emit(pts)
            ev.accept()
            return
        if self._panning and ev.button() == Qt.RightButton:
            self._panning = False
            self._pan_start = None
            self._range_start = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class SpatialViewQDock(QDockWidget):
    """PyQtGraph-based spatial view with lasso selection."""

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial View", parent)
        self.bus = bus
        self.session: SessionModel | None = None
        self.embedding: np.ndarray | None = None
        self.color_map: dict[str, str] = {}
        self.image_items: list[QGraphicsPixmapItem] = []

        self.view = LassoViewBox()
        self.view.sigLassoFinished.connect(self._on_lasso_select)
        self.plot = pg.PlotWidget(viewBox=self.view)
        self._apply_theme()

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.plot)
        self.setWidget(widget)

        self.scatter = None
        self.bus.edgesChanged.connect(self._on_edges_changed)

    # ------------------------------------------------------------------
    def _is_dark_mode(self) -> bool:
        pal = self.palette()
        col = pal.color(QPalette.Window)
        lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
        return lum < 128

    # ------------------------------------------------------------------
    def _apply_theme(self):
        if self._is_dark_mode():
            bg = "#333333"
            fg = "#FFFFFF"
        else:
            bg = "#FFFFFF"
            fg = "#000000"
        self.plot.setBackground(bg)
        ax = self.plot.getPlotItem()
        ax.getAxis('bottom').setPen(fg)
        ax.getAxis('left').setPen(fg)
        ax.getAxis('bottom').setTextPen(fg)
        ax.getAxis('left').setTextPen(fg)

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        self.plot.clear()
        self._clear_image_items()
        if not session:
            self.embedding = None
            self.scatter = None
            return

        self.embedding = umap.UMAP(n_components=2).fit_transform(session.features)
        self.scatter = pg.ScatterPlotItem(
            x=self.embedding[:, 0],
            y=self.embedding[:, 1],
            size=5,
            pen=None,
            brush=pg.mkBrush('gray'),
            symbol='o',
        )
        self.scatter.setZValue(0)
        self.plot.addItem(self.scatter)
        # auto-range once after adding scatter plot
        self.plot.enableAutoRange(pg.ViewBox.XYAxes, True)
        self.plot.autoRange()
        self.plot.enableAutoRange(pg.ViewBox.XYAxes, False)

        edges = list(session.hyperedges)
        self.color_map = {
            n: pg.mkColor(pg.intColor(i, hues=len(edges))).name()
            for i, n in enumerate(edges)
        }

    # ------------------------------------------------------------------
    def _on_lasso_select(self, pts: list[QPointF]):
        if self.embedding is None:
            return
        poly = [(p.x(), p.y()) for p in pts]
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])
        path = MplPath(poly)
        idxs = np.nonzero(path.contains_points(self.embedding))[0]
        self.bus.set_images(list(map(int, idxs)))

    # ------------------------------------------------------------------
    def _on_edges_changed(self, names: list[str]):
        if not self.session or self.embedding is None or self.scatter is None:
            return
        self._clear_image_items()

        colors = ['gray'] * len(self.embedding)
        if names:
            main = names[0]
            selected = set(self.session.hyperedges.get(main, set()))
            overlaps = set()
            for idx in selected:
                overlaps.update(self.session.image_mapping.get(idx, set()))
            overlaps.discard(main)

            for i in range(len(self.embedding)):
                edges = self.session.image_mapping.get(i, set())
                if main in edges:
                    colors[i] = self.color_map.get(main, 'red')
                else:
                    ov = next((e for e in overlaps if e in edges), None)
                    if ov:
                        colors[i] = self.color_map.get(ov, 'blue')

            if selected:
                pts = self.embedding[list(selected)]
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                dx = xmax - xmin
                dy = ymax - ymin
                pad_x = dx * 0.1 if dx > 0 else 1
                pad_y = dy * 0.1 if dy > 0 else 1
                self.plot.setXRange(xmin - pad_x, xmax + pad_x)
                self.plot.setYRange(ymin - pad_y, ymax + pad_y)

            unique = [i for i in selected
                      if len(self.session.image_mapping.get(i, set())) == 1]
            imgs = unique[:3] if unique else list(selected)[:3]
            self._add_sample_images(imgs, self.color_map.get(main, 'yellow'))

            for edge in overlaps:
                inter = list(self.session.hyperedges.get(edge, set()) & selected)
                self._add_sample_images(inter[:3],
                                        self.color_map.get(edge, 'yellow'))

        brushes = [pg.mkBrush(c) for c in colors]
        self.scatter.setBrush(brushes)
        self.plot.repaint()

    # ------------------------------------------------------------------
    def _add_sample_images(self, idxs: list[int], color: str):
        for idx in idxs:
            if idx >= len(self.embedding):
                continue
            path = self.session.im_list[idx]
            pix = QPixmap(path)
            if pix.isNull():
                continue
            pix = pix.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QGraphicsPixmapItem(pix)
            item.setOffset(-pix.width() / 2, -pix.height() / 2)
            item.setPos(self.embedding[idx, 0], self.embedding[idx, 1])
            item.setFlag(QGraphicsPixmapItem.ItemIgnoresTransformations)
            item.setZValue(1)
            rect = QGraphicsRectItem(
                -pix.width() / 2,
                -pix.height() / 2,
                pix.width(),
                pix.height(),
                item,
            )
            rect.setFlag(QGraphicsRectItem.ItemIgnoresTransformations)
            pen = QPen(QColor(color))
            pen.setWidth(2)
            rect.setPen(pen)
            self.plot.addItem(item)
            self.image_items.append(item)

    # ------------------------------------------------------------------
    def _clear_image_items(self):
        for item in self.image_items:
            if item.scene():
                item.scene().removeItem(item)
        self.image_items.clear()