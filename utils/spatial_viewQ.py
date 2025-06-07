from __future__ import annotations
import os
# Force pyqtgraph to use the same Qt binding as the rest of the app
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import numpy as np
import pyqtgraph as pg
# Pull all Qt classes through pyqtgraph's wrapper so bindings stay consistent
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
from matplotlib.path import Path as MplPath
from matplotlib import pyplot as plt
import umap

from .selection_bus import SelectionBus
from .session_model import SessionModel

# Disable OpenGL if needed
pg.setConfigOption('useOpenGL', False)

# Aliases for convenience
Signal = QtCore.Signal
Qt = QtCore.Qt
QPointF = QtCore.QPointF
QPen = QtGui.QPen
QColor = QtGui.QColor
QPainterPath = QtGui.QPainterPath
QPalette = QtGui.QPalette
QPixmap = QtGui.QPixmap
QGraphicsPixmapItem = QtWidgets.QGraphicsPixmapItem
QGraphicsRectItem = QtWidgets.QGraphicsRectItem
QDockWidget = QtWidgets.QDockWidget
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

class LassoViewBox(pg.ViewBox):
    """
    A ViewBox that supports lasso selection with the left mouse button
    and standard panning with the right mouse button.
    """
    sigLassoFinished = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lasso_path_item: QtWidgets.QGraphicsPathItem | None = None
        self._is_drawing_lasso = False
        self._is_panning = False
        self._pan_start_pos = QPointF()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._is_drawing_lasso = True
            if self._lasso_path_item and self._lasso_path_item.scene():
                self.removeItem(self._lasso_path_item)
            start_pos = self.mapToView(ev.pos())
            self._lasso_path = QPainterPath(start_pos)
            pen = QPen(QColor("yellow"), 2)
            pen.setCosmetic(True)
            self._lasso_path_item = QtWidgets.QGraphicsPathItem(self._lasso_path)
            self._lasso_path_item.setPen(pen)
            self.addItem(self._lasso_path_item, ignoreBounds=True)
            ev.accept()
        elif ev.button() == Qt.RightButton:
            self._is_panning = True
            self._pan_start_pos = self.mapToView(ev.pos())
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._is_drawing_lasso:
            self._lasso_path.lineTo(self.mapToView(ev.pos()))
            self._lasso_path_item.setPath(self._lasso_path)
            ev.accept()
        elif self._is_panning:
            current_pos = self.mapToView(ev.pos())
            delta = self._pan_start_pos - current_pos
            self.translateBy(delta)
            self._pan_start_pos = current_pos
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._is_drawing_lasso and ev.button() == Qt.LeftButton:
            self._is_drawing_lasso = False
            polygon = self._lasso_path.toSubpathPolygons()
            if polygon:
                self.sigLassoFinished.emit(polygon[0])
            if self._lasso_path_item and self._lasso_path_item.scene():
                self.removeItem(self._lasso_path_item)
            self._lasso_path_item = None
            ev.accept()
        elif self._is_panning and ev.button() == Qt.RightButton:
            self._is_panning = False
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)


class SpatialViewQDock(QDockWidget):
    """Pyqtgraph‑based spatial view with lasso selection."""

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial View", parent)
        self.bus = bus
        self.session: SessionModel | None = None
        self.embedding: np.ndarray | None = None
        self.color_map: dict[str, str] = {}
        self.image_items: list[QGraphicsPixmapItem] = []
        self.scatter: pg.ScatterPlotItem | None = None

        self.view = LassoViewBox()
        self.plot = pg.PlotWidget(viewBox=self.view)

        
        self.setWidget(self.plot)
        self._apply_theme()
        
        self.view.sigLassoFinished.connect(self._on_lasso_select)
        self.bus.edgesChanged.connect(self._on_edges_changed)
        print('self.scatter',self.scatter)
        print('self.plot',self.plot)

    def _is_dark_mode(self) -> bool:
        pal = self.palette()
        col = pal.color(QPalette.Window)
        return col.lightness() < 128

    def _apply_theme(self):
        bg = "#333333" if self._is_dark_mode() else "#FFFFFF"
        fg = "#FFFFFF" if self._is_dark_mode() else "#000000"
        self.plot.setBackground(bg)
        for ax in ('bottom', 'left'):
            axis = self.plot.getAxis(ax)
            axis.setPen(fg)
            axis.setTextPen(fg)

    def set_model(self, session: SessionModel | None):
        print("\n--- [set_model] called ---")
        self.session = session
        self.plot.clear()
        self._clear_image_items()
        self.scatter = None
        if not session or session.features is None or len(session.features) == 0:
            print("[set_model] No data; clearing.")
            self.embedding = None
            return
        self.embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(session.features)
        self.scatter = pg.ScatterPlotItem(
            x=self.embedding[:,0], y=self.embedding[:,1],
            size=7, pen=None, brush=pg.mkBrush('gray'), symbol='o'
        )
        self.plot.addItem(self.scatter)
        edges = list(session.hyperedges)
        cmap = plt.get_cmap("tab20")
        self.color_map = {
            n: plt.matplotlib.colors.to_hex(cmap(i / len(edges)))
            for i, n in enumerate(edges)
        }
        self.plot.autoRange()
        print('self.scatter2',self.scatter.getData())

    def _on_lasso_select(self, pts: list[QPointF]):
        if self.embedding is None:
            return
        poly = np.array([(p.x(), p.y()) for p in pts])
        if poly.shape[0] < 3:
            return
        path = MplPath(poly)
        idxs = np.nonzero(path.contains_points(self.embedding))[0]
        print(f"Lasso selected {len(idxs)} points.")
        self.bus.set_images(list(idxs))

    def _on_edges_changed(self, names: list[str]):
        if not self.session or self.embedding is None or self.scatter is None:
            return
        self._clear_image_items()

        brushes = [pg.mkBrush('gray')] * len(self.embedding)
        if names:
            main = names[0]
            selected = set(self.session.hyperedges.get(main, set()))
            overlaps = set()
            for idx in selected:
                overlaps.update(self.session.image_mapping.get(idx, set()))
            overlaps.discard(main)

            for idx in range(len(self.embedding)):
                edges = self.session.image_mapping.get(idx, set())
                if main in edges:
                    brushes[idx] = pg.mkBrush(self.color_map.get(main, 'red'))
                else:
                    ov = next((e for e in overlaps if e in edges), None)
                    if ov:
                        brushes[idx] = pg.mkBrush(self.color_map.get(ov, 'blue'))

            pts = self.embedding[list(selected)]
            if len(pts):
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                dx = xmax - xmin
                dy = ymax - ymin
                pad_x = dx * 0.1 if dx > 0 else 1
                pad_y = dy * 0.1 if dy > 0 else 1
                self.view.setXRange(xmin - pad_x, xmax + pad_x)
                self.view.setYRange(ymin - pad_y, ymax + pad_y)

            unique = [i for i in selected
                      if len(self.session.image_mapping.get(i, set())) == 1]
            if not unique:
                imgs = list(selected)
                if len(imgs) > 3:
                    imgs = list(np.random.choice(imgs, 3, replace=False))
            else:
                imgs = unique[:3]
            self._add_sample_images(imgs, self.color_map.get(main, 'yellow'))

            for edge in overlaps:
                inter = list(self.session.hyperedges.get(edge, set()) & selected)
                if len(inter) > 3:
                    inter = list(np.random.choice(inter, 3, replace=False))
                self._add_sample_images(inter, self.color_map.get(edge, 'yellow'))

        self.scatter.setData(x=self.embedding[:,0], y=self.embedding[:,1],
                             brush=brushes, size=7, pen=None, symbol='o')

    def _clear_image_items(self):
        for item in self.image_items:
            if item.scene():
                self.plot.removeItem(item)
        self.image_items.clear()

    def _add_sample_images(self, idxs: list[int], color: str):
        for idx in idxs:
            img_path = self.session.im_list[idx]
            pix = QPixmap(img_path)
            if pix.isNull():
                continue
            if pix.width() > pix.height():
                pix = pix.scaledToWidth(64, Qt.SmoothTransformation)
            else:
                pix = pix.scaledToHeight(64, Qt.SmoothTransformation)

            item = QGraphicsPixmapItem(pix)
            item.setOffset(-pix.width() / 2, -pix.height() / 2)
            item.setPos(self.embedding[idx, 0], self.embedding[idx, 1])
            rect = QGraphicsRectItem(0, 0, pix.width(), pix.height(), parent=item)
            pen = QPen(QColor(color))
            pen.setWidth(2)
            rect.setPen(pen)
            rect.setBrush(Qt.NoBrush)
            self.plot.addItem(item)
            self.image_items.append(item)

    def _run_sanity_check_test(self):
        test_scatter = pg.ScatterPlotItem(x=[0,1,2], y=[0,1,0], size=20, pen=None, brush='r')
        self.plot.addItem(test_scatter)
        self.plot.autoRange()
        print("[SANITY CHECK] View range:", self.plot.viewRange())

