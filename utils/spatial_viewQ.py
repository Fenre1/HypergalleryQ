from __future__ import annotations
import os
# Force pyqtgraph to use the same Qt binding as the rest of the app
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import numpy as np
import pyqtgraph as pg
# Pull all Qt classes through pyqtgraph's wrapper so bindings stay consistent
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
from matplotlib.path import Path as MplPath
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
        # self.plot.clear()
        # self._clear_image_items()
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
        # Example: re-center on selection or recolor...
        # (Implementation omitted for brevity.)

    def _clear_image_items(self):
        for item in self.image_items:
            if item.scene():
                self.plot.removeItem(item)
        self.image_items.clear()

    def _run_sanity_check_test(self):
        test_scatter = pg.ScatterPlotItem(x=[0,1,2], y=[0,1,0], size=20, pen=None, brush='r')
        self.plot.addItem(test_scatter)
        self.plot.autoRange()
        print("[SANITY CHECK] View range:", self.plot.viewRange())

