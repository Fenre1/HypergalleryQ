from __future__ import annotations

import os
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QApplication,
    QPushButton,
)
from PyQt5.QtGui import QPalette, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, pyqtSignal as Signal, QTimer, QEvent

from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import umap
from matplotlib.path import Path as MplPath
from .selection_bus import SelectionBus
from .session_model import SessionModel
from .image_popup import show_image_metadata
from .fa2_layout import HyperedgeForceAtlas2
from .fast_sim_engine import SimulationEngine 



class LassoViewBox(pg.ViewBox):
    """ViewBox that emits a polygon drawn with Shift + left mouse."""
  
    sigLassoFinished = Signal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drawing = False
        self._path = QPainterPath()
        self._item = None

    def mousePressEvent(self, ev):
        if (
            ev.button() == Qt.LeftButton
            and ev.modifiers() & Qt.ShiftModifier
        ):
            self._drawing = True
            self._path = QPainterPath(self.mapToView(ev.pos()))
            pen = QPen(pg.mkColor("y"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            # Use pg.QtWidgets for compatibility
            self._item = pg.QtWidgets.QGraphicsPathItem() 
            self._item.setPen(pen)
            self.addItem(self._item)
            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._path.lineTo(self.mapToView(ev.pos()))
            self._item.setPath(self._path)
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drawing and ev.button() == Qt.LeftButton:
            self._drawing = False
            self.removeItem(self._item)
            path = []
            for i in range(self._path.elementCount()):
                el = self._path.elementAt(i)
                path.append(QPointF(el.x, el.y))
            if len(path) > 2:
                self.sigLassoFinished.emit(path)
            ev.accept()
            return

        super().mouseReleaseEvent(ev)

class MiniMapViewBox(pg.ViewBox):
    """ViewBox used for the minimap to handle click navigation."""

    sigGoto = Signal(float, float)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.mapToView(ev.pos())
            self.sigGoto.emit(pos.x(), pos.y())
            ev.accept()
        else:
            super().mouseClickEvent(ev)





class SpatialViewQDock(QDockWidget):
    """PyQtGraph-based spatial view that visualizes hyperedges with a ForceAtlas2 layout."""
    
    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Hyperedge View", parent)
        self.bus = bus
        self.session: SessionModel | None = None
        self.color_map: dict[str, str] = {}
        
        self.fa2_layout: HyperedgeForceAtlas2 | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._update_frame)

        self.auto_stop_ms = 10000
        self.run_button = QPushButton("Pause Layout")
        self.run_button.clicked.connect(self.on_run_button_clicked)
        self.run_button.setEnabled(False)

        # MODIFIED: Use LassoViewBox and connect its signal
        self.view = LassoViewBox()
        self.view.sigRangeChanged.connect(self._update_minimap_view)
        self.view.sigLassoFinished.connect(self._on_lasso_select)
        
        self.plot = pg.PlotWidget(viewBox=self.view)
        self.plot.setBackground('#444444')
        self.plot.scene().sigMouseClicked.connect(self._on_scene_mouse_clicked)

        self.minimap_view = MiniMapViewBox(enableMenu=False)
        self.minimap_view.sigGoto.connect(self._goto_position)
        self.minimap = pg.PlotWidget(viewBox=self.minimap_view, parent=self.plot)
        self.minimap.setFixedSize(200, 200)
        self.minimap.hideAxis('bottom')
        self.minimap.hideAxis('left')
        self.minimap.setBackground('#333333')
        self.minimap.setMouseEnabled(False, False)
        self.minimap_scatter = None
        pen = pg.mkPen('r', width=2, cosmetic=True)
        self.minimap_rect = pg.RectROI([0, 0], [1, 1], pen=pen, movable=True, resizable=False)
        self.minimap_view.addItem(self.minimap_rect)
        self.plot.installEventFilter(self)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.run_button)
        layout.addWidget(self.plot)
        self.setWidget(widget)
        self._position_minimap()
        
        self.scatter: pg.ScatterPlotItem | None = None
        self.scatter_colors: np.ndarray | None = None
        
        self.bus.edgesChanged.connect(self._on_edges_changed)

    def eventFilter(self, obj, event):
        if obj is self.plot and event.type() == QEvent.Resize:
            self._position_minimap()
        return super().eventFilter(obj, event)

    def _position_minimap(self):
        if not self.minimap: return
        pw, mm = self.plot.size(), self.minimap.size()
        self.minimap.move(pw.width() - mm.width() - 10, 10)
        self.minimap.raise_()

    def _goto_position(self, x: float, y: float):
        xr, yr = self.view.viewRange()
        dx, dy = (xr[1] - xr[0]) / 2, (yr[1] - yr[0]) / 2
        self.view.setRange(xRange=(x - dx, x + dx), yRange=(y - dy, y + dy), padding=0)

    def _on_minimap_rect_moved(self):
        r, sz = self.minimap_rect.pos(), self.minimap_rect.size()
        self.view.setRange(xRange=(r.x(), r.x() + sz.x()), yRange=(r.y(), r.y() + sz.y()), padding=0)
        
    def start_simulation(self):
        if self.fa2_layout and not self.timer.isActive():
            self.timer.start()
            self.run_button.setText("Pause Layout")
            print("Layout simulation started.")
            QTimer.singleShot(self.auto_stop_ms, self.stop_simulation)

    def stop_simulation(self):
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Resume Layout")
            print("Layout simulation paused.")
        self._update_minimap_view()

    def on_run_button_clicked(self):
        self.auto_stop_ms = 60000 * 5
        if self.timer.isActive():
            self.stop_simulation()
        else:
            self.start_simulation()

    def set_model(self, session: SessionModel | None):
        self.stop_simulation()
        self.plot.clear()
        if self.minimap_scatter:
            self.minimap.plotItem.removeItem(self.minimap_scatter)
        self.session, self.fa2_layout, self.scatter, self.scatter_colors, self.minimap_scatter = None, None, None, None, None
        if not session:
            self.run_button.setEnabled(False)
            return
        self.session = session
        print("Initializing ForceAtlas2 layout for hyperedges...")
        edges = list(session.hyperedges)
        overlap_data = { rk: {ck: len(session.hyperedges[rk] & session.hyperedges[ck]) for ck in edges} for rk in edges }
        self.fa2_layout = HyperedgeForceAtlas2(overlap_data, session)
        num_hyperedges = len(self.fa2_layout.names)
        self.scatter_colors = np.array(['#808080'] * num_hyperedges, dtype=object)
        initial_pos = np.array([self.fa2_layout.positions[name] for name in self.fa2_layout.names])
        self.scatter = pg.ScatterPlotItem(
            pos=initial_pos,
            size=self.fa2_layout.node_sizes,
            brush=[pg.mkBrush(c) for c in self.scatter_colors],
            pen=None,
            pxMode=True,
            useOpenGL=True,
            data=self.fa2_layout.names 
        )
        self.plot.addItem(self.scatter)
        self._update_minimap_view()
        self._position_minimap()
        self.color_map = session.edge_colors.copy() 
        self.run_button.setEnabled(True)
        self.start_simulation()

    def _update_frame(self):
        if not self.fa2_layout or not self.scatter:
            return
        self.fa2_layout.step(iterations=1)
        self._refresh_scatter_plot()
        
    def _refresh_scatter_plot(self):
        """Updates the scatter plot using the current state of the fa2_layout."""
        if not self.fa2_layout or not self.scatter or self.scatter_colors is None:
            return
        
        # Get the current positions in the correct order
        current_pos = np.array([self.fa2_layout.positions[name] for name in self.fa2_layout.names])
        brushes = [pg.mkBrush(c) for c in self.scatter_colors]
        
        self.scatter.setData(
            pos=current_pos,
            brush=brushes,
            size=self.fa2_layout.node_sizes  # <--- THE FIX: Add this line back in
        )
            # Update minimap
        if self.minimap_scatter is None:
            self.minimap_scatter = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush('w'), size=3, pxMode=True, useOpenGL=True,
            )
            self.minimap.plotItem.addItem(self.minimap_scatter)
        self.minimap_scatter.setData(pos=current_pos)

        self._update_minimap_view()

    def _update_minimap_view(self):
        if not self.fa2_layout:
            return
        positions = np.array(list(self.fa2_layout.positions.values()))
        if positions.size == 0: return
        xmin, ymin = positions.min(axis=0)
        xmax, ymax = positions.max(axis=0)
        self.minimap.plotItem.setXRange(xmin, xmax, padding=0.1)
        self.minimap.plotItem.setYRange(ymin, ymax, padding=0.1)
        xr, yr = self.view.viewRange()
        self.minimap_rect.setPos(xr[0], yr[0])
        self.minimap_rect.setSize([xr[1]-xr[0], yr[1]-yr[0]])

    def _on_lasso_select(self, pts: list[QPointF]):
        """Handles a finished lasso selection to select hyperedges."""
        if not self.fa2_layout:
            return
        ordered_names = self.fa2_layout.names
        current_pos = np.array([self.fa2_layout.positions[name] for name in ordered_names])
        poly = [(p.x(), p.y()) for p in pts]
        path = MplPath(poly)
        contained_indices = np.nonzero(path.contains_points(current_pos))[0]
        selected_names = [ordered_names[i] for i in contained_indices]
        if selected_names:
            print(f"Lasso selected {len(selected_names)} hyperedges.")
            self.bus.set_edges(selected_names)

    def _on_scene_mouse_clicked(self, ev):
        if not ev.button() == Qt.LeftButton:
            return
        if not self.scatter or not self.session:
            return
        pos = self.view.mapSceneToView(ev.scenePos())
        pts = self.scatter.pointsAt(pos)
        if pts:
            hyperedge_name = pts[0].data()
            if hyperedge_name:
                print(f"Clicked on hyperedge: {hyperedge_name}")
                self.bus.set_edges([hyperedge_name])
                ev.accept()

    def _on_edges_changed(self, names: list[str]):
        if not self.fa2_layout or self.scatter_colors is None:
            return
        self.scatter_colors[:] = '#808080'
        name_to_index = {name: i for i, name in enumerate(self.fa2_layout.names)}
        for name in names:
            if name in name_to_index:
                idx = name_to_index[name]
                color = self.color_map.get(name, 'yellow')
                self.scatter_colors[idx] = color
        self._refresh_scatter_plot()

    def closeEvent(self, event):
        self.stop_simulation()
        super().closeEvent(event)