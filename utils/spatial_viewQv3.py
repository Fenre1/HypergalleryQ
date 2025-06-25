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
    QGraphicsEllipseItem, 
    QGraphicsItem,
    QGraphicsLineItem,
)

from PyQt5.QtGui import QPalette, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, pyqtSignal as Signal, QTimer, QEvent, QLineF
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
import umap
from matplotlib.path import Path as MplPath
from .selection_bus import SelectionBus
from .session_model import SessionModel
from .image_popup import show_image_metadata
from .fa2_layout import HyperedgeForceAtlas2
# from .fast_sim_engine import SimulationEngine # Assuming this is not used here
from math import cos, sin, pi
from .similarity import SIM_METRIC


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
        self._radial_layout_cache: tuple[dict, list] | None = None
        self.hyperedgeItems: dict[str, QGraphicsEllipseItem] = {}
        self.imageItems: dict[tuple[str,int], QGraphicsPixmapItem] = {}
        self.session: SessionModel | None = None
        self.color_map: dict[str, str] = {}

        self.MIN_HYPEREDGE_DIAMETER = 0.5
        self.NODE_SIZE_SCALER = 0.1
        self.fa2_layout: HyperedgeForceAtlas2 | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._update_frame)

        self.auto_stop_ms = 10000
        self.run_button = QPushButton("Pause Layout")
        self.run_button.clicked.connect(self.on_run_button_clicked)
        self.run_button.setEnabled(False)

        self.view = LassoViewBox()
        self.view.sigRangeChanged.connect(self._update_minimap_view)
        ## NOTE: Connect view changes to update the image layer
        self.view.sigRangeChanged.connect(self._update_image_layer)
        self.view.sigRangeChanged.connect(self._refresh_scatter_plot) 
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
        self.minimap_rect = pg.RectROI([0, 0], [1, 1], pen=pen, movable=False, resizable=False) # Not user-movable
        self.minimap_view.addItem(self.minimap_rect)
        self.plot.installEventFilter(self)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.run_button)
        layout.addWidget(self.plot)
        self.setWidget(widget)
        self._position_minimap()
        self.image_links: list[pg.PlotCurveItem] = []
        
        self.zoom_threshold = 70
        
        ## NOTE: This factor is to place nodes on the circumference (1.0) or slightly outside (>1.0)
        self.radial_placement_factor = 1.1

        self._selected_edges: list[str] = []
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

    def _edge_index(self, name: str) -> int:
        return self.fa2_layout.names.index(name)

      
    def _compute_radial_layout(self, sel_name: str):
        """Return a radial layout for image nodes around hyperedges.

        The function places images of the selected hyperedge evenly along its
        perimeter based on similarity to the hyperedge centroid. For every other
        hyperedge that shares images with the selected one, shared images are
        used as angular anchors and the remaining images are distributed between
        them according to similarity.
        """

        offsets: dict[tuple[str, int], np.ndarray] = {}
        links: list[tuple[tuple[str, int], tuple[str, int]]] = []

        if not self.session or not self.fa2_layout:
            return offsets, links

        features = self.session.features

        # ---- Selected hyperedge placement ---------------------------------
        sel_indices = list(self.session.hyperedges.get(sel_name, []))
        if not sel_indices:
            return offsets, links

        sel_center = self.fa2_layout.positions[sel_name]
        sel_radius = (
            self.fa2_layout.node_sizes[self._edge_index(sel_name)] / 2.0
        ) * self.radial_placement_factor

        centroid = self.session.hyperedge_avg_features[sel_name]
        sims = SIM_METRIC(features[sel_indices], centroid.reshape(1, -1)).flatten()
        order = np.argsort(-sims)

        for rank, img_idx in enumerate(np.array(sel_indices)[order]):
            ang = pi / 2 - 2 * pi * rank / len(order)
            offsets[(sel_name, img_idx)] = np.array([cos(ang), sin(ang)]) * sel_radius

        # ---- Linked hyperedge placement -----------------------------------
        for tgt in self.session.hyperedges:
            if tgt == sel_name:
                continue
            tgt_indices = list(self.session.hyperedges.get(tgt, []))
            shared = [i for i in tgt_indices if i in sel_indices]
            if not shared:
                continue

            tgt_center = self.fa2_layout.positions[tgt]
            tgt_radius = (
                self.fa2_layout.node_sizes[self._edge_index(tgt)] / 2.0
            ) * self.radial_placement_factor

            anchors = []
            for idx in shared:
                pos_on_sel = sel_center + offsets[(sel_name, idx)]
                vec = pos_on_sel - tgt_center
                if np.linalg.norm(vec) < 1e-6:
                    vec = np.array([1.0, 0.0])
                unit = vec / np.linalg.norm(vec)
                offsets[(tgt, idx)] = unit * tgt_radius
                links.append(((sel_name, idx), (tgt, idx)))
                angle = np.arctan2(unit[1], unit[0])
                anchors.append({"id": idx, "angle": angle, "feat": features[idx]})

            anchors.sort(key=lambda a: a["angle"])
            n_anchors = len(anchors)
            remaining = [i for i in tgt_indices if i not in shared]
            if not remaining:
                continue

            if n_anchors == 1:
                anchor = anchors[0]
                rem_feats = features[remaining]
                rem_sims = SIM_METRIC(rem_feats, anchor["feat"].reshape(1, -1)).flatten()
                order = np.argsort(-rem_sims)
                start_angle = anchor["angle"] + pi / 4
                arc_span = 1.5 * pi
                for k, idx in enumerate(np.array(remaining)[order]):
                    frac = (k + 1) / (len(order) + 1)
                    ang = start_angle + frac * arc_span
                    offsets[(tgt, idx)] = np.array([cos(ang), sin(ang)]) * tgt_radius
            else:
                seg_lists: dict[int, list[tuple[int, float]]] = {j: [] for j in range(n_anchors)}
                rem_feats = features[remaining]
                for idx, vec in zip(remaining, rem_feats):
                    best_seg, best_score = 0, -1.0
                    for j in range(n_anchors):
                        a1 = anchors[j]
                        a2 = anchors[(j + 1) % n_anchors]
                        s1 = SIM_METRIC(vec.reshape(1, -1), a1["feat"].reshape(1, -1))[0, 0]
                        s2 = SIM_METRIC(vec.reshape(1, -1), a2["feat"].reshape(1, -1))[0, 0]
                        score = max(s1, s2)
                        if score > best_score:
                            best_score, best_seg = score, j
                    seg_lists[best_seg].append((idx, best_score))

                for j in range(n_anchors):
                    items = seg_lists.get(j, [])
                    if not items:
                        continue
                    items.sort(key=lambda x: x[1], reverse=True)
                    start_a = anchors[j]
                    end_a = anchors[(j + 1) % n_anchors]
                    angular_dist = (end_a["angle"] - start_a["angle"] + 2 * pi) % (2 * pi)
                    for k, (idx, _) in enumerate(items):
                        frac = (k + 1) / (len(items) + 1)
                        ang = start_a["angle"] + frac * angular_dist
                        offsets[(tgt, idx)] = np.array([cos(ang), sin(ang)]) * tgt_radius

        return offsets, links


    

            
    def _update_image_layer(self):
        # 1) Nothing to do if no layout or no selection
        if self._radial_layout_cache is None:
            # Clear any existing items if the cache was just invalidated
            for circ in self.imageItems.values():
                if circ.scene(): circ.scene().removeItem(circ)
            self.imageItems.clear()
            for ln in self.image_links:
                if ln.scene(): ln.scene().removeItem(ln)
            self.image_links.clear()
            return

        # 2) Zoom-out cutoff: clear everything
        xr, _ = self.view.viewRange()
        print('zoom',(xr[1] - xr[0]))
        if (xr[1] - xr[0]) > self.zoom_threshold:
            for circ in self.imageItems.values():
                if circ.scene(): circ.scene().removeItem(circ)
            self.imageItems.clear()
            for ln in self.image_links:
                if ln.scene(): ln.scene().removeItem(ln)
            self.image_links.clear()
            return

        # 3) Unpack your cached offsets & cross-links
        relative_layout, links = self._radial_layout_cache
        alive = set()

        # 4) Create OR MOVE each small circle
        xyz = 0
        for (edge_name, img_idx), offset in relative_layout.items():
            key = (edge_name, img_idx)
            alive.add(key)

            if key not in self.imageItems:
                # create a fixed-pixel circle
                r = 4  # radius in px
                circ = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                circ.setPen(pg.mkPen('k'))                # black border
                circ.setBrush(pg.mkBrush('w'))            # white fill
                circ.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                self.imageItems[key] = circ
                self.view.scene().addItem(circ) # Add the circle directly to the scene

            # THIS IS THE CORE FIX: This logic must be INSIDE the loop.
            # It calculates and sets the position for the CURRENT item in the loop.
            # 1. Get the parent hyperedge's data position.
            parent_data_pos = self.fa2_layout.positions[edge_name]
            # 2. Calculate the absolute data position of the small circle.
            abs_data_pos = parent_data_pos + offset
            # 3. Map this absolute data position to a scene (pixel) position.
            scene_pos = self.view.mapViewToScene(QPointF(*abs_data_pos))
            # 4. Set the circle's position in the scene.
            self.imageItems[key].setPos(scene_pos)
            if xyz == 0:
                print('sp',scene_pos)
                xyz+=1

        # 5) Remove any stale circles
        for key in list(self.imageItems):
            if key not in alive:
                item = self.imageItems.pop(key)
                if item.scene():
                    item.scene().removeItem(item)

        # 6) Clear old link-items
        for ln in self.image_links:
            if ln.scene():
                ln.scene().removeItem(ln)
        self.image_links.clear()

        # 7) Draw new links as scene-space lines (This part was already correct)
        pen = pg.mkPen(color=(255, 255, 255, 150), width=1) # Reduced width for clarity
        zyc = 0
        for (e1, i1), (e2, i2) in links:
            # compute data-space endpoints
            # Ensure we don't try to access a key that might not be in the layout (edge case)
            if (e1, i1) not in relative_layout or (e2, i2) not in relative_layout:
                continue
                
            d1 = self.fa2_layout.positions[e1] + relative_layout[(e1, i1)]
            d2 = self.fa2_layout.positions[e2] + relative_layout[(e2, i2)]
            
            # map once into scene coords
            p1 = self.view.mapViewToScene(QPointF(*d1))
            p2 = self.view.mapViewToScene(QPointF(*d2))
            if zyc == 0:
                print('d',d1,d2)
                print('p',p1,p2)
                zyc+=1
            # create a line in scene
            line = QGraphicsLineItem(QLineF(p1, p2))
            line.setPen(pen)
            line.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            self.view.scene().addItem(line)
            self.image_links.append(line)
            
    def _goto_position(self, x: float, y: float):
        xr, yr = self.view.viewRange()
        dx, dy = (xr[1] - xr[0]) / 2, (yr[1] - yr[0]) / 2
        self.view.setRange(xRange=(x - dx, x + dx), yRange=(y - dy, y + dy), padding=0)

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
        # remove old hyperedges
        for item in self.hyperedgeItems.values():
            item.scene().removeItem(item)
        self.hyperedgeItems.clear()
        # remove old images
        for pix in self.imageItems.values():
            pix.scene().removeItem(pix)
        self.imageItems.clear()
        # remove old links
        for ln in self.image_links:
            self.plot.removeItem(ln)
        self.image_links.clear()        
        if self.minimap_scatter:
            self.minimap.plotItem.removeItem(self.minimap_scatter)

        # Reset state
        self.session = None
        self.fa2_layout = None
        self.hyperedgeItems = {}
        self.imageItems = {}
        self.minimap_scatter = None

        if not session:
            self.run_button.setEnabled(False)
            return

        self.session = session
        self.color_map = session.edge_colors.copy()

        # Initialize FA2 layout
        edges = list(session.hyperedges)
        overlap_data = {
            rk: {ck: len(session.hyperedges[rk] & session.hyperedges[ck]) for ck in edges}
            for rk in edges
        }
        self.fa2_layout = HyperedgeForceAtlas2(overlap_data, session)

        # Compute initial positions and node sizes
        names = self.fa2_layout.names
        initial_pos = np.array([self.fa2_layout.positions[name] for name in names])
        counts = np.array([np.sqrt(len(session.hyperedges[name])) for name in names])
        sizes = np.maximum(counts * self.NODE_SIZE_SCALER, self.MIN_HYPEREDGE_DIAMETER)
        self.fa2_layout.node_sizes = sizes

        for name, (x,y), diameter in zip(names, initial_pos, sizes):
            r = diameter / 2.0
            # Create an ellipse centered at (0,0) with a given radius in DATA coordinates
            ellipse = QGraphicsEllipseItem(-r, -r, diameter, diameter) 
            ellipse.setPen(pg.mkPen(self.color_map[name]))
            ellipse.setBrush(pg.mkBrush(self.color_map[name]))
            
            # Set its position in DATA coordinates
            ellipse.setPos(x, y)
            
            # REMOVE the ItemIgnoresTransformations flag
            # ellipse.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            
            # ADD the ellipse to the VIEWBOX, not the scene.
            self.view.addItem(ellipse) 
            self.hyperedgeItems[name] = ellipse


        # for name, (x,y), diameter in zip(names, initial_pos, sizes):
        #     r = diameter/2
        #     ellipse = QGraphicsEllipseItem(-r,-r, diameter, diameter)
        #     ellipse.setPen(pg.mkPen(self.color_map[name]))
        #     ellipse.setBrush(pg.mkBrush(self.color_map[name]))
        #     ellipse.setPos(x, y)
        #     print('nodexy',x,y)
        #     # hyperedges should scale/pan with the view:
        #     ellipse.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)            
        #     self.view.scene().addItem(ellipse)
        #     self.hyperedgeItems[name] = ellipse



        # Set up minimap and start
        self._update_minimap_view()
        self._position_minimap()
        self.run_button.setEnabled(True)
        self.start_simulation()


    def _update_frame(self):
        if not self.fa2_layout:
            return
        self.fa2_layout.step(iterations=1)
        self._refresh_scatter_plot()
        
        ## FIX: Call the image layer update so radial nodes move with their parents.
        self._update_image_layer()


    def _refresh_scatter_plot(self):
        if not self.fa2_layout:
            return
        # 1) Re‐position each hyperedge
        for name, ellipse in self.hyperedgeItems.items():
            x, y = self.fa2_layout.positions[name]
            # Just set the data position. PyQtGraph handles the transform.
            ellipse.setPos(x, y) 


        # for name, ellipse in self.hyperedgeItems.items():
        #     x, y = self.fa2_layout.positions[name]
        #     # map the data‐space point into the SCENE (pixel) coordinates
        #     scene_pt = self.view.mapViewToScene(QPointF(x, y))
        #     ellipse.setPos(scene_pt)
        # 2) And update the minimap as before…
        positions = np.array([self.fa2_layout.positions[n] for n in self.fa2_layout.names])
        if self.minimap_scatter is None:
            self.minimap_scatter = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush('w'), size=3, pxMode=True, useOpenGL=True
            )
            self.minimap.plotItem.addItem(self.minimap_scatter)
        self.minimap_scatter.setData(pos=positions)
        self._update_minimap_view()
            
        ## FIX: The image layer update is now triggered by the animation frame or by selection/view changes.
        ## We removed the call from here to avoid redundancy and put it in _update_frame.
        # self._update_image_layer() # This call is moved to _update_frame

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
        self.minimap_rect.setPos(QPointF(xr[0], yr[0]))
        self.minimap_rect.setSize(QPointF(xr[1]-xr[0], yr[1]-yr[0]))

    def _on_lasso_select(self, pts: list[QPointF]):
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
        
        # Find items at the click position in the scene
        items = self.plot.scene().items(ev.scenePos())
        
        clicked_edge_name = None
        for item in items:
            # Check if the clicked item is one of our hyperedge ellipses
            for name, ellipse in self.hyperedgeItems.items():
                if item is ellipse:
                    clicked_edge_name = name
                    break
            if clicked_edge_name:
                break
        
        if clicked_edge_name:
            print(f"Clicked on hyperedge: {clicked_edge_name}")
            self.bus.set_edges([clicked_edge_name])
            ev.accept()
        else:
            # If click is not on a node, clear selection
            if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
                self.bus.set_edges([])


          
    def _on_edges_changed(self, names: list[str]):
        if not self.fa2_layout:
            return

        # 1) Reset all hyperedge colors to gray
        for name, ellipse in self.hyperedgeItems.items():
            pen = pg.mkPen('#808080')
            brush = pg.mkBrush('#808080')
            ellipse.setPen(pen)
            ellipse.setBrush(brush)

        # 2) Highlight selected edges
        for name in names:
            ellipse = self.hyperedgeItems.get(name)
            if ellipse:
                color = self.color_map.get(name, 'yellow')
                pen = pg.mkPen(color)
                brush = pg.mkBrush(color)
                ellipse.setPen(pen)
                ellipse.setBrush(brush)

        # 3) Recompute and draw image layout
        if len(names) == 1:
            self._radial_layout_cache = self._compute_radial_layout(names[0])
        else:
            self._radial_layout_cache = None

        # Refresh positions and images/links
        self._refresh_scatter_plot()
        self._update_image_layer()


    

    def closeEvent(self, event):
        self.stop_simulation()
        super().closeEvent(event)