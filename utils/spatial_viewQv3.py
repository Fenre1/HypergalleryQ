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
        self.session: SessionModel | None = None
        self.color_map: dict[str, str] = {}

        self.MIN_HYPEREDGE_DIAMETER = 30.0 
        self.NODE_SIZE_SCALER = 2.5
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

        self.scatter: pg.ScatterPlotItem | None = None
        self.scatter_colors: np.ndarray | None = None

        ## NOTE: Image nodes should be small and fixed-size, so pxMode=True is correct here.
        self.image_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush('w'), size=8, pxMode=True)
        self.plot.addItem(self.image_scatter)
        self.image_links: list[pg.PlotCurveItem] = []
        
        ## FIX: Set a reasonable zoom threshold in data coordinates.
        ## This value will depend on the scale of your FA2 layout. You may need to adjust it.
        self.zoom_threshold = 200.0
        
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

    def _clear_image_layer(self):
        self.image_scatter.setData(pos=np.empty((0, 2)))
        for ln in self.image_links:
            self.plot.removeItem(ln)
        self.image_links.clear()

      
    def _compute_radial_layout(self, sel_name: str):
        relative_layout: dict[tuple[str, int], np.ndarray] = {}
        links_by_ref: list[tuple[tuple[str, int], tuple[str, int]]] = []

        if not self.session or not self.fa2_layout:
            return relative_layout, links_by_ref

        features = self.session.features

        # We still need the centers for the direction calculation, but we won't store them.
        sel_center = self.fa2_layout.positions[sel_name]
        sel_diameter = self.fa2_layout.node_sizes[self._edge_index(sel_name)]
        sel_r = (sel_diameter / 2.0) * self.radial_placement_factor
        sel_indices = list(self.session.hyperedges.get(sel_name, []))
        if not sel_indices:
            return relative_layout, links_by_ref

        # ... (rest of the similarity calculation is identical) ...
        centroid = self.session.hyperedge_avg_features[sel_name]
        sims = SIM_METRIC(features[sel_indices], centroid.reshape(1, -1)).flatten()
        order = np.argsort(-sims)

        for rank, idx in enumerate(np.array(sel_indices)[order]):
            ang = pi/2 - 2 * pi * rank / len(order)
            # Store the OFFSET vector, not the absolute position
            offset = np.array([cos(ang), sin(ang)]) * sel_r
            relative_layout[(sel_name, idx)] = offset

        # ... (The logic for linked hyperedges also now stores offsets) ...
        for tgt in self.session.hyperedges:
            if tgt == sel_name: continue
            tgt_indices = list(self.session.hyperedges.get(tgt, []))
            shared_ids = [i for i in tgt_indices if i in sel_indices]
            if not shared_ids: continue

            center_t = self.fa2_layout.positions[tgt]
            tgt_diameter = self.fa2_layout.node_sizes[self._edge_index(tgt)]
            r_t = (tgt_diameter / 2.0) * self.radial_placement_factor

            anchors = []
            for img_idx in shared_ids:
                # Reconstruct absolute position of source node to get the direction
                pos_on_selected = sel_center + relative_layout[(sel_name, img_idx)]
                direction_vec = pos_on_selected - center_t
                if np.linalg.norm(direction_vec) < 1e-6: direction_vec = np.array([1.0, 0.0])
                
                norm_direction = direction_vec / np.linalg.norm(direction_vec)
                # Store the OFFSET for the target node
                offset_on_target = norm_direction * r_t
                relative_layout[(tgt, img_idx)] = offset_on_target
                
                # Store links by REFERENCE ((parent, id), (parent, id))
                links_by_ref.append( ((sel_name, img_idx), (tgt, img_idx)) )
                
                angle = np.arctan2(norm_direction[1], norm_direction[0])
                anchors.append({'id': img_idx, 'angle': angle, 'feat': features[img_idx]})

            # Sort anchors by their angle to define the arcs between them
            anchors.sort(key=lambda a: a['angle'])
            n_anchors = len(anchors)
            
            # --- Distribute Remaining Nodes into Arcs ---
            remaining_ids = [i for i in tgt_indices if i not in shared_ids]
            if remaining_ids:
                # Handle single-anchor case separately
                if n_anchors == 1:
                    anchor = anchors[0]
                    rem_feats = features[remaining_ids]
                    rem_sims = SIM_METRIC(rem_feats, anchor['feat'].reshape(1,-1)).flatten()
                    order = np.argsort(-rem_sims)
                    
                    # Fill a 270-degree arc opposite the anchor
                    start_angle = anchor['angle'] + pi/4
                    arc_span = 1.5 * pi # 270 degrees
                    for k, idx in enumerate(np.array(remaining_ids)[order]):
                        frac = (k + 1) / (len(order) + 1)
                        ang = start_angle + frac * arc_span
                        pos = center_t + np.array([cos(ang), sin(ang)]) * r_t
                        # layout[(tgt, idx)] = pos
                        offset = np.array([cos(ang), sin(ang)]) * r_t
                        relative_layout[(tgt, idx)] = offset


                else: # General case with 2+ anchors
                    seg_lists: dict[int, list[tuple[int, float]]] = {j: [] for j in range(n_anchors)}
                    rem_feats = features[remaining_ids]
                    
                    for idx, vec in zip(remaining_ids, rem_feats):
                        best_seg, best_score = -1, -1.0
                        for j in range(n_anchors):
                            a1 = anchors[j]
                            a2 = anchors[(j + 1) % n_anchors]
                            s1 = SIM_METRIC(vec.reshape(1,-1), a1['feat'].reshape(1,-1))[0,0]
                            s2 = SIM_METRIC(vec.reshape(1,-1), a2['feat'].reshape(1,-1))[0,0]
                            score = max(s1, s2)
                            if score > best_score:
                                best_score, best_seg = score, j
                        seg_lists[best_seg].append((idx, best_score))

                    for j in range(n_anchors):
                        items_in_segment = seg_lists.get(j, [])
                        if not items_in_segment: continue
                        
                        items_in_segment.sort(key=lambda x: x[1], reverse=True)
                        
                        start_anchor = anchors[j]
                        end_anchor = anchors[(j + 1) % n_anchors]
                        
                        # Calculate angular distance, handling wrapping around -pi/pi
                        angular_dist = (end_anchor['angle'] - start_anchor['angle'] + 2*pi) % (2*pi)

                        for k, (idx, _) in enumerate(items_in_segment):
                            # Interpolate angle within the arc segment
                            frac = (k + 1) / (len(items_in_segment) + 1)
                            ang = start_anchor['angle'] + frac * angular_dist
                            pos = center_t + np.array([cos(ang), sin(ang)]) * r_t
                            # layout[(tgt, idx)] = pos
                            offset = np.array([cos(ang), sin(ang)]) * r_t
                            relative_layout[(tgt, idx)] = offset

        return relative_layout, links_by_ref


    

          
    def _update_image_layer(self):
        # This function is now responsible for using the cache to build final positions
        if self._radial_layout_cache is None:
            self._clear_image_layer()
            return

        xr, _ = self.view.viewRange()
        view_width = xr[1] - xr[0]
        if view_width > self.zoom_threshold:
            self._clear_image_layer()
            return
        
        relative_layout, links_by_ref = self._radial_layout_cache
        if not relative_layout:
            self._clear_image_layer()
            return

        # FIX 2: Reconstruct absolute positions every frame. This is very fast.
        final_positions = {}
        for (parent_name, img_idx), offset in relative_layout.items():
            parent_center = self.fa2_layout.positions[parent_name]
            final_positions[(parent_name, img_idx)] = parent_center + offset

        # Draw the nodes
        pos_array = np.array(list(final_positions.values()))
        self._clear_image_layer()
        self.image_scatter.setData(pos=pos_array)
        
        # Draw the links using the reconstructed positions
        link_pen = pg.mkPen(color=(255, 255, 255, 150), width=1)
        for ref1, ref2 in links_by_ref:
            p1 = final_positions[ref1]
            p2 = final_positions[ref2]
            ln = pg.PlotCurveItem(x=[p1[0], p2[0]], y=[p1[1], p2[1]], pen=link_pen)
            self.plot.addItem(ln)
            self.image_links.append(ln)

    

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
        self.plot.clear()
        self.plot.addItem(self.image_scatter)
        
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

        ## ------------------------------------------------------------------
        ## FIX 1: Correctly calculate and apply scalable node sizes
        ## ------------------------------------------------------------------
        # Using sqrt to prevent giant hyperedges from becoming too dominant
        content_based_sizes = np.array([np.sqrt(len(session.hyperedges[name])) for name in self.fa2_layout.names])
        scaled_sizes = content_based_sizes * self.NODE_SIZE_SCALER
        final_node_sizes = np.maximum(scaled_sizes, self.MIN_HYPEREDGE_DIAMETER)
        self.fa2_layout.node_sizes = final_node_sizes
        ## ------------------------------------------------------------------

        # Create the scatter plot with pxMode=False
        self.scatter = pg.ScatterPlotItem(
            pos=initial_pos,
            size=self.fa2_layout.node_sizes,  # Use the correctly calculated sizes
            brush=[pg.mkBrush(c) for c in self.scatter_colors],
            pen=None,
            pxMode=True,  # <-- CRITICAL: Set to False for scalable nodes
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
        
        ## FIX: Call the image layer update so radial nodes move with their parents.
        self._update_image_layer()


    def _refresh_scatter_plot(self):
        if not self.fa2_layout or not self.scatter or self.scatter_colors is None:
            return
        
        current_pos = np.array([self.fa2_layout.positions[name] for name in self.fa2_layout.names])
        brushes = [pg.mkBrush(c) for c in self.scatter_colors]
        
        self.scatter.setData(
            pos=current_pos,
            brush=brushes,
            size=self.fa2_layout.node_sizes # This is now world size
        )
        
        if self.minimap_scatter is None:
            self.minimap_scatter = pg.ScatterPlotItem(
                pen=None, brush=pg.mkBrush('w'), size=3, pxMode=True, useOpenGL=True,
            )
            self.minimap.plotItem.addItem(self.minimap_scatter)
        self.minimap_scatter.setData(pos=current_pos)

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
        else:
            # If click is not on a node, maybe clear selection
            self.bus.set_edges([])


          
    def _on_edges_changed(self, names: list[str]):
        if not self.fa2_layout or self.scatter_colors is None:
            return
            
        self._selected_edges = names
        self.scatter_colors[:] = '#808080'
        name_to_index = {name: i for i, name in enumerate(self.fa2_layout.names)}
        for name in names:
            if name in name_to_index:
                idx = name_to_index[name]
                color = self.color_map.get(name, 'yellow')
                self.scatter_colors[idx] = color
        
        ## FIX: Re-compute and cache the layout ONLY when the selection changes.
        if len(names) == 1:
            self._radial_layout_cache = self._compute_radial_layout(names[0])
        else:
            self._radial_layout_cache = None

        self._refresh_scatter_plot() # Redraw hyperedges with new colors
        self._update_image_layer() # Trigger an initial draw of the new layout

    

    def closeEvent(self, event):
        self.stop_simulation()
        super().closeEvent(event)