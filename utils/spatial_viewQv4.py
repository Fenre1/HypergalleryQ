from __future__ import annotations

import os
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"   # force pyqtgraph to PyQt5
os.environ.pop("QT_API", None)             # avoid other libs nudging Qt differently

import numpy as np
from math import cos, sin, pi
from time import perf_counter
import numba as nb
import umap
from types import SimpleNamespace
import base64
from pathlib import Path
from PyQt5.QtCore import (
    Qt,
    QPointF,
    QEvent,
    pyqtSignal as Signal,
    QUrl,
    QPoint,
    QTimer,
    pyqtSlot,
    QBuffer,
    QIODevice,
)

from PyQt5.QtGui import QPainterPath, QPen, QColor
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QApplication,
    QPushButton, QGraphicsEllipseItem, QToolTip, QLabel, QGraphicsSceneHoverEvent
)

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


from matplotlib.path import Path as MplPath
import math
from sklearn.decomposition import IncrementalPCA
import time

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .similarity import SIM_METRIC       # kept for non‑cosine fallback
from .image_utils import qimage_from_data

THUMB_SIZE = 128



@nb.njit(fastmath=True, cache=True)
def _resolve_overlaps_numba(pos, radii, iterations, strength):
    n = pos.shape[0]
    for itr in range(iterations):
        moved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_d = radii[i] + radii[j]
                if 1e-9 < dist_sq < min_d*min_d:
                    dist = np.sqrt(dist_sq)
                    overlap = (min_d - dist) * strength * 0.5
                    push_x = dx / dist * overlap
                    push_y = dy / dist * overlap
                    pos[i, 0] += push_x
                    pos[i, 1] += push_y
                    pos[j, 0] -= push_x
                    pos[j, 1] -= push_y
                    moved = True
        if not moved:
            break
    return pos


def _resolve_overlaps(positions: np.ndarray, radii: np.ndarray,
                      iterations: int = 10000, strength: float = 1.0) -> np.ndarray:

    pos32   = np.ascontiguousarray(positions, dtype=np.float32)
    radii32 = np.ascontiguousarray(radii,     dtype=np.float32)

    pos_out32 = _resolve_overlaps_numba(pos32, radii32,
                                        iterations, strength)

    return pos_out32.astype(positions.dtype, copy=False)



class _RecalcWorker(QtCore.QObject):
    imageEmbeddingReady = Signal(str, dict)
    layoutReady = Signal(dict)
    radialReady = Signal(str, dict, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: SessionModel | None = None
        self.layout: SimpleNamespace | None = None
        self.image_umap: dict[str, dict[int, np.ndarray]] = {}
        self.limit_images_enabled = False
        self.limit_images_value = 10
        self.limit_edges_enabled = False
        self.limit_edges_value = 10
        self.hidden_edges: set[str] = set()
        self.edge_index: dict[str, int] = {}
        self.radial_placement_factor = SpatialViewQDock.radial_placement_factor

    @pyqtSlot(str)
    def recompute(self, edge_name: str):
        session = self.session
        if session is None:
            return

        # If an edge_name is provided, recompute its UMAP (existing logic)
        if edge_name:
            feats_norm = session.features_unit
            mapping: dict[int, np.ndarray] = {}
            if edge_name in session.hyperedges:
                idx = list(session.hyperedges[edge_name])
                if idx:
                    emb = umap.UMAP(n_components=2, random_state=42, n_jobs=-1).fit_transform(feats_norm[idx])
                    emb = emb - emb.mean(axis=0)
                    m = np.max(np.linalg.norm(emb, axis=1))
                    if m > 0:
                        emb = emb / m
                    mapping = {i: emb[k] for k, i in enumerate(idx)}
            self.imageEmbeddingReady.emit(edge_name, mapping)
            # We only recomputed one image embedding, so we are done for this path.
            return

        # If edge_name is empty, compute the full hyperedge layout (NEW LOGIC)
        edges = list(session.hyperedges)
        if not edges:
            self.layoutReady.emit({})
            return

        edge_feats = np.stack([session.hyperedge_avg_features[e] for e in edges]).astype(np.float32)
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.8, n_jobs=-1)
        initial_pos = reducer.fit_transform(edge_feats)
        diameters = np.maximum(
            np.array([np.sqrt(len(session.hyperedges[n])) for n in edges]) * SpatialViewQDock.NODE_SIZE_SCALER,
            SpatialViewQDock.MIN_HYPEREDGE_DIAMETER,
        )

        raw_scale = np.max(np.abs(initial_pos))
        if raw_scale == 0:
            raw_scale = 1.0

        pos_for_resolver = initial_pos * (10.0 / raw_scale)
        radii_for_resolver = diameters / 2.0
        resolved = _resolve_overlaps(pos_for_resolver, radii_for_resolver)
        pos = resolved - resolved.mean(axis=0)

        layout = {n: (pos[i], diameters[i]) for i, n in enumerate(edges)}
        self.layoutReady.emit(layout)

    @pyqtSlot(str)
    def compute_radial(self, sel_name: str):
        session, layout = self.session, self.layout
        if session is None or layout is None or sel_name not in layout.names:
            self.radialReady.emit(sel_name, {}, [])
            return

        radius_map = {
            n: (layout.node_sizes[self.edge_index[n]] / 2)
            * self.radial_placement_factor
            for n in layout.names
        }

        sel_full = session.hyperedges[sel_name]
        if not sel_full:
            self.radialReady.emit(sel_name, {}, [])
            return

        if self.limit_images_enabled:
            idx_list = list(sel_full)
            feats = session.features[idx_list]
            avg = session.hyperedge_avg_features[sel_name].reshape(1, -1)
            sims = SIM_METRIC(avg, feats)[0]
            order = np.argsort(sims)
            n_top = self.limit_images_value // 2
            n_bottom = self.limit_images_value - n_top
            chosen = [idx_list[i] for i in order[::-1][:n_top]]
            for i in order[:n_bottom]:
                cand = idx_list[i]
                if cand not in chosen:
                    chosen.append(cand)
                if len(chosen) >= self.limit_images_value:
                    break
            sel_display = set(chosen)
        else:
            sel_display = set(sel_full)

        offsets: dict[tuple[str, int], np.ndarray] = {}
        links: list[tuple[tuple[str, int], tuple[str, int]]] = []
        for idx in sel_display:
            vec = self.image_umap.get(sel_name, {}).get(idx, np.zeros(2))
            offsets[(sel_name, idx)] = vec * radius_map[sel_name]

        inter_counts: dict[str, int] = {}
        shared_sets: dict[str, set[int]] = {}
        for e in session.hyperedges:
            if e == sel_name or e not in layout.names or e in self.hidden_edges:
                continue
            shared = session.hyperedges[e] & sel_full
            if shared:
                inter_counts[e] = len(shared)
                shared_sets[e] = shared

        if self.limit_edges_enabled:
            top_edges = {
                e
                for e, _ in sorted(
                    inter_counts.items(), key=lambda x: x[1], reverse=True
                )[: self.limit_edges_value]
            }
        else:
            top_edges = set(inter_counts)

        for tgt, shared_full in shared_sets.items():
            if self.limit_edges_enabled and tgt not in top_edges:
                continue

            if self.limit_images_enabled:
                shared = shared_full & sel_display
                if not shared:
                    continue
            else:
                shared = shared_full

            for idx in shared:
                vec = self.image_umap.get(tgt, {}).get(idx, np.zeros(2))
                offsets[(tgt, idx)] = vec * radius_map[tgt]
                links.append(((sel_name, idx), (tgt, idx)))

            extras = session.hyperedges[tgt] - shared_full
            if self.limit_images_enabled:
                extras = list(extras)[: self.limit_images_value]
            for idx in extras:
                vec = self.image_umap.get(tgt, {}).get(idx, np.zeros(2))
                offsets[(tgt, idx)] = vec * radius_map[tgt]

        self.radialReady.emit(sel_name, offsets, links)
    
class TooltipManager:
    """Manages a custom QLabel widget to provide persistent tooltips."""
    def __init__(self, parent_widget: QWidget):
        self.tooltip = QLabel(parent_widget)
        self.tooltip.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.tooltip.setStyleSheet(
            "QLabel { background-color: #FFFFE0; color: black; "
            "border: 1px solid black; padding: 2px; }"
        )
        self.tooltip.hide()

    def _as_qpoint(self, pos) -> QPoint:
        """Coerce various point types to ``QPoint``."""
        if isinstance(pos, QPoint):
            return pos
        if hasattr(pos, "toPoint"):
            return pos.toPoint()
        return QPoint(int(pos.x()), int(pos.y()))

    def show(self, global_pos: QPoint, html_content: str):
        self.tooltip.setText(html_content)
        self.tooltip.adjustSize()
        self.tooltip.move(global_pos + QPoint(15, 10))
        self.tooltip.show()


    def hide(self):
        """Hides the tooltip."""
        self.tooltip.hide()

# ---------------------------------------------------------------------------- #
# Helper view‑boxes                                                            #
# ---------------------------------------------------------------------------- #

class LassoViewBox(pg.ViewBox):
    sigLassoFinished = Signal(list)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._drawing = False
        self._path = QPainterPath()
        self._item = None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ShiftModifier:
            self._drawing = True
            self._path = QPainterPath(self.mapToView(ev.pos()))
            pen = QPen(pg.mkColor("y")); pen.setWidth(2)
            pen.setCosmetic(True)
            self._item = QtWidgets.QGraphicsPathItem()
            self._item.setPen(pen)
            self.addItem(self._item); ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drawing:
            self._path.lineTo(self.mapToView(ev.pos()))
            if self._item: self._item.setPath(self._path)
            ev.accept(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drawing and ev.button() == Qt.LeftButton:
            self._drawing = False
            if self._item: self.removeItem(self._item)
            pts = [QPointF(self._path.elementAt(i).x, self._path.elementAt(i).y)
                   for i in range(self._path.elementCount())]
            if len(pts) > 2: self.sigLassoFinished.emit(pts)
            ev.accept(); return
        super().mouseReleaseEvent(ev)


class MiniMapViewBox(pg.ViewBox):
    sigGoto = Signal(float, float)
    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            p = self.mapToView(ev.pos()); self.sigGoto.emit(p.x(), p.y()); ev.accept()
        else: super().mouseClickEvent(ev)


# ---------------------------------------------------------------------------- #
# Graphics items (now simplified)                                              #
# ---------------------------------------------------------------------------- #
# class HyperedgeItem(QGraphicsEllipseItem):
#     """
#     Represents a hyperedge. Tooltip logic is handled by the parent view.
#     It now only stores its name for identification.
#     """
#     def __init__(self, name: str, rect):
#         super().__init__(rect)
#         self.name = name
#         # No longer needs hover events; the main view will handle it.

class HyperedgeItem(QGraphicsEllipseItem):
    """
    Represents a hyperedge.
    This version includes a custom, mathematically accurate `contains` method
    to ensure tooltips only trigger when the cursor is inside the ellipse.
    """
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def contains(self, pos: QPointF) -> bool:
        """
        Overrides the default QGraphicsItem.contains() to use a precise
        mathematical check for whether a point is inside the ellipse, rather
        than just its bounding box.
        """
        rect = self.rect()
        center_x = rect.x() + rect.width() / 2.0
        center_y = rect.y() + rect.height() / 2.0
        rx = rect.width() / 2.0
        ry = rect.height() / 2.0

        if rx <= 0 or ry <= 0:
            return False

        norm_x = pos.x() - center_x
        norm_y = pos.y() - center_y
        
        return ((norm_x**2) / (rx**2)) + ((norm_y**2) / (ry**2)) <= 1


class ImageScatterItem(pg.ScatterPlotItem):
    """
    Represents image nodes. Tooltip logic is handled by the parent view.
    This class is now just a standard ScatterPlotItem.
    """
    # No custom logic needed here anymore.
    pass

# ---------------------------------------------------------------------------- #
# Main dock widget                                                             #
# ---------------------------------------------------------------------------- #
class SpatialViewQDock(QDockWidget):
    requestRecalc = Signal(str)
    requestRadial = Signal(str)    
    MIN_HYPEREDGE_DIAMETER = 0.5
    NODE_SIZE_SCALER       = 0.1
    zoom_threshold         = 400.0
    hyperedge_tooltip_zoom_threshold = 200.0
    image_tooltip_zoom_threshold = 50.0
    radial_placement_factor = 1.1
    highlight_pen_width_ratio = 0.0005
    highlight_anim_base_ratio = 0.01
    highlight_anim_peak_ratio = 0.3

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial Hypergraph", parent)
        self.bus = bus
        self._layout_version: int | None = None   # current data version
        self._rendered_version: int | None = None # last version sent to GPU

        self._highlight_anim_duration = 1000  # ms (1 second)
        self._highlight_anim_steps = 20
        
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setInterval(self._highlight_anim_duration // self._highlight_anim_steps)
        self._highlight_timer.timeout.connect(self._update_highlight_animation)
        
        self._animating_items = []
        self._anim_step_count = 0
        # runtime
        self.session: SessionModel | None = None
        self.fa2_layout: SimpleNamespace | None = None
        self.hyperedgeItems: dict[str, HyperedgeItem] = {}
        self.image_scatter: ImageScatterItem | None = None
        self.link_curve: pg.PlotCurveItem | None = None
        self.selected_scatter: pg.ScatterPlotItem | None = None
        self.selected_links: pg.PlotCurveItem | None = None
        self.minimap_scatter: pg.ScatterPlotItem | None = None
        self._radial_layout_cache: tuple[dict, list] | None = None
        self._radial_cache_by_edge: dict[str, tuple[dict, list]] = {}
        self._abs_pos_cache: dict[tuple[str, int], np.ndarray] = {}
        self._selected_nodes: set[tuple[str, int]] = set()
        self.color_map: dict[str, str] = {}
        self.hidden_edges: set[str] = set()
        self._overview_triplets: dict[str, tuple[int | None, ...]] | None = None

        self._current_edge: str | None = None

        # limit settings
        self.limit_images_enabled = False
        self.limit_images_value = 10
        self.limit_edges_enabled = False
        self.limit_edges_value = 10
        self._use_full = True

        # fast‑similarity pre‑computes
        self._features_norm: np.ndarray | None = None
        self._centroid_norm: dict[str, np.ndarray] = {}
        self._centroid_sim: dict[str, np.ndarray] = {}
        self._image_umap: dict[str, dict[int, np.ndarray]] = {}
        
        # background worker for recomputing embeddings
        self._worker_thread = QtCore.QThread(self)
        self._worker = _RecalcWorker()
        self._worker.moveToThread(self._worker_thread)
        self.requestRecalc.connect(self._worker.recompute)
        self.requestRadial.connect(self._worker.compute_radial)        
        self._worker_thread.start()
        self._worker.imageEmbeddingReady.connect(self._on_worker_image)
        self._worker.layoutReady.connect(self._on_worker_layout)
        self._worker.radialReady.connect(self._on_worker_radial)        
        
        # GUI Setup
        self.view = LassoViewBox(); self.view.setBackgroundColor("#444444")
        self.view.sigRangeChanged.connect(self._update_minimap_view)
        #self.view.sigRangeChanged.connect(self._update_image_layer)      
        self.view.sigRangeChanged.connect(self._update_visibility_on_zoom)

    

        self.view.sigLassoFinished.connect(self._on_lasso)
        self.plot = pg.PlotWidget(viewBox=self.view); self.plot.setBackground("#444444")
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        # NEW: Install event filter to centralize hover logic
        self.plot.scene().installEventFilter(self)

        self.minimap_view = MiniMapViewBox(enableMenu=False)
        self.minimap = pg.PlotWidget(viewBox=self.minimap_view, parent=self.plot)
        self.minimap.setFixedSize(200, 200); self.minimap.hideAxis('bottom'); self.minimap.hideAxis('left')
        self.minimap.setBackground("#333333"); self.minimap.setMouseEnabled(False, False)
        self.minimap_view.sigGoto.connect(self._goto)
        pen = pg.mkPen('r', width=2, cosmetic=True)
        w = QWidget(); l=QVBoxLayout(w); l.addWidget(self.plot)
        self.minimap_rect = pg.RectROI([0,0],[1,1], pen=pen, movable=False, resizable=False)
        self.minimap_view.addItem(self.minimap_rect)
        self.plot.installEventFilter(self) # For resize

        self.tooltip_manager = TooltipManager(self.plot)
        
        self.legend = QWidget(self.plot)
        self.legend.setStyleSheet("background-color: rgba(30,30,30,200); color: white;")
        self.legend_layout = QVBoxLayout(self.legend)
        self.legend_layout.setContentsMargins(4, 4, 4, 4)
        self.legend.hide()

        self.setWidget(w); self._pos_minimap()

        # Bus connections
        # self.bus.edgesChanged.connect(self._on_edges)
        self.bus.edgesChanged.connect(self._on_edges, type=Qt.QueuedConnection)
        self.bus.imagesChanged.connect(self._on_images)

    def eventFilter(self, obj, event: QEvent) -> bool:

        if obj is self.plot and event.type() == QEvent.Resize:
            self._pos_minimap()
            self._pos_legend()

        if obj is self.plot.scene():
            if event.type() == QEvent.GraphicsSceneMouseMove:
                self._update_tooltip(event)
            elif event.type() == QEvent.GraphicsSceneHoverLeave:
                self.tooltip_manager.hide()

        return super().eventFilter(obj, event)

    def _pos_minimap(self):
        pw,mm=self.plot.size(),self.minimap.size()
        self.minimap.move(pw.width()-mm.width()-10,10); self.minimap.raise_()
    
    def _pos_legend(self):
        if self.legend.isHidden():
            return
        pw = self.plot.size()
        self.legend.adjustSize()
        self.legend.move(10, pw.height() - self.legend.height() - 10)
        self.legend.raise_()

    # ------------------------------------------------------------------
    def update_colors(self, mapping: dict[str, str]):
        """Update colors of hyperedge nodes."""
        self.color_map = mapping.copy()
        for name, ell in self.hyperedgeItems.items():
            col = self.color_map.get(name, '#AAAAAA')
            ell.setPen(pg.mkPen(col))
            ell.setBrush(pg.mkBrush(col))

    def show_legend(self, mapping: dict[str, str]):
        self.legend.show()  
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for label, color in mapping.items():
            lab = QLabel(f"<span style='color:{color}'>■</span> {label}")
            self.legend_layout.addWidget(lab)
        self.legend.show()
        self._pos_legend()



    def hide_legend(self):
        self.legend.hide()

    
    def set_edge_visible(self, name: str, visible: bool) -> None:
        """Toggle visibility of a single hyperedge item."""
        if visible:
            changed = name in self.hidden_edges
            self.hidden_edges.discard(name)
        else:
            changed = name not in self.hidden_edges
            self.hidden_edges.add(name)
        if name in self.hyperedgeItems:
            self.hyperedgeItems[name].setVisible(visible)
        self._selected_nodes = {
            k for k in self._selected_nodes if k[0] not in self.hidden_edges
        }
        self._update_mini_scatter(); self._update_minimap_view()
        if changed:
            self._radial_cache_by_edge.clear()
            # self._radial_layout_cache = None  # <<< REMOVE THIS LINE
            self._layout_version = (self._layout_version or 0) + 1
            self._update_image_layer()
        self._update_selected_overlay()

  

        
    def set_hidden_edges(self, names: set[str]) -> None:
        """Replace the set of hidden hyperedges and refresh the view."""
        if names == self.hidden_edges:
            return
        self.hidden_edges = set(names)
        for n, it in self.hyperedgeItems.items():
            it.setVisible(n not in self.hidden_edges)
        self._selected_nodes = {
            k for k in self._selected_nodes if k[0] not in self.hidden_edges
        }
        self._update_mini_scatter(); self._update_minimap_view()
        self._radial_cache_by_edge.clear()
        # self._radial_layout_cache = None  # <<< REMOVE THIS LINE
        self._layout_version = (self._layout_version or 0) + 1
        self._update_image_layer()
        self._update_selected_overlay()

  



    def set_image_limit(self, enabled: bool, value: int):
        self.limit_images_enabled = enabled
        self.limit_images_value = value
        self._worker.limit_images_enabled = enabled # Update worker state
        self._worker.limit_images_value = value
        self._radial_cache_by_edge.clear() # Invalidate all caches

        if self._current_edge:
            # Request the worker to recompute the radial layout for the current edge
            self.requestRadial.emit(self._current_edge)

    def set_intersection_limit(self, enabled: bool, value: int):
        self.limit_edges_enabled = enabled
        self.limit_edges_value = value
        self._worker.limit_edges_enabled = enabled # Update worker state
        self._worker.limit_edges_value = value
        self._radial_cache_by_edge.clear() # Invalidate all caches

        if self._current_edge:
            # Request the worker to recompute the radial layout for the current edge
            self.requestRadial.emit(self._current_edge)



    # ============================================================================ #
    # Session / model setup (No changes here, kept for context)                    #
    # ============================================================================ #
    # def set_model(self, session: SessionModel | None):
    #     self._clear_scene()
    #     if self.session:
    #         try:
    #             self.session.hyperedgeModified.disconnect(self._on_hyperedge_modified)
    #         except TypeError:
    #             pass
    #         try:
    #             self.session.edgeRenamed.disconnect(self._on_edge_renamed)
    #         except TypeError:
    #             pass
    #     self.session = session
    #     self.fa2_layout = None
    #     self._radial_cache_by_edge = {}
    #     self._radial_layout_cache = None
    #     self._layout_version = (self._layout_version or 0) + 1

    #     self._overview_triplets = None
    #     if session is None:
    #         return
    #     self.session.hyperedgeModified.connect(self._on_hyperedge_modified)
    #     self.session.edgeRenamed.connect(self._on_edge_renamed)
    #     self.color_map = session.edge_colors.copy()
    #     # edges = list(session.hyperedges)
    #     # edge_feats = np.stack([session.hyperedge_avg_features[e] for e in edges]).astype(np.float32)
    #     # sizes = np.maximum(np.array([np.sqrt(len(session.hyperedges[n])) for n in edges]) * self.NODE_SIZE_SCALER,
    #     #                    self.MIN_HYPEREDGE_DIAMETER)
    #     # reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.8, n_jobs=-1)
    #     # initial_pos = reducer.fit_transform(edge_feats)

    #     # final_diameters = np.maximum(
    #     #     np.array([np.sqrt(len(session.hyperedges[n])) for n in edges]) * self.NODE_SIZE_SCALER,
    #     #     self.MIN_HYPEREDGE_DIAMETER
    #     # )

    #     # raw_scale = np.max(np.abs(initial_pos))
    #     # if raw_scale == 0: raw_scale = 1.0
        
    #     # pos_for_resolver = initial_pos * (10.0 / raw_scale)        
    #     # radii_for_resolver = final_diameters / 2.0
    #     # resolved_pos = _resolve_overlaps(pos_for_resolver, radii_for_resolver)
    #     # pos = resolved_pos - resolved_pos.mean(axis=0)
    #     # self.edge_index = {n: i for i, n in enumerate(edges)}
    #     # self.fa2_layout = SimpleNamespace(names=edges, node_sizes=final_diameters,
    #     #                                   positions={n: p for n, p in zip(edges, pos)})
    #     # for name, size in zip(edges, final_diameters):
    #     #     r = size / 2
    #     #     # Use the simplified HyperedgeItem
    #     #     ell = HyperedgeItem(name, QtCore.QRectF(-r, -r, size, size))
    #     #     col = self.color_map.get(name, '#AAAAAA')
    #     #     ell.setPen(pg.mkPen(col)); ell.setBrush(pg.mkBrush(col))
    #     #     self.view.addItem(ell)
    #     #     ell.setZValue(-1)
    #     #     self.hyperedgeItems[name] = ell
    #     self._worker.session = self.session
    #     # Request a full recompute by passing an empty string
    #     self.requestRecalc.emit("")

    #     self.hidden_edges.intersection_update(session.hyperedges.keys())
    #     for n, it in self.hyperedgeItems.items():
    #         it.setVisible(n not in self.hidden_edges)

    #     self._features_norm = session.features_unit
    #     self._centroid_norm.clear(); self._centroid_sim.clear()
    #     self._image_umap = session.image_umap or {}




    #     if not self._image_umap:
    #         # Precompute global PCA → UMAP if not already done
    #         if not hasattr(session, "features_pca"):
    #             pca = IncrementalPCA(n_components=64, batch_size=2048)
    #             session.features_pca = pca.fit_transform(session.features.astype(np.float32))
            
    #         if not hasattr(session, "global_xy"):
    #             reducer = umap.UMAP(
    #                 n_components=2,
    #                 n_neighbors=15,
    #                 metric="euclidean",
    #                 random_state=42,
    #                 n_jobs=-1
    #             )
    #             session.global_xy = reducer.fit_transform(session.features_pca)

    #         for edge in edges:
    #             c = session.hyperedge_avg_features[edge].astype(np.float32)
    #             c /= max(np.linalg.norm(c), 1e-9)
    #             self._centroid_norm[edge] = c

    #             idx = list(session.hyperedges[edge])
    #             self._centroid_sim[edge] = self._features_norm[idx] @ c if idx else np.array([])

    #             if idx:
    #                 emb = session.global_xy[idx]
    #                 emb -= emb.mean(0)
    #                 r = np.linalg.norm(emb, axis=1).max()
    #                 if r > 0:
    #                     emb /= r
    #                 self._image_umap[edge] = dict(zip(idx, emb))
    #             else:
    #                 self._image_umap[edge] = {}

    #         session.image_umap = self._image_umap
    #     else:
    #         for edge in edges:
    #             c = session.hyperedge_avg_features[edge].astype(np.float32)
    #             c /= max(np.linalg.norm(c), 1e-9)
    #             self._centroid_norm[edge] = c

    #             idx = list(session.hyperedges[edge])
    #             self._centroid_sim[edge] = self._features_norm[idx] @ c if idx else np.array([])


    #     self._refresh_edges()
    #     self._update_mini_scatter()
    #     self._update_minimap_view()
    #     self._pos_minimap()

    def set_model(self, session: SessionModel | None):
        self._clear_scene()
        if self.session:
            try:
                self.session.hyperedgeModified.disconnect(self._on_hyperedge_modified)
                self.session.edgeRenamed.disconnect(self._on_edge_renamed)
            except TypeError:
                pass # Ignore errors if not connected
                
        self.session = session
        self.fa2_layout = None
        self._radial_cache_by_edge = {}
        self._radial_layout_cache = None
        self._layout_version = (self._layout_version or 0) + 1
        self._overview_triplets = None

        if session is None:
            return

        self.session.hyperedgeModified.connect(self._on_hyperedge_modified)
        self.session.edgeRenamed.connect(self._on_edge_renamed)
        self.color_map = session.edge_colors.copy()

        # --- START OF FIXES ---
        
        # 1. DEFINE `edges` HERE. This was the source of the NameError.
        edges = list(session.hyperedges)

        # 2. Set a loading message and ask the worker to compute the layout.
        self.plot.setTitle("Calculating hypergraph layout...")
        self._worker.session = self.session
        self.requestRecalc.emit("") # Empty string signals a full layout recompute

        # The rest of this function is for pre-calculating things that
        # don't depend on the final layout positions. This is fine to keep here.
        self.hidden_edges.intersection_update(session.hyperedges.keys())

        self._features_norm = session.features_unit
        self._centroid_norm.clear()
        self._centroid_sim.clear()
        self._image_umap = session.image_umap or {}

        if not self._image_umap:
            # Precompute global PCA → UMAP if not already done
            if not hasattr(session, "features_pca"):
                pca = IncrementalPCA(n_components=64, batch_size=2048)
                session.features_pca = pca.fit_transform(session.features.astype(np.float32))
            
            if not hasattr(session, "global_xy"):
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    metric="euclidean",
                    random_state=42,
                    n_jobs=-1
                )
                session.global_xy = reducer.fit_transform(session.features_pca)

            for edge in edges:
                c = session.hyperedge_avg_features[edge].astype(np.float32)
                c /= max(np.linalg.norm(c), 1e-9)
                self._centroid_norm[edge] = c

                idx = list(session.hyperedges[edge])
                self._centroid_sim[edge] = self._features_norm[idx] @ c if idx else np.array([])

                if idx:
                    emb = session.global_xy[idx]
                    emb -= emb.mean(0)
                    r = np.linalg.norm(emb, axis=1).max()
                    if r > 0: emb /= r
                    self._image_umap[edge] = dict(zip(idx, emb))
                else:
                    self._image_umap[edge] = {}
            session.image_umap = self._image_umap
        else:
            for edge in edges:
                c = session.hyperedge_avg_features[edge].astype(np.float32)
                c /= max(np.linalg.norm(c), 1e-9)
                self._centroid_norm[edge] = c

                idx = list(session.hyperedges[edge])
                self._centroid_sim[edge] = self._features_norm[idx] @ c if idx else np.array([])

    # ============================================================================ #
    # Tooltip Logic (REFACTORED and CENTRALIZED)                                   #
    # ============================================================================ #

    def _update_tooltip(self, event: QtGui.QGraphicsSceneMouseEvent):
        """
        Main handler for showing/hiding tooltips with a clear priority:
        1. Image node (small, precise target)
        2. Hyperedge node (large, general target)
        """
        # --- Priority 1: Check for Image Node ---
        if self._should_show_image_tooltip() and self.image_scatter and self.image_scatter.isVisible():
            point = self._get_image_point_at(event.scenePos())
            if point:
                self._show_image_tooltip(point, event.screenPos())
                return  # Found image node, so we are done.

        # --- Priority 2: Check for Hyperedge Node ---
        if self._should_show_hyperedge_tooltip():
            # Broad-phase: get all items under cursor (fast, uses bounding boxes)
            items = self.plot.scene().items(event.scenePos())
            for item in items:
                if isinstance(item, HyperedgeItem):
                    # Narrow-phase: check if cursor is inside the item's actual shape
                    if item.contains(item.mapFromScene(event.scenePos())):
                        self._show_hyperedge_tooltip(item.name, event.screenPos())
                        return  # Found hyperedge, so we are done.

        # --- If nothing relevant was found, hide the tooltip ---
        self.tooltip_manager.hide()

    # def _get_image_point_at(self, scene_pos: QPointF):
    #     """
    #     Performs a precise hit-test on the image scatter plot.
    #     Returns the specific point under the cursor, or None.
    #     """
    #     if not self.image_scatter or self.image_scatter.points().size == 0:
    #         return None

    #     # FIX: Ensure we use coordinates relative to the ViewBox, which is the
    #     # scatter plot's parent. This makes the hit-test reliable.
    #     view_pos = self.view.mapSceneToView(scene_pos)
    #     points = self.image_scatter.pointsAt(view_pos)
    #     if not points:
    #         return None

    #     # The rest of this is a precise check against the circular point area
    #     pixel_size = self.view.pixelSize()
    #     radius = self.image_scatter.opts['size'] * 0.5
    #     radius_in_scene_coords_sq = (radius * pixel_size[0])**2

    #     for p in points:
    #         p_pos = p.pos()
    #         dist_sq = (p_pos.x() - scene_pos.x())**2 + (p_pos.y() - scene_pos.y())**2
    #         if dist_sq <= radius_in_scene_coords_sq:
    #             return p  # Return the first point that is a true match

    #     return None

      
    def _get_image_point_at(self, scene_pos: QPointF):
        # 1) Early exit if the scatter plot is invalid or has no points.
        if not self.image_scatter or len(self.image_scatter.points()) == 0:
            return None

        view_pos = self.view.mapSceneToView(scene_pos)

        # Fast broad-phase: candidates under cursor
        points = self.image_scatter.pointsAt(view_pos)
        if len(points) == 0:
            return None

        # Narrow-phase: consistent units (ViewBox data coords)
        pixel_width_in_view_coords = self.view.pixelWidth()
        radius = self.image_scatter.opts['size'] * 0.5
        radius_in_view_coords_sq = (radius * pixel_width_in_view_coords) ** 2

        for p in points:
            p_pos = p.pos()  # in view coords
            dist_sq = (p_pos.x() - view_pos.x())**2 + (p_pos.y() - view_pos.y())**2
            if dist_sq <= radius_in_view_coords_sq:
                return p
        return None

    


    def _should_show_hyperedge_tooltip(self) -> bool:
        xr, _ = self.view.viewRange()
        return (xr[1] - xr[0]) <= self.hyperedge_tooltip_zoom_threshold

    def _should_show_image_tooltip(self) -> bool:
        xr, _ = self.view.viewRange()
        return (xr[1] - xr[0]) <= self.image_tooltip_zoom_threshold

    def _show_hyperedge_tooltip(self, name: str, screen_pos: QPoint):
        if self._overview_triplets is None:
            self._overview_triplets = self._compute_overview_triplets()

        trip = self._overview_triplets.get(name)
        if not trip or self.session is None: return

        html_parts = [
            f'<img src="{QUrl.fromLocalFile(self.session.im_list[i]).toString()}" '
            f'width="{THUMB_SIZE}" height="{THUMB_SIZE}" style="margin:2px;">'
            for i in trip if i is not None
        ]
        if not html_parts:
            return

        html = f"<b>{name}</b><br>" + "".join(html_parts)
        self.tooltip_manager.show(screen_pos, html)

        # self.tooltip_manager.show(screen_pos, "".join(html_parts))

    def _show_image_tooltip(self, point, screen_pos: QPoint):
        if self.session is None:
            return
        data = point.data()
        if not data:
            return

        idx = data[1]
        url: str
        if not self._use_full and self.session.thumbnail_data:
            if self.session.thumbnails_are_embedded:
                thumb_bytes = self.session.thumbnail_data[idx]
                img = qimage_from_data(thumb_bytes)
                buf = QBuffer()
                buf.open(QIODevice.WriteOnly)
                img.save(buf, "PNG")
                b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
                url = f"data:image/png;base64,{b64}"
                buf.close()
            else:
                tpath = Path(self.session.h5_path).parent / self.session.thumbnail_data[idx]
                url = QUrl.fromLocalFile(str(tpath)).toString()
        else:
            fn = self.session.im_list[idx]
            url = QUrl.fromLocalFile(fn).toString()

        html = f'<img src="{url}" width="{THUMB_SIZE}" height="{THUMB_SIZE}">'
        self.tooltip_manager.show(screen_pos, html)

      
    def _update_highlight_animation(self):
        if self._anim_step_count <= 0 or not self._animating_items:
            self._highlight_timer.stop()
            self._animating_items = []
            return

        # Calculate progress (from 0.0 to 1.0) over the animation duration
        progress = (self._highlight_anim_steps - self._anim_step_count) / self._highlight_anim_steps
        
        # Use a sine wave for a smooth pulse up and down: sin(pi * x)
        pulse_factor = math.sin(math.pi * progress)
        
        for item in self._animating_items:
            diam = item.rect().width()
            base_width = diam * self.highlight_anim_base_ratio
            peak_width = diam * self.highlight_anim_peak_ratio
            current_width = base_width + (peak_width - base_width) * pulse_factor
            pen = QPen(pg.mkColor('w'))
            pen.setWidthF(current_width)
            pen.setCosmetic(False)
            pen.setStyle(Qt.SolidLine)
            item.setPen(pen)

        self._anim_step_count -= 1

    
    # ============================================================================ #
    # Other methods (unchanged)                                                    #
    # ============================================================================ #

    def _clear_scene(self):
        for it in self.hyperedgeItems.values():
            if it.scene(): it.scene().removeItem(it)
        self.hyperedgeItems.clear()
        for item in (self.image_scatter,self.link_curve,
                     self.selected_scatter,self.selected_links):
            if item: self.view.removeItem(item)
        self.image_scatter=self.link_curve=None
        self.selected_scatter=self.selected_links=None
        self._selected_nodes.clear()
        self._abs_pos_cache={}
        self._overview_triplets=None
        self.tooltip_manager.hide()
        if self.minimap_scatter:
            self.minimap.plotItem.removeItem(self.minimap_scatter)
        self.minimap_scatter=None
        self.legend.hide()
        
    def _refresh_edges(self):
        if not self.fa2_layout: return
        for name,ell in self.hyperedgeItems.items():
            x,y=self.fa2_layout.positions[name]; ell.setPos(x,y)
            ell.setVisible(name not in self.hidden_edges)
        self._update_mini_scatter(); self._update_minimap_view()

    def _update_mini_scatter(self):
        if not self.fa2_layout: return
        pos=np.array([
            self.fa2_layout.positions[n]
            for n in self.fa2_layout.names
            if n not in self.hidden_edges
        ])
        if self.minimap_scatter is None:
            self.minimap_scatter=pg.ScatterPlotItem(pen=None,brush=pg.mkBrush('w'),
                                                    size=3,pxMode=True,useOpenGL=True)
            self.minimap.plotItem.addItem(self.minimap_scatter)
        self.minimap_scatter.setData(pos=pos)

    def _update_minimap_view(self):
        if not self.fa2_layout: return
        pos=np.array(list(self.fa2_layout.positions.values()))
        if pos.size==0: return
        xmin,ymin=pos.min(0); xmax,ymax=pos.max(0)
        self.minimap.plotItem.setXRange(xmin,xmax,padding=0.1)
        self.minimap.plotItem.setYRange(ymin,ymax,padding=0.1)
        xr,yr=self.view.viewRange()
        self.minimap_rect.setPos(QPointF(xr[0],yr[0]))
        self.minimap_rect.setSize(QPointF(xr[1]-xr[0],yr[1]-yr[0]))


    def _update_visibility_on_zoom(self):
        """
        A very fast function connected to sigRangeChanged.
        It only toggles the visibility of items, it does not re-calculate them.
        """
        # Get the current horizontal view width
        xr, _ = self.view.viewRange()
        view_width = xr[1] - xr[0]

        # Decide if the detailed view should be visible
        is_visible = view_width <= self.zoom_threshold

        # Toggle visibility (this is a very cheap operation)
        if self.image_scatter:
            self.image_scatter.setVisible(is_visible)
        if self.link_curve:
            self.link_curve.setVisible(is_visible)
        
        # Also hide the tooltip if we zoom out too far
        if not is_visible:
            self.tooltip_manager.hide()



    def _update_image_layer(self):
        if self._layout_version == self._rendered_version:
            return

        if self._radial_layout_cache is None:
            if self.image_scatter: self.image_scatter.hide()
            if self.link_curve: self.link_curve.hide()
            self.tooltip_manager.hide()
            return
        if self.image_scatter: 
            self.image_scatter.show()
        if self.link_curve: 
            self.link_curve.show()

        if self.image_scatter is None:
            self.image_scatter = ImageScatterItem(
                size=8, symbol='o', pxMode=True,
                brush=pg.mkBrush('w'), pen=pg.mkPen('k'),
                useOpenGL=True
            )
            self.image_scatter.sigClicked.connect(self._on_image_clicked)
            self.view.addItem(self.image_scatter)
        
        if not self._should_show_image_tooltip() and not self._should_show_hyperedge_tooltip():
            self.tooltip_manager.hide()

        if self.link_curve is None:
            self.link_curve = pg.PlotCurveItem(
                pen=pg.mkPen(QColor(255, 255, 255, 150), width=1),
                autoDownsample=True,  
                downsample=10         
            )
            self.view.addItem(self.link_curve)
            
        if self.selected_scatter is None:
            self.selected_scatter = pg.ScatterPlotItem(size=8, symbol='o', pxMode=True,
                                                        brush=pg.mkBrush('r'), pen=pg.mkPen('k'),
                                                        useOpenGL=True)
            self.view.addItem(self.selected_scatter)


        if self.selected_links is None:
            self.selected_links = pg.PlotCurveItem(
                pen=pg.mkPen(QColor(255, 0, 0, 150), width=1),
                autoDownsample=True, 
                downsample=10
            )
            self.view.addItem(self.selected_links)


        rel,links=self._radial_layout_cache
        if self.hidden_edges:
            rel = {k: off for k, off in rel.items() if k[0] not in self.hidden_edges}
            links = [
                (a, b) for (a, b) in links
                if a[0] not in self.hidden_edges and b[0] not in self.hidden_edges
            ]        
        k_list=list(rel.keys())
        if not k_list:
            self.image_scatter.setData([],[]); self.link_curve.setData([],[])
            self.selected_scatter.setData([],[]); self.selected_links.setData([],[])
            self._abs_pos_cache={}
            return
        offsets=np.array(list(rel.values()),dtype=float)
        centres=np.array([self.fa2_layout.positions[e] for e,_ in k_list])
        abs_pos=centres+offsets
        self.image_scatter.setData(pos=abs_pos, data=k_list)
        self._abs_pos_cache={k:p for k,p in zip(k_list,abs_pos)}
        if links:
            pairs=np.empty((2*len(links),2),dtype=float)
            abs_dict=self._abs_pos_cache
            for n,(a,b) in enumerate(links):
                pairs[2*n]=abs_dict[a]; pairs[2*n+1]=abs_dict[b]
            self.link_curve.setData(pairs[:,0],pairs[:,1],connect='pairs')
        else:
            self.link_curve.setData([],[])
        self._update_selected_overlay()
        self._rendered_version = self._layout_version

    def _update_selected_overlay(self):
        if not self._selected_nodes:
            if self.selected_scatter:
                self.selected_scatter.setData([], [])
            if self.selected_links:
                self.selected_links.setData([], [])
            return

        pairs: list[np.ndarray] = []
        if self._radial_layout_cache:
            abs_pos_cache = self._abs_pos_cache
            sel_pos = [abs_pos_cache[k] for k in self._selected_nodes if k in abs_pos_cache]
            if self.selected_scatter:
                if sel_pos:
                    self.selected_scatter.setData(pos=np.array(sel_pos))
                else:
                    self.selected_scatter.setData([], [])

            _, links = self._radial_layout_cache
            for a, b in links:
                if a in self._selected_nodes or b in self._selected_nodes:
                    if a in abs_pos_cache and b in abs_pos_cache:
                        pairs.append(abs_pos_cache[a])
                        pairs.append(abs_pos_cache[b])
        else:
            centres = getattr(self.fa2_layout, 'positions', {})
            sel_edges = {edge for edge, _ in self._selected_nodes if edge in centres}
            sel_pos = [centres[e] for e in sel_edges]
            if self.selected_scatter:
                if sel_pos:
                    self.selected_scatter.setData(pos=np.array(sel_pos))
                else:
                    self.selected_scatter.setData([], [])

            by_idx: dict[int, list[str]] = {}
            for edge, idx in self._selected_nodes:
                if edge in centres:
                    by_idx.setdefault(idx, []).append(edge)
            for edges in by_idx.values():
                coords = [centres[e] for e in edges if e in centres]
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        pairs.append(coords[i])
                        pairs.append(coords[j])

        if pairs and self.selected_links:
            arr = np.array(pairs)
            self.selected_links.setData(arr[:, 0], arr[:, 1], connect='pairs')
        elif self.selected_links:
            self.selected_links.setData([], [])

    def _compute_overview_triplets(self) -> dict[str, tuple[int | None, ...]]:
        session = self.session
        if session is None: return {}
        return session.compute_overview_triplets()

    def _compute_radial_layout(self, sel_name: str):
        
        if sel_name in self._radial_cache_by_edge:
            return self._radial_cache_by_edge[sel_name]

        session, layout = self.session, self.fa2_layout
        if session is None or layout is None:
            return {}, []

        offsets, links = {}, []
        if sel_name not in layout.names:
            return {}, []

        radius_map = {
            n: (layout.node_sizes[self.edge_index[n]] / 2) * self.radial_placement_factor
            for n in layout.names
        }

        sel_full = session.hyperedges[sel_name]
        if not sel_full:
            return {}, []

        if self.limit_images_enabled:
            idx_list = list(sel_full)
            feats = session.features[idx_list]
            avg = session.hyperedge_avg_features[sel_name].reshape(1, -1)
            sims = SIM_METRIC(avg, feats)[0]
            order = np.argsort(sims)
            n_top = self.limit_images_value // 2
            n_bottom = self.limit_images_value - n_top
            chosen = [idx_list[i] for i in order[::-1][:n_top]]
            for i in order[:n_bottom]:
                cand = idx_list[i]
                if cand not in chosen:
                    chosen.append(cand)
                if len(chosen) >= self.limit_images_value:
                    break
            sel_display = set(chosen)
        else:
            sel_display = set(sel_full)

        for idx in sel_display:
            vec = self._image_umap.get(sel_name, {}).get(idx, np.zeros(2))
            offsets[(sel_name, idx)] = vec * radius_map[sel_name]

        inter_counts = {}
        shared_sets = {}
        for e in session.hyperedges:
            if e == sel_name or e not in layout.names or e in self.hidden_edges:
                continue
            shared = session.hyperedges[e] & sel_full
            if shared:
                inter_counts[e] = len(shared)
                shared_sets[e] = shared

        if self.limit_edges_enabled:
            top_edges = {
                e for e, _ in sorted(inter_counts.items(), key=lambda x: x[1], reverse=True)[: self.limit_edges_value]
            }
        else:
            top_edges = set(inter_counts)

        for tgt, shared_full in shared_sets.items():
            if self.limit_edges_enabled and tgt not in top_edges:
                continue

            if self.limit_images_enabled:
                shared = shared_full & sel_display
                if not shared:
                    continue
            else:
                shared = shared_full

            show_all = (not self.limit_edges_enabled) or (tgt in top_edges)

            for idx in shared:
                vec = self._image_umap.get(tgt, {}).get(idx, np.zeros(2))
                offsets[(tgt, idx)] = vec * radius_map[tgt]
                links.append(((sel_name, idx), (tgt, idx)))


            extras = session.hyperedges[tgt] - shared_full
            if self.limit_images_enabled:
                extras = list(extras)[: self.limit_images_value]
            for idx in extras:
                vec = self._image_umap.get(tgt, {}).get(idx, np.zeros(2))
                offsets[(tgt, idx)] = vec * radius_map[tgt]


        self._radial_cache_by_edge[sel_name] = (offsets, links)
        self._layout_version = (self._layout_version or 0) + 1

        return offsets, links



    def _on_edges(self, names: list[str]):
        for name, ell in self.hyperedgeItems.items():
            col = self.color_map.get(name, '#AAAAAA')
            ell.setPen(pg.mkPen(col))
            ell.setBrush(pg.mkBrush(col))

        self._selected_nodes.clear()
        if len(names) == 1:
            sel = names[0]
            self._current_edge = sel
            self._radial_layout_cache = None
            if self.image_scatter: self.image_scatter.setData([], [])
            if self.link_curve: self.link_curve.setData([], [])
            if self.selected_scatter: self.selected_scatter.setData([], [])
            if self.selected_links: self.selected_links.setData([], [])

            if sel in self._radial_cache_by_edge:
                # If layout is cached, use it immediately
                self._radial_layout_cache = self._radial_cache_by_edge[sel]
                self._layout_version = (self._layout_version or 0) + 1
                self._update_image_layer() # Redraw with cached data
            else:
                # Otherwise, request from worker (UI is now clean and responsive)
                if self.session and self.fa2_layout:
                    self._worker.session = self.session
                    self._worker.layout = self.fa2_layout
                    self._worker.image_umap = self._image_umap
                    self._worker.limit_images_enabled = self.limit_images_enabled
                    self._worker.limit_images_value = self.limit_images_value
                    self._worker.limit_edges_enabled = self.limit_edges_enabled
                    self._worker.limit_edges_value = self.limit_edges_value
                    self._worker.hidden_edges = self.hidden_edges.copy()
                    self._worker.edge_index = self.edge_index
                    self._worker.radial_placement_factor = self.radial_placement_factor
                    self.requestRadial.emit(sel)
        else:
            # Multiple or no edges selected, clear the detailed view
            self._current_edge = None
            self._radial_layout_cache = None
            self._layout_version = (self._layout_version or 0) + 1
            self._update_image_layer() # This will hide items as the cache is None
      
    def _on_images(self, idxs: list[int]):
        if not self.session:
            return

        if self._highlight_timer.isActive():
            self._highlight_timer.stop()
            for item in self._animating_items:
                pass
            self._animating_items = []

        edges_to_highlight = {e for i in idxs for e in self.session.image_mapping.get(i, [])}

        items_to_animate = []
        for name, ell in self.hyperedgeItems.items():
            col = self.color_map.get(name, '#AAAAAA')
            if name in edges_to_highlight:
                width = ell.rect().width() * self.highlight_pen_width_ratio
                pen = QPen(pg.mkColor('w'))
                pen.setWidthF(0.05)
                pen.setCosmetic(False)
                pen.setStyle(Qt.DashLine)
                ell.setPen(pen)
                ell.setBrush(pg.mkBrush(col))
                items_to_animate.append(ell)
            else:
                ell.setPen(pg.mkPen(col))
                ell.setBrush(pg.mkBrush(col))

        if items_to_animate:
            if len(items_to_animate) < 50:
                self._animating_items = items_to_animate
                self._anim_step_count = self._highlight_anim_steps
                self._highlight_timer.start()

        self._selected_nodes = {(e, i) for i in idxs for e in self.session.image_mapping.get(i, [])}
        self._update_selected_overlay()


    def _on_hyperedge_modified(self, name: str):
        if not self.session:
            return
        self._worker.session = self.session
        self.requestRecalc.emit(name)

    def _on_edge_renamed(self, old: str, new: str) -> None:
        if old in self.hyperedgeItems:
            item = self.hyperedgeItems.pop(old)
            item.name = new
            self.hyperedgeItems[new] = item
        if old in self.color_map:
            self.color_map[new] = self.color_map.pop(old)
        if self.fa2_layout:
            if hasattr(self.fa2_layout, "positions") and old in self.fa2_layout.positions:
                self.fa2_layout.positions[new] = self.fa2_layout.positions.pop(old)
            if hasattr(self.fa2_layout, "names"):
                names = list(self.fa2_layout.names)
                try:
                    idx = names.index(old)
                    names[idx] = new
                    self.fa2_layout.names = names
                except ValueError:
                    pass
            if hasattr(self, "edge_index") and old in self.edge_index:
                idx = self.edge_index.pop(old)
                self.edge_index[new] = idx
        # Update all cached radial layouts so they no longer reference the old name
        def _rename_cache(cache: tuple[dict, list]):
            offs, links = cache
            new_offs = {(new if e == old else e, i): off for (e, i), off in offs.items()}
            new_links = [
                ((new if a[0] == old else a[0], a[1]), (new if b[0] == old else b[0], b[1]))
                for a, b in links
            ]
            return new_offs, new_links

        if self._radial_layout_cache is not None:
            self._radial_layout_cache = _rename_cache(self._radial_layout_cache)

        if self._radial_cache_by_edge:
            new_cache: dict[str, tuple[dict, list]] = {}
            for k, v in self._radial_cache_by_edge.items():
                key = new if k == old else k
                new_cache[key] = _rename_cache(v)
            self._radial_cache_by_edge = new_cache
        if old in self._centroid_norm:
            self._centroid_norm[new] = self._centroid_norm.pop(old)
        if old in self._centroid_sim:
            self._centroid_sim[new] = self._centroid_sim.pop(old)
        if old in self._image_umap:
            self._image_umap[new] = self._image_umap.pop(old)
        if old in self.hidden_edges:
            self.hidden_edges.remove(old)
            self.hidden_edges.add(new)
        if self._current_edge == old:
            self._current_edge = new
        self._selected_nodes = {(new if e == old else e, i) for e, i in self._selected_nodes}
        self._abs_pos_cache = {(new if e == old else e, i): pos for (e, i), pos in self._abs_pos_cache.items()}
        if self._overview_triplets is not None and old in self._overview_triplets:
            self._overview_triplets[new] = self._overview_triplets.pop(old)
        self._update_image_layer()


    def _on_worker_image(self, edge: str, mapping: dict):
        if not self.session:
            return
        self._image_umap[edge] = mapping
        if self.session.image_umap is None:
            self.session.image_umap = {}
        self.session.image_umap[edge] = mapping
        self._radial_cache_by_edge.pop(edge, None)
        self._radial_layout_cache = None
        self._layout_version = (self._layout_version or 0) + 1        
        self._update_image_layer()

    def _on_worker_layout(self, layout: dict):
        if not self.session:
            return
            
        self.plot.setTitle("") # Clear the "Loading..." message
        names = list(layout)
        node_sizes = np.array([layout[n][1] for n in names])
        positions = {n: layout[n][0] for n in names}
        self.fa2_layout = SimpleNamespace(names=names, node_sizes=node_sizes, positions=positions)
        self.edge_index = {n: i for i, n in enumerate(names)}
        self._radial_cache_by_edge.clear()
        self._radial_layout_cache = None
        self._layout_version = (self._layout_version or 0) + 1

        for name in list(self.hyperedgeItems):
            if name not in layout:
                self.view.removeItem(self.hyperedgeItems.pop(name))
                
        for name in names:
            size = layout[name][1]
            r = size / 2
            if name not in self.hyperedgeItems:
                ell = HyperedgeItem(name, QtCore.QRectF(-r, -r, size, size))
                col = self.color_map.get(name, '#AAAAAA')
                ell.setPen(pg.mkPen(col))
                ell.setBrush(pg.mkBrush(col))
                self.view.addItem(ell)
                ell.setZValue(-1)
                self.hyperedgeItems[name] = ell
            else:
                self.hyperedgeItems[name].setRect(QtCore.QRectF(-r, -r, size, size))
            x, y = positions[name]
            self.hyperedgeItems[name].setPos(x, y)
            self.hyperedgeItems[name].setVisible(name not in self.hidden_edges)

        self._refresh_edges()
        self._update_mini_scatter()
        self._update_minimap_view()
        self._pos_minimap()
        self._update_image_layer() # Ensure the view is updated

    def _on_worker_radial(self, sel_name: str, offsets: dict, links: list):
        self._radial_cache_by_edge[sel_name] = (offsets, links)
        if self._current_edge == sel_name:
            self._radial_layout_cache = (offsets, links)
            self._layout_version = (self._layout_version or 0) + 1
            self._update_image_layer()

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton:
            return

        scene_pos = ev.scenePos()

        # If we clicked an actual scatter POINT, let the scatter's own handler deal with it
        if self.image_scatter and self.image_scatter.isVisible():
            hit = self._get_image_point_at(scene_pos)
            if hit is not None:
                return  # do not change hyperedge selection

        # Otherwise try to select the top-most hyperedge actually under the cursor
        for it in self.plot.scene().items(scene_pos):
            if isinstance(it, HyperedgeItem) and it.isVisible():
                if it.contains(it.mapFromScene(scene_pos)):   # precise in-ellipse test
                    self.bus.set_edges([it.name])
                    ev.accept()
                    return

        # Clicked empty space: clear selection (unless Shift held)
        if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
            self.bus.set_edges([])



    def _on_image_clicked(self, scatter, points):
        if not points: return
        sel_nodes = {pt.data() for pt in points if pt.data()}
        sel_imgs = [idx for (_e, idx) in sel_nodes]
        if sel_imgs:
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                self._selected_nodes.update(sel_nodes)
            else:
                self._selected_nodes = set(sel_nodes)
            self._update_selected_overlay()
            self.bus.set_images(sel_imgs)

    def _on_lasso(self, pts: list[QPointF]):
        if not self.fa2_layout:
            return

        poly = np.array([(p.x(), p.y()) for p in pts], dtype=float)
        if len(poly) < 3:
            return

        mpl_path = MplPath(poly, closed=True)

        names = [n for n in self.fa2_layout.names if n not in self.hidden_edges]
        if names:
            centers = np.array([self.fa2_layout.positions[n] for n in names], dtype=float)
            radii   = np.array([self.fa2_layout.node_sizes[self.edge_index[n]] * 0.5
                                for n in names], dtype=float)

            K = 32
            theta = np.linspace(0.0, 2.0*np.pi, K, endpoint=False)
            unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)                 
            boundary = centers[:, None, :] + radii[:, None, None] * unit[None, :, :]  

            inside_flags = mpl_path.contains_points(boundary.reshape(-1, 2))
            fully_inside = inside_flags.reshape(len(names), K).all(axis=1)
            sel_edges = [names[i] for i in np.flatnonzero(fully_inside)]
        else:
            sel_edges = []

        if sel_edges:
            self.bus.set_edges(sel_edges)
            return

        if self._radial_layout_cache is None or not self._radial_layout_cache[0]:
            return
        rel, _ = self._radial_layout_cache
        keys = list(rel.keys())
        centres = np.array([self.fa2_layout.positions[e] for e, _ in keys], dtype=float)
        offsets = np.array(list(rel.values()), dtype=float)
        abs_pos = centres + offsets

        contained = mpl_path.contains_points(abs_pos)  # center-in test for images is fine
        sel_nodes = {k for k, inside in zip(keys, contained) if inside}
        sel_imgs = [idx for (_e, idx) in sel_nodes]
        if sel_imgs:
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                self._selected_nodes.update(sel_nodes)
            else:
                self._selected_nodes = set(sel_nodes)
            self._update_selected_overlay()
            self.bus.set_images(sel_imgs)

    def _goto(self,x,y):
        xr,yr=self.view.viewRange()
        dx,dy=(xr[1]-xr[0])/2,(yr[1]-yr[0])/2
        self.view.setRange(xRange=(x-dx,x+dx),yRange=(y-dy,y+dy),padding=0)

    def closeEvent(self, e):
        if self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()
        super().closeEvent(e)