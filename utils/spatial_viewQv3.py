"""
Spatial view – fast version, including
  • GPU scatter/curve drawing (previous drop‑in)
  • Vectorised & cached radial‑layout computation (NEW).

Replace the previous file with this one.  No other code changes required.
"""

from __future__ import annotations
import numpy as np
from math import cos, sin, pi
from time import perf_counter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt, QPointF, QTimer, QEvent, pyqtSignal as Signal, QUrl, QPoint
from PyQt5.QtGui import QPainterPath, QPen, QColor
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QApplication,
    QPushButton, QGraphicsEllipseItem, QToolTip, QLabel, QGraphicsSceneHoverEvent
)
from matplotlib.path import Path as MplPath

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .fa2_layout import HyperedgeForceAtlas2
from .similarity import SIM_METRIC       # kept for non‑cosine fallback

THUMB_SIZE = 128

class TooltipManager:
    """Manages a custom QLabel widget to provide persistent tooltips."""
    def __init__(self, parent_widget: QWidget):
        self.tooltip = QLabel(parent_widget)
        # Use the ToolTip window flag to make it frameless and stay on top
        self.tooltip.setWindowFlags(Qt.ToolTip)
        # Style it to look like a standard tooltip
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

    def show(self, screen_pos, html_content: str):
        """Shows the tooltip with the given content at the specified screen position."""
        screen_pos = self._as_qpoint(screen_pos)
        self.tooltip.setText(html_content)
        self.tooltip.adjustSize()  # Resize to fit content
        # Offset slightly from the cursor
        self.tooltip.move(screen_pos + QPoint(15, 10))
        self.tooltip.show()

    def hide(self):
        """Hides the tooltip."""
        self.tooltip.hide()

    def update_position(self, screen_pos):
        """Updates the position of the tooltip if it's visible."""
        if self.tooltip.isVisible():
            screen_pos = self._as_qpoint(screen_pos)
            self.tooltip.move(screen_pos + QPoint(15, 10))

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
            pen = QPen(pg.mkColor("y")); pen.setWidth(2); pen.setCosmetic(True)
            self._item = pg.QtWidgets.QGraphicsPathItem(); self._item.setPen(pen)
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
# Graphics item with hover tooltip support                                     #
# ---------------------------------------------------------------------------- #
class HyperedgeItem(QGraphicsEllipseItem):
    """Ellipse item that can show overview tooltips when hovered."""

    def __init__(self, name: str, rect, dock: "SpatialViewQDock"):
        super().__init__(rect)
        self.name = name
        self.dock = dock
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        # Call the new handler in the dock
        self.dock._handle_hyperedge_hover_enter(self.name, event)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        # Call the new handler in the dock
        self.dock._handle_hyperedge_hover_leave(event)
        super().hoverLeaveEvent(event)
    
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        # NEW: Update tooltip position as the mouse moves
        self.dock._handle_hyperedge_hover_move(event)
        super().hoverMoveEvent(event)


class ImageScatterItem(pg.ScatterPlotItem):
    """Scatter item that shows a thumbnail tooltip when hovered."""

    def __init__(self, dock: "SpatialViewQDock", **kwargs):
        super().__init__(**kwargs)
        self.dock = dock
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, event):
        if event.isExit():
            self.dock._handle_image_hover_leave(event)
            super().hoverEvent(event)
            return

        pts = self.pointsAt(event.pos())
        if pts:
            # show tooltip for the first point under cursor
            self.dock._handle_image_hover_enter(pts[0], event)
        else:
            self.dock._handle_image_hover_leave(event)
        self.dock._handle_image_hover_move(event)
        super().hoverEvent(event)

# ---------------------------------------------------------------------------- #
# Main dock widget                                                             #
# ---------------------------------------------------------------------------- #
class SpatialViewQDock(QDockWidget):
    MIN_HYPEREDGE_DIAMETER = 0.5
    NODE_SIZE_SCALER       = 0.1
    zoom_threshold         = 400.0
    tooltip_zoom_threshold = 200.0
    image_tooltip_zoom_threshold = 50.0
    radial_placement_factor = 1.1
    

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Hyperedge View", parent)
        self.bus = bus

        # runtime ----------------------------------------------------------------
        self.session: SessionModel | None = None
        self.fa2_layout: HyperedgeForceAtlas2 | None = None
        self.hyperedgeItems: dict[str, QGraphicsEllipseItem] = {}
        self.image_scatter: pg.ScatterPlotItem | None = None
        self.link_curve: pg.PlotCurveItem | None = None
        self.selected_scatter: pg.ScatterPlotItem | None = None
        self.selected_links: pg.PlotCurveItem | None = None
        self.minimap_scatter: pg.ScatterPlotItem | None = None
        self._radial_layout_cache: tuple[dict, list] | None = None
        self._radial_cache_by_edge: dict[str, tuple[dict, list]] = {}
        self._abs_pos_cache: dict[tuple[str, int], np.ndarray] = {}
        self._selected_nodes: set[tuple[str, int]] = set()
        self.color_map: dict[str, str] = {}
        self._overview_triplets: dict[str, tuple[int | None, ...]] | None = None

        # fast‑similarity pre‑computes (NEW) -------------------------------------
        self._features_norm: np.ndarray | None = None              # (N,D)   unit‑vectors
        self._centroid_norm: dict[str, np.ndarray] = {}            # edge → (D,)
        self._centroid_sim: dict[str, np.ndarray] = {}             # edge → (n_imgs,)

        # ---------------------------------------------------------------- GUI --
        self.run_button = QPushButton("Pause Layout"); self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._toggle_sim)

        self.view = LassoViewBox(); self.view.setBackgroundColor("#444444")
        self.view.sigRangeChanged.connect(self._update_minimap_view)
        self.view.sigRangeChanged.connect(self._update_image_layer)
        self.view.sigLassoFinished.connect(self._on_lasso)
        self.plot = pg.PlotWidget(viewBox=self.view); self.plot.setBackground("#444444")
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        self.minimap_view = MiniMapViewBox(enableMenu=False)
        self.minimap = pg.PlotWidget(viewBox=self.minimap_view, parent=self.plot)
        self.minimap.setFixedSize(200, 200); self.minimap.hideAxis('bottom'); self.minimap.hideAxis('left')
        self.minimap.setBackground("#333333"); self.minimap.setMouseEnabled(False, False)
        self.minimap_view.sigGoto.connect(self._goto)
        pen = pg.mkPen('r', width=2, cosmetic=True)
        self.minimap_rect = pg.RectROI([0,0],[1,1], pen=pen, movable=False, resizable=False)
        self.minimap_view.addItem(self.minimap_rect)
        self.plot.installEventFilter(self)

        self.tooltip_manager = TooltipManager(self.plot)

        w = QWidget(); l=QVBoxLayout(w); l.addWidget(self.run_button); l.addWidget(self.plot)
        self.setWidget(w); self._pos_minimap()

        # timer ------------------------------------------------------------------
        self.timer = QTimer(self); self.timer.setInterval(16); self.timer.timeout.connect(self._frame)
        self.auto_stop_ms = 10_000

        # bus --------------------------------------------------------------------
        self.bus.edgesChanged.connect(self._on_edges)

    # ---------------------------------------------------------------- GUI small --
    def eventFilter(self,obj,e):
        if obj is self.plot and e.type()==QEvent.Resize: self._pos_minimap()
        return super().eventFilter(obj,e)
    def _pos_minimap(self):
        pw,mm=self.plot.size(),self.minimap.size()
        self.minimap.move(pw.width()-mm.width()-10,10); self.minimap.raise_()

    # ============================================================================ #
    # Session / model setup                                                        #
    # ============================================================================ #
    def set_model(self, session: SessionModel | None):
        self._clear_scene()
        self.session=session; self.fa2_layout=None; self._radial_cache_by_edge={}
        self._radial_layout_cache=None
        self._overview_triplets=None

        if session is None:
            self.run_button.setEnabled(False); return

        self.color_map = session.edge_colors.copy()

        # ---------------- ForceAtlas2 initialisation ----------------------------
        edges=list(session.hyperedges)
        overlap={rk:{ck:len(session.hyperedges[rk]&session.hyperedges[ck]) for ck in edges}
                 for rk in edges}
        self.fa2_layout = HyperedgeForceAtlas2(overlap, session)
        names = self.fa2_layout.names
        counts=np.array([np.sqrt(len(session.hyperedges[n])) for n in names])
        sizes=np.maximum(counts*self.NODE_SIZE_SCALER, self.MIN_HYPEREDGE_DIAMETER)
        self.fa2_layout.node_sizes=sizes

        for name,size in zip(names,sizes):
            r=size/2
            ell=HyperedgeItem(name, QtCore.QRectF(-r,-r,size,size), self)
            col=self.color_map.get(name,'#AAAAAA')
            ell.setPen(pg.mkPen(col)); ell.setBrush(pg.mkBrush(col))
            self.view.addItem(ell); self.hyperedgeItems[name]=ell

        # -------------- NEW: one‑time similarity pre‑compute --------------------
        feats=session.features.astype(np.float32)
        norms=np.linalg.norm(feats,axis=1,keepdims=True); norms[norms==0]=1
        self._features_norm=feats/norms                                     # (N,D)

        self._centroid_norm.clear(); self._centroid_sim.clear()
        for edge in edges:
            c=session.hyperedge_avg_features[edge].astype(np.float32)
            c/=max(np.linalg.norm(c),1e-9); self._centroid_norm[edge]=c
            idx=list(session.hyperedges[edge])
            self._centroid_sim[edge]=self._features_norm[idx]@c             # (n_imgs,)

        # -----------------------------------------------------------------------
        self._update_mini_scatter(); self._update_minimap_view(); self._pos_minimap()
        self.run_button.setEnabled(True); self._start_sim()

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
        # QToolTip.hideText()
        self.tooltip_manager.hide()
        if self.minimap_scatter: 
            self.minimap.plotItem.removeItem(self.minimap_scatter)
        self.minimap_scatter=None
        self.timer.stop()


    # ============================================================================ #
    # Animation                                                                    #
    # ============================================================================ #
    def _start_sim(self):
        if self.fa2_layout and not self.timer.isActive():
            self.timer.start(); self.run_button.setText("Pause Layout")
            QtCore.QTimer.singleShot(self.auto_stop_ms, self._stop_sim)
    def _stop_sim(self):
        if self.timer.isActive():
            self.timer.stop(); self.run_button.setText("Resume Layout")
            self._update_minimap_view()
    def _toggle_sim(self):
        self.auto_stop_ms=300_000
        (self._stop_sim() if self.timer.isActive() else self._start_sim())

    def _frame(self):
        if not self.fa2_layout: return
        self.fa2_layout.step(1); self._refresh_edges(); self._update_image_layer()

    # --------------------------------------------------------------------------- #
    # Cheap hyperedge refresh                                                     #
    # --------------------------------------------------------------------------- #
    def _refresh_edges(self):
        for name,ell in self.hyperedgeItems.items():
            x,y=self.fa2_layout.positions[name]; ell.setPos(x,y)
        self._update_mini_scatter(); self._update_minimap_view()
    def _update_mini_scatter(self):
        if not self.fa2_layout: return
        pos=np.array([self.fa2_layout.positions[n] for n in self.fa2_layout.names])
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

    # ============================================================================ #
    # Expensive radial image‑layer (vectorised + cached)                           #
    # ============================================================================ #
    def _update_image_layer(self):
        if self._radial_layout_cache is None:
            if self.image_scatter: self.image_scatter.hide()
            if self.link_curve: self.link_curve.hide()
            self.tooltip_manager.hide()
            return

        xr,_=self.view.viewRange()
        if (xr[1]-xr[0])>self.zoom_threshold:
            if self.image_scatter: self.image_scatter.hide()
            if self.link_curve: self.link_curve.hide()
            self.tooltip_manager.hide()
            return
        if self.image_scatter: self.image_scatter.show()
        if self.link_curve: self.link_curve.show()

        if self.image_scatter is None:
            self.image_scatter=ImageScatterItem(self,
                                               size=8,
                                               symbol='o',
                                               pxMode=True,
                                               brush=pg.mkBrush('w'),
                                               pen=pg.mkPen('k'),
                                               useOpenGL=True)
            self.image_scatter.sigClicked.connect(self._on_image_clicked)
            self.view.addItem(self.image_scatter)
        if self.link_curve is None:
            self.link_curve=pg.PlotCurveItem(pen=pg.mkPen(QColor(255,255,255,150),width=1))
            self.view.addItem(self.link_curve)
        if self.selected_scatter is None:
            self.selected_scatter=pg.ScatterPlotItem(size=8,symbol='o',pxMode=True,
                                                     brush=pg.mkBrush('r'),pen=pg.mkPen('k'),
                                                     useOpenGL=True)
            self.view.addItem(self.selected_scatter)
        if self.selected_links is None:
            self.selected_links=pg.PlotCurveItem(pen=pg.mkPen(QColor(255,0,0,150),width=1))
            self.view.addItem(self.selected_links)

        rel,links=self._radial_layout_cache
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

    def _update_selected_overlay(self):
        if not self._selected_nodes or not self._radial_layout_cache:
            if self.selected_scatter:
                self.selected_scatter.setData([], [])
            if self.selected_links:
                self.selected_links.setData([], [])
            return

        abs_pos_cache = self._abs_pos_cache
        sel_pos = [abs_pos_cache[k] for k in self._selected_nodes if k in abs_pos_cache]

        if self.selected_scatter:
            self.selected_scatter.setData(pos=np.array(sel_pos))

        rel, links = self._radial_layout_cache
        pairs = []
        for a, b in links:
            if a in self._selected_nodes or b in self._selected_nodes:
                if a in abs_pos_cache and b in abs_pos_cache:
                    pairs.append(abs_pos_cache[a])
                    pairs.append(abs_pos_cache[b])
        if pairs and self.selected_links:
            arr = np.array(pairs)
            self.selected_links.setData(arr[:,0], arr[:,1], connect='pairs')
        elif self.selected_links:
            self.selected_links.setData([], [])


    # -------------------------------------------------------------------
    # Tooltip helpers
    # -------------------------------------------------------------------
    # def _should_show_tooltip(self) -> bool:
    #     xr, _ = self.view.viewRange()
    #     return (xr[1] - xr[0]) <= self.tooltip_zoom_threshold

    # def _compute_overview_triplets(self) -> dict[str, tuple[int | None, ...]]:
    #     session = self.session
    #     if session is None:
    #         return {}
    #     return session.compute_overview_triplets()

    # def _maybe_show_tooltip(self, name: str, event):
    #     # Early‑outs and cache unchanged … ---------------------------------
    #     if not self._should_show_tooltip():
    #         return
    #     if self._overview_triplets is None:
    #         self._overview_triplets = self._compute_overview_triplets()

    #     trip = self._overview_triplets.get(name)
    #     if not trip or self.session is None:
    #         return

    #     # ------------------------------------------------------------------
    #     # Build an HTML snippet with <img> tags.
    #     # Qt needs a *URL*, so wrap local paths with file://
    #     # ------------------------------------------------------------------
    #     html_parts = []
    #     for i in trip:
    #         if i is None:
    #             continue
    #         fn = self.session.im_list[i]
    #         url = QUrl.fromLocalFile(fn).toString()      # guarantees escaping
    #         html_parts.append(
    #             f'<img src="{url}" width="{THUMB_SIZE}" '
    #             f'height="{THUMB_SIZE}" style="margin:2px;">'
    #         )

    #     # Join thumbnails in a row; wrap in <html> so Qt treats it as rich text
    #     html = "<html>" + "".join(html_parts) + "</html>"

    #     # Show it
    #     pos = event.screenPos()                 
    #     QToolTip.showText(pos, html, self.plot)

    def _should_show_tooltip(self) -> bool:
        xr, _ = self.view.viewRange()
        return (xr[1] - xr[0]) <= self.tooltip_zoom_threshold

    def _compute_overview_triplets(self) -> dict[str, tuple[int | None, ...]]:
        session = self.session
        if session is None:
            return {}
        return session.compute_overview_triplets()

    # --- NEW HANDLERS FOR HyperedgeItem ---

    def _handle_hyperedge_hover_enter(self, name: str, event: QGraphicsSceneHoverEvent):
        """Handles mouse entering a hyperedge item."""
        # This contains the logic from your old `_maybe_show_tooltip`
        
        # Early-outs and cache unchanged …
        if not self._should_show_tooltip():
            return
        if self._overview_triplets is None:
            self._overview_triplets = self._compute_overview_triplets()

        trip = self._overview_triplets.get(name)
        if not trip or self.session is None:
            return

        # Build an HTML snippet with <img> tags.
        html_parts = []
        for i in trip:
            if i is None:
                continue
            fn = self.session.im_list[i]
            url = QUrl.fromLocalFile(fn).toString()
            html_parts.append(
                f'<img src="{url}" width="{THUMB_SIZE}" '
                f'height="{THUMB_SIZE}" style="margin:2px;">'
            )

        html = "".join(html_parts)
        if not html:
            return
            
        # Show the tooltip using our manager
        self.tooltip_manager.show(event.screenPos(), html)

    def _handle_hyperedge_hover_leave(self, event: QGraphicsSceneHoverEvent):
        """Handles mouse leaving a hyperedge item."""
        self.tooltip_manager.hide()

    def _handle_hyperedge_hover_move(self, event: QGraphicsSceneHoverEvent):
        """Handles mouse moving over a hyperedge item."""
        self.tooltip_manager.update_position(event.screenPos())

    def _should_show_image_tooltip(self) -> bool:
        xr, _ = self.view.viewRange()
        return (xr[1] - xr[0]) <= self.image_tooltip_zoom_threshold

    def _handle_image_hover_enter(self, point, event):
        if not self._should_show_image_tooltip():
            return
        if self.session is None:
            return
        data = point.data()
        if not data:
            return
        idx = data[1]
        fn = self.session.im_list[idx]
        url = QUrl.fromLocalFile(fn).toString()
        html = f'<img src="{url}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" style="margin:2px;">'
        self.tooltip_manager.show(event.screenPos(), html)

    def _handle_image_hover_leave(self, event):
        self.tooltip_manager.hide()

    def _handle_image_hover_move(self, event):
        self.tooltip_manager.update_position(event.screenPos())


    # ============================================================================ #
    # Fast radial‑layout computation                                               #
    # ============================================================================ #
    def _compute_radial_layout(self, sel_name: str):
        """Vectorised + cached radial layout for a selected hyperedge."""
        if sel_name in self._radial_cache_by_edge:
            return self._radial_cache_by_edge[sel_name]

        session=self.session; fa=self.fa2_layout
        if session is None or fa is None: return {},[]

        offsets:dict[tuple[str,int],np.ndarray]={}
        links:list[tuple[tuple[str,int],tuple[str,int]]]=[]
        feats_norm=self._features_norm

        # ---------- selected edge ----------------------------------------------
        sel_idx=list(session.hyperedges[sel_name])
        if not sel_idx: return {},[]
        radius=(fa.node_sizes[fa.names.index(sel_name)]/2)*self.radial_placement_factor

        sims=self._centroid_sim[sel_name]
        order=np.argsort(-sims)
        for k,img_idx in enumerate(np.array(sel_idx)[order]):
            ang=pi/2 - 2*pi*k/len(order)
            offsets[(sel_name,img_idx)]=np.array([cos(ang),sin(ang)])*radius

        # ---------- other edges -------------------------------------------------
        sel_centre=fa.positions[sel_name]
        for tgt in session.hyperedges:
            if tgt==sel_name: continue
            tgt_idx=list(session.hyperedges[tgt])
            shared=[i for i in tgt_idx if i in sel_idx]
            if not shared: continue

            tgt_radius=(fa.node_sizes[fa.names.index(tgt)]/2)*self.radial_placement_factor
            tgt_centre=fa.positions[tgt]

            anchors=[]
            for idx in shared:
                pos_on_sel=sel_centre+offsets[(sel_name,idx)]
                unit=(pos_on_sel - tgt_centre); d=np.linalg.norm(unit)
                unit=np.array([1.,0.]) if d<1e-6 else unit/d
                offsets[(tgt,idx)]=unit*tgt_radius
                links.append(((sel_name,idx),(tgt,idx)))
                anchors.append({'id':idx,'angle':np.arctan2(unit[1],unit[0]),
                                'feat':feats_norm[idx]})

            anchors.sort(key=lambda a:a['angle'])
            n_anchor=len(anchors)
            remaining=[i for i in tgt_idx if i not in shared]
            if not remaining: continue

            if n_anchor==1:
                a=anchors[0]
                rem_feats=feats_norm[remaining]
                rem_sims=rem_feats@a['feat']
                order=np.argsort(-rem_sims)
                start=a['angle']+pi/4; span=1.5*pi
                for k,idx in enumerate(np.array(remaining)[order]):
                    ang=start+(k+1)/(len(order)+1)*span
                    offsets[(tgt,idx)]=np.array([cos(ang),sin(ang)])*tgt_radius
            else:
                anchor_feats=np.stack([a['feat'] for a in anchors],axis=1)  # (D,nA)
                sims_mat=feats_norm[remaining]@anchor_feats              # (m,nA)
                best_anchor=np.argmax(sims_mat,axis=1)
                best_score=sims_mat[np.arange(len(remaining)),best_anchor]
                seg_lists={j:[] for j in range(n_anchor)}
                for idx,seg,sc in zip(remaining,best_anchor,best_score):
                    seg_lists[seg].append((idx,sc))
                for j in range(n_anchor):
                    items=seg_lists[j]; 
                    if not items: continue
                    items.sort(key=lambda x:x[1],reverse=True)
                    a1,a2=anchors[j],anchors[(j+1)%n_anchor]
                    span=(a2['angle']-a1['angle']+2*pi)%(2*pi)
                    for k,(idx,_) in enumerate(items):
                        ang=a1['angle']+(k+1)/(len(items)+1)*span
                        offsets[(tgt,idx)]=np.array([cos(ang),sin(ang)])*tgt_radius

        self._radial_cache_by_edge[sel_name]=(offsets,links)        # --- cache
        return offsets,links

    # ----------------------------------------------------------------------- #
    # Event handlers                                                          #
    # ----------------------------------------------------------------------- #
    def _on_edges(self, names: list[str]):
        # restore original colours
        for name, ell in self.hyperedgeItems.items():
            col = self.color_map.get(name, '#AAAAAA')
            ell.setPen(pg.mkPen(col))
            ell.setBrush(pg.mkBrush(col))

        self._selected_nodes.clear()

        # ────────────────────────────────────────────────────────────────────
        # Only overwrite the cache if we _really_ have exactly one edge.
        # Otherwise leave the previous radial layout untouched.
        # ────────────────────────────────────────────────────────────────────
        if len(names) == 1:
            self._radial_layout_cache = self._compute_radial_layout(names[0])
        # else: keep the existing layout

        self._update_image_layer()


    def _on_click(self, ev):
        print('fire1')
        if ev.button() != Qt.LeftButton:
            return

        scene_pos = ev.scenePos()
        item = self.plot.scene().itemAt(scene_pos, QtGui.QTransform())

        # Ignore clicks on image points so edge selection persists
        if isinstance(item, pg.ScatterPlotItem) or (
            item is not None and isinstance(item.parentItem(), pg.ScatterPlotItem)
        ):
            print('fire2')
            return

        # for name, ell in self.hyperedgeItems.items():
        #     print('fire3')
        #     if ell is item:
        #         self.bus.set_edges([name])
        #         ev.accept()
        #         return

        # if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
        #     print('fire4')
        #     self.bus.set_edges([])

    def _on_image_clicked(self, scatter, points):
        print('fire5')
        if not points:
            print('fire6')
            return
        sel_nodes = [pt.data() for pt in points]
        print('fire7', sel_nodes)
        # idxs  = [n[1] for n in nodes if n]
        sel_imgs  = [idx for (_e, idx) in sel_nodes]
        print('fire8', sel_imgs)
        if sel_imgs:
            # Hold **Ctrl** while dragging to add to the existing selection.
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                print('fire9')
                self._selected_nodes.update(sel_nodes)   # additive
            else:
                print('fire10')
                self._selected_nodes = set(sel_nodes)    # replace
            self._update_selected_overlay()
            print('fire11')
            self.bus.set_images(sel_imgs)
            print('fire12')


        # if QApplication.keyboardModifiers() & Qt.ShiftModifier:
        #     self._selected_nodes.update(nodes)
        # else:
        #     self._selected_nodes = set(nodes)
        # self._update_selected_overlay()
        # if idxs:
        #     self.bus.set_images(sorted(set(idxs)))

    def _on_lasso(self, pts: list[QPointF]):
        """
        If the lasso encloses ≥1 hyperedge node → select edges.
        Otherwise, if it encloses ≥1 image‑node → select those images
        while *keeping* the current edge selection so the radial layout
        stays visible.
        """
        if not self.fa2_layout:
            return

        poly      = [(p.x(), p.y()) for p in pts]
        mpl_path  = MplPath(poly)

        # --- 1. hyperedge test ------------------------------------------- #
        names     = self.fa2_layout.names
        pos_edges = np.array([self.fa2_layout.positions[n] for n in names])
        mask_edges = mpl_path.contains_points(pos_edges)
        sel_edges  = [names[i] for i in np.nonzero(mask_edges)[0]]

        if sel_edges:
            self.bus.set_edges(sel_edges)         # normal behaviour
            return

        # --- 2. thumbnail test ------------------------------------------- #
        if self._radial_layout_cache is None:
            return
        rel, _links = self._radial_layout_cache
        if not rel:
            return

        keys      = list(rel.keys())              # (edge, img_idx)
        centres   = np.array([self.fa2_layout.positions[e] for e, _ in keys])
        offsets   = np.array(list(rel.values()))
        abs_pos   = centres + offsets
        mask_imgs = mpl_path.contains_points(abs_pos)
        sel_nodes = [k for k, inside in zip(keys, mask_imgs) if inside]
        sel_imgs  = [idx for (_e, idx) in sel_nodes]

        if sel_imgs:
            print('lasso',sel_imgs)
            # Hold **Ctrl** while dragging to add to the existing selection.
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                self._selected_nodes.update(sel_nodes)   # additive
            else:
                self._selected_nodes = set(sel_nodes)    # replace
            self._update_selected_overlay()
            self.bus.set_images(sel_imgs)



    def _goto(self,x,y):
        xr,yr=self.view.viewRange()
        dx,dy=(xr[1]-xr[0])/2,(yr[1]-yr[0])/2
        self.view.setRange(xRange=(x-dx,x+dx),yRange=(y-dy,y+dy),padding=0)

    # ----------------------------------------------------------------------- #
    def closeEvent(self,e): self._stop_sim(); super().closeEvent(e)
