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
from PyQt5.QtCore import Qt, QPointF, QTimer, QEvent, pyqtSignal as Signal
from PyQt5.QtGui import QPainterPath, QPen, QColor
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QApplication,
    QPushButton, QGraphicsEllipseItem,
)
from matplotlib.path import Path as MplPath

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .fa2_layout import HyperedgeForceAtlas2
from .similarity import SIM_METRIC       # kept for non‑cosine fallback


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
# Main dock widget                                                             #
# ---------------------------------------------------------------------------- #
class SpatialViewQDock(QDockWidget):
    MIN_HYPEREDGE_DIAMETER = 0.5
    NODE_SIZE_SCALER       = 0.1
    zoom_threshold         = 400.0
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
        self.minimap_scatter: pg.ScatterPlotItem | None = None
        self._radial_layout_cache: tuple[dict, list] | None = None
        self._radial_cache_by_edge: dict[str, tuple[dict, list]] = {}   # NEW
        self.color_map: dict[str, str] = {}

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
            r=size/2; ell=QGraphicsEllipseItem(-r,-r,size,size)
            col=self.color_map.get(name,'#AAAAAA'); ell.setPen(pg.mkPen(col)); ell.setBrush(pg.mkBrush(col))
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
        for item in (self.image_scatter,self.link_curve):
            if item: self.view.removeItem(item)
        self.image_scatter=self.link_curve=None
        if self.minimap_scatter: self.minimap.plotItem.removeItem(self.minimap_scatter)
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
            if self.link_curve: self.link_curve.hide(); return

        xr,_=self.view.viewRange()
        if (xr[1]-xr[0])>self.zoom_threshold:
            if self.image_scatter: self.image_scatter.hide()
            if self.link_curve: self.link_curve.hide(); return
        if self.image_scatter: self.image_scatter.show()
        if self.link_curve: self.link_curve.show()

        if self.image_scatter is None:
            self.image_scatter=pg.ScatterPlotItem(size=8,symbol='o',pxMode=True,
                                                  brush=pg.mkBrush('w'),pen=pg.mkPen('k'),
                                                  useOpenGL=True)
            self.view.addItem(self.image_scatter)
        if self.link_curve is None:
            self.link_curve=pg.PlotCurveItem(pen=pg.mkPen(QColor(255,255,255,150),width=1))
            self.view.addItem(self.link_curve)

        rel,links=self._radial_layout_cache
        k_list=list(rel.keys())
        if not k_list:
            self.image_scatter.setData([],[]); self.link_curve.setData([],[]); return

        offsets=np.array(list(rel.values()),dtype=float)
        centres=np.array([self.fa2_layout.positions[e] for e,_ in k_list])
        abs_pos=centres+offsets
        self.image_scatter.setData(pos=abs_pos)

        if links:
            pairs=np.empty((2*len(links),2),dtype=float)
            abs_dict={k:p for k,p in zip(k_list,abs_pos)}
            for n,(a,b) in enumerate(links):
                pairs[2*n]=abs_dict[a]; pairs[2*n+1]=abs_dict[b]
            self.link_curve.setData(pairs[:,0],pairs[:,1],connect='pairs')
        else:
            self.link_curve.setData([],[])

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
    def _on_edges(self,names:list[str]):
        # grey
        for ell in self.hyperedgeItems.values():
            ell.setPen(pg.mkPen('#808080')); ell.setBrush(pg.mkBrush('#808080'))
        # highlight
        for name in names:
            ell=self.hyperedgeItems.get(name)
            if ell:
                col=self.color_map.get(name,'yellow')
                ell.setPen(pg.mkPen(col)); ell.setBrush(pg.mkBrush(col))

        self._radial_layout_cache = (self._compute_radial_layout(names[0])
                                     if len(names)==1 else None)
        self._update_image_layer()

    def _on_click(self,ev):
        if ev.button()!=Qt.LeftButton: return
        for name,ell in self.hyperedgeItems.items():
            if ell is self.plot.scene().itemAt(ev.scenePos(), QtGui.QTransform()):
                self.bus.set_edges([name]); ev.accept(); return
        if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
            self.bus.set_edges([])

    def _on_lasso(self,pts):
        if not self.fa2_layout: return
        names=self.fa2_layout.names
        pos=np.array([self.fa2_layout.positions[n] for n in names])
        poly=MplPath([(p.x(),p.y()) for p in pts])
        sel=[names[i] for i in np.nonzero(poly.contains_points(pos))[0]]
        self.bus.set_edges(sel)

    def _goto(self,x,y):
        xr,yr=self.view.viewRange()
        dx,dy=(xr[1]-xr[0])/2,(yr[1]-yr[0])/2
        self.view.setRange(xRange=(x-dx,x+dx),yRange=(y-dy,y+dy),padding=0)

    # ----------------------------------------------------------------------- #
    def closeEvent(self,e): self._stop_sim(); super().closeEvent(e)
