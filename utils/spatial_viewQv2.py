from __future__ import annotations



import numpy as np

import pyqtgraph as pg


from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QGraphicsPixmapItem, QGraphicsRectItem
from PyQt5.QtGui import QPalette, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPointF, pyqtSignal as Signal


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





    def mousePressEvent(self, ev):


        if ev.button() == Qt.LeftButton:


            self._drawing = True


            self._path = QPainterPath(self.mapToView(ev.pos()))


            pen = QPen(pg.mkColor('y'))


            self._item = pg.QtWidgets.QGraphicsPathItem()


            self._item.setPen(pen)


            self.addItem(self._item)


            ev.accept()


        else:


            super().mousePressEvent(ev)





    def mouseMoveEvent(self, ev):


        if self._drawing:


            self._path.lineTo(self.mapToView(ev.pos()))


            self._item.setPath(self._path)


            ev.accept()


        else:


            super().mouseMoveEvent(ev)





    def mouseReleaseEvent(self, ev):


        if self._drawing and ev.button() == Qt.LeftButton:


            self._drawing = False


            self.removeItem(self._item)


            pts = [self.mapToView(ev.pos())]


            path = []


            for i in range(self._path.elementCount()):


                el = self._path.elementAt(i)


                path.append(QPointF(el.x, el.y))


            if len(path) > 2:


                self.sigLassoFinished.emit(path)


            ev.accept()


        else:


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


        self.scatter = pg.ScatterPlotItem(x=self.embedding[:, 0],


                                          y=self.embedding[:, 1],


                                          size=5,


                                          brush=pg.mkBrush('gray'))


        self.plot.addItem(self.scatter)





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


            item.setOffset(-pix.width()/2, -pix.height()/2)


            item.setPos(self.embedding[idx, 0], self.embedding[idx, 1])


            item.setFlag(QGraphicsPixmapItem.ItemIgnoresTransformations)


            rect = QGraphicsRectItem(0, 0, pix.width(), pix.height(), item)


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