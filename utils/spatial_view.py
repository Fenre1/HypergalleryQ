from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout
from PySide6.QtGui import QPalette

from .selection_bus import SelectionBus
from .session_model import SessionModel

import umap


class SpatialViewDock(QDockWidget):
    """Interactive 2D embedding view with lasso selection."""

    def __init__(self, bus: SelectionBus, parent=None):
        super().__init__("Spatial View", parent)
        self.bus = bus
        self.session: SessionModel | None = None
        self.embedding: np.ndarray | None = None
        self.color_map: dict[str, str] = {}
        self.image_annotations: list[AnnotationBbox] = []

        self.fig = Figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)
        self._apply_theme()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setWidget(widget)

        self.scatter = None
        self.lasso = None
        self._pan = None

        self.bus.edgesChanged.connect(self._on_edges_changed)

    # ------------------------------------------------------------------
    def _is_dark_mode(self) -> bool:
        pal = self.palette()
        col = pal.color(QPalette.Window)
        # simple luminance check
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
        self.fig.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.tick_params(colors=fg)
        for spine in self.ax.spines.values():
            spine.set_color(fg)
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)

    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        base = 1.2
        scale = base if event.button == "down" else 1 / base
        cur_xmin, cur_xmax = self.ax.get_xlim()
        cur_ymin, cur_ymax = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        width = (cur_xmax - cur_xmin) * scale
        height = (cur_ymax - cur_ymin) * scale
        relx = (cur_xmax - xdata) / (cur_xmax - cur_xmin)
        rely = (cur_ymax - ydata) / (cur_ymax - cur_ymin)
        self.ax.set_xlim(xdata - (1 - relx) * width, xdata + relx * width)
        self.ax.set_ylim(ydata - (1 - rely) * height, ydata + rely * height)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_press(self, event):
        if event.button == 3 and event.inaxes == self.ax:
            self._pan = (
                event.x,
                event.y,
                self.ax.get_xlim(),
                self.ax.get_ylim(),
            )

    # ------------------------------------------------------------------
    def _on_release(self, event):
        if event.button == 3:
            self._pan = None

    # ------------------------------------------------------------------
    def _on_motion(self, event):
        if not self._pan or event.inaxes != self.ax:
            return
        x0, y0, (xmin, xmax), (ymin, ymax) = self._pan
        dx = event.x - x0
        dy = event.y - y0
        width = xmax - xmin
        height = ymax - ymin
        dx_data = dx / self.canvas.width() * width
        dy_data = dy / self.canvas.height() * height
        self.ax.set_xlim(xmin - dx_data, xmax - dx_data)
        self.ax.set_ylim(ymin - dy_data, ymax - dy_data)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        self.ax.clear()
        self.image_annotations.clear()
        self._apply_theme()
        if not session:
            self.canvas.draw_idle()
            return

        self.embedding = umap.UMAP(n_components=2).fit_transform(session.features)

        self.scatter = self.ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            s=10,
            color="gray",
            picker=True,
        )

        cmap = plt.get_cmap("tab20")
        edges = list(session.hyperedges)
        self.color_map = {
            n: plt.matplotlib.colors.to_hex(cmap(i / len(edges)))
            for i, n in enumerate(edges)
        }

        self.lasso = LassoSelector(self.ax, onselect=self._on_lasso_select, button=1)
        self.ax.set_title("UMAP embedding")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_lasso_select(self, verts):
        if self.embedding is None:
            return
        path = Path(verts)
        idxs = np.nonzero(path.contains_points(self.embedding))[0]
        self.bus.set_images(list(map(int, idxs)))

    # ------------------------------------------------------------------
    def _on_edges_changed(self, names: list[str]):
        if not self.session or self.embedding is None or self.scatter is None:
            return
        self._clear_image_annotations()

        colors = ["gray"] * len(self.embedding)
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
                    colors[idx] = self.color_map.get(main, "red")
                else:
                    ov = next((e for e in overlaps if e in edges), None)
                    if ov:
                        colors[idx] = self.color_map.get(ov, "blue")

            # zoom to fit selected hyperedge
            pts = self.embedding[list(selected)]
            if len(pts):
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                dx = xmax - xmin
                dy = ymax - ymin
                pad_x = dx * 0.1 if dx > 0 else 1
                pad_y = dy * 0.1 if dy > 0 else 1
                self.ax.set_xlim(xmin - pad_x, xmax + pad_x)
                self.ax.set_ylim(ymin - pad_y, ymax + pad_y)

            # show sample images
            # images unique to the main hyperedge (or random if none)
            unique = [i for i in selected
                      if len(self.session.image_mapping.get(i, set())) == 1]
            if not unique:
                imgs = list(selected)
                if len(imgs) > 3:
                    imgs = list(np.random.choice(imgs, 3, replace=False))
            else:
                imgs = unique[:3]
            self._add_sample_images(imgs, self.color_map.get(main, "yellow"))

            for edge in overlaps:
                inter = list(self.session.hyperedges.get(edge, set()) & selected)
                if len(inter) > 3:
                    inter = list(np.random.choice(inter, 3, replace=False))
                self._add_sample_images(inter, self.color_map.get(edge, "yellow"))

        self.scatter.set_color(colors)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _add_sample_images(self, idxs: list[int], color: str):
        for idx in idxs:
            img_path = self.session.im_list[idx]
            try:
                arr = plt.imread(img_path)
            except Exception:
                continue
            zoom = 64 / max(arr.shape[0], arr.shape[1])
            im = OffsetImage(arr, zoom=zoom)
            ab = AnnotationBbox(
                im,
                self.embedding[idx],
                xycoords="data",
                boxcoords="offset points",
                frameon=True,
                bboxprops=dict(edgecolor=color, linewidth=2),
            )
            self.ax.add_artist(ab)
            self.image_annotations.append(ab)

    # ------------------------------------------------------------------
    def _clear_image_annotations(self):
        for ab in self.image_annotations:
            ab.remove()
        self.image_annotations.clear()
        self.canvas.draw_idle()