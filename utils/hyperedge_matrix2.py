from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSize,
    QEvent,
    QPoint,
    QUrl,
)
from PyQt5.QtGui import QPixmap, QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QDockWidget,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QShortcut,
)

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .image_popup import show_image_metadata
from .image_utils import pixmap_from_file

class TooltipManager:
    """Simple helper to show persistent HTML tooltips."""

    def __init__(self, parent_widget):
        from PyQt5.QtWidgets import QLabel

        self.tooltip = QLabel(parent_widget)
        self.tooltip.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.tooltip.setStyleSheet(
            "QLabel { background-color: #FFFFE0; color: black; "
            "border: 1px solid black; padding: 2px; }"
        )
        self.tooltip.hide()

    def show(self, global_pos: QPoint, html: str):
        self.tooltip.setText(html)
        self.tooltip.adjustSize()
        self.tooltip.move(global_pos + QPoint(15, 10))
        self.tooltip.show()

    def hide(self):
        self.tooltip.hide()


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def f1_colour(score: float, max_score: float) -> QColor:
    """Convert similarity score → heat‑map colour (dark‑grey→red)."""
    if max_score == 0:
        return QColor("#555555")
    start = (0x55, 0x55, 0x55)
    end = (0xFF, 0x55, 0x55)
    t = max(0.0, min(score, max_score)) / max_score
    r = int(start[0] + (end[0] - start[0]) * t)
    g = int(start[1] + (end[1] - start[1]) * t)
    b = int(start[2] + (end[2] - start[2]) * t)
    return QColor(r, g, b)


# ──────────────────────────────────────────────────────────────────────
# QAbstractTableModel implementation
# ──────────────────────────────────────────────────────────────────────
class HyperedgeMatrixModel(QAbstractTableModel):
    """Light‑weight, virtualised model for a hyperedge overlap matrix."""

    def __init__(self, session: SessionModel | None, thumb_size: int = 64, parent=None):
        super().__init__(parent)
        self._session = session
        self._thumb_size = thumb_size
        self._edges: List[str] = list(session.hyperedges.keys()) if session else []
        self._overlap: List[List[int]] = []
        self._scores: List[List[float]] = []
        self._max_score: float = 0.0
        if session:
            self._build_matrix()

    # ------------------------------------------------------------------
    # Public helpers ----------------------------------------------------
    def set_session(self, session: SessionModel | None):
        self.beginResetModel()
        self._session = session
        self._edges = list(session.hyperedges.keys()) if session else []
        self._overlap.clear()
        self._scores.clear()
        self._max_score = 0.0
        if session:
            self._build_matrix()
        self.endResetModel()

    def set_thumb_size(self, size: int):
        if size == self._thumb_size:
            return
        self._thumb_size = size
        # cached icons depend on thumb size → clear:
        self._load_thumb.cache_clear()
        # header repaint will be triggered by view in dock widget

    # ------------------------------------------------------------------
    # QAbstractTableModel mandatory interface ---------------------------
    def rowCount(self, parent=QModelIndex()) -> int:     # noqa: N802
        return 0 if parent.isValid() else len(self._edges)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._edges)

    # Roles: DisplayRole, BackgroundRole, TextAlignmentRole -------------
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or not self._session:
            return None

        r, c = index.row(), index.column()

        if role == Qt.DisplayRole:
            return str(self._overlap[r][c])

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        if role == Qt.BackgroundRole:
            if r == c:
                return QColor("#5555FF")
            return f1_colour(self._scores[r][c], self._max_score)

        return None

    # Header thumbnails & tooltips -------------------------------------
    def headerData(self, section: int, orient: Qt.Orientation, role: int = Qt.DisplayRole):
        if not self._session or section >= len(self._edges):
            return None
        name = self._edges[section]

        if role == Qt.ToolTipRole:
            return name

        if role == Qt.DecorationRole and orient in (Qt.Horizontal, Qt.Vertical):
            return self._load_thumb(name)

        if role == Qt.SizeHintRole:
            return QSize(self._thumb_size, self._thumb_size)

        return None

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    @lru_cache(maxsize=1024)
    def _load_thumb(self, edge_name: str) -> QPixmap:
        """Load & scale the first image of the hyperedge."""
        if not self._session:
            return QPixmap()
        idxs = sorted(self._session.hyperedges[edge_name])
        if not idxs:
            return QPixmap()
        path = self._session.im_list[idxs[0]]
        pix = pixmap_from_file(path)
        return pix.scaled(
            self._thumb_size,
            self._thumb_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def _build_matrix(self):
        """Pre‑compute overlaps + F1 scores to speed up delegate drawing."""
        edges = self._edges
        sz = len(edges)
        self._overlap = [[0] * sz for _ in range(sz)]
        self._scores = [[0.0] * sz for _ in range(sz)]
        self._max_score = 0.0

        # Cache set lengths to avoid recomputing
        len_cache = {name: len(self._session.hyperedges[name]) for name in edges}

        for r, r_name in enumerate(edges):
            r_imgs = self._session.hyperedges[r_name]
            len_r = len_cache[r_name]
            for c, c_name in enumerate(edges):
                c_imgs = self._session.hyperedges[c_name]
                overlap = len(r_imgs & c_imgs)
                self._overlap[r][c] = overlap
                if r == c:
                    self._scores[r][c] = -1.0
                    continue
                len_c = len_cache[c_name]
                p1 = overlap / len_r if len_r else 0.0
                p2 = overlap / len_c if len_c else 0.0
                score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0.0
                self._scores[r][c] = score
                if score > self._max_score:
                    self._max_score = score


# ──────────────────────────────────────────────────────────────────────
# Dock widget with zoom support
# ──────────────────────────────────────────────────────────────────────
class HyperedgeMatrixDock(QDockWidget):
    """Dock widget that shows the overlap matrix with zooming capability."""

    _MIN_SIZE = 16
    _MAX_SIZE = 512
    _ZOOM_STEP = 1.15  # multiplicative; Excel steps are ~15 %

    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 64):
        super().__init__("Hyperedge Overlap", parent)
        self.bus = bus
        self._base_thumb = thumb_size
        self._zoom = 1.0
        self._overview_triplets: Dict[str, tuple[int | None, ...]] | None = None
        self._last_index = QModelIndex()
        # --- TableView & Model ----------------------------------------
        self._view = QTableView(self)
        self._view.setSelectionMode(QAbstractItemView.NoSelection)
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.verticalHeader().setSectionsClickable(True)
        self._view.horizontalHeader().setSectionsClickable(True)
        self._view.setAlternatingRowColors(False)

        # speed: per‑pixel scrolling keeps wheel nice & smooth
        self._view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        self._model = HyperedgeMatrixModel(None, thumb_size)
        self._view.setModel(self._model)
        self.setWidget(self._view)

        # --- Size policy for headers ----------------------------------
        for hdr in (self._view.horizontalHeader(), self._view.verticalHeader()):
            hdr.setSectionResizeMode(QHeaderView.Fixed)
            hdr.setDefaultSectionSize(thumb_size)
            hdr.setMinimumSectionSize(self._MIN_SIZE)
            hdr.setIconSize(QSize(thumb_size, thumb_size))

        # --- Signals ---------------------------------------------------
        self._view.clicked.connect(self._on_cell_clicked)
        self._view.horizontalHeader().sectionDoubleClicked.connect(
            lambda s: self._on_header_double_clicked(s)
        )
        self._view.verticalHeader().sectionDoubleClicked.connect(
            lambda s: self._on_header_double_clicked(s)
        )

        # --- Zoom shortcuts -------------------------------------------
        QShortcut(QKeySequence.ZoomIn, self, self.zoom_in)
        QShortcut(QKeySequence.ZoomOut, self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.zoom_reset)

        # --- Ctrl+Wheel event filter ----------------------------------
        self._view.viewport().installEventFilter(self)
        self._view.viewport().setMouseTracking(True)

        self.tooltip_manager = TooltipManager(self._view)
    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        """Load / clear the matrix."""
        self._model.set_session(session)
        self._overview_triplets = None
        self.zoom_reset()  # ensures headers match current _base_thumb

    def update_matrix(self):
        """Compatibility wrapper used by the main window to refresh data."""
        # Rebuild the model using whatever session is currently loaded
        self._model.set_session(self._model._session)

    # ------------------------------------------------------------------
    # Zoom handlers -----------------------------------------------------
    def zoom_in(self):
        self._set_zoom(self._zoom * self._ZOOM_STEP)

    def zoom_out(self):
        self._set_zoom(self._zoom / self._ZOOM_STEP)

    def zoom_reset(self):
        self._set_zoom(1.0)

    def _set_zoom(self, factor: float):
        factor = max(self._MIN_SIZE / self._base_thumb, min(factor, self._MAX_SIZE / self._base_thumb))
        if abs(factor - self._zoom) < 1e-3:
            return
        self._zoom = factor
        size = int(round(self._base_thumb * self._zoom))

        # update model thumbnails + headers
        self._model.set_thumb_size(size)
        for hdr in (self._view.horizontalHeader(), self._view.verticalHeader()):
            hdr.setDefaultSectionSize(size)
            hdr.setIconSize(QSize(size, size))

        # repaint everything
        self._model.dataChanged.emit(
            QModelIndex(), QModelIndex(), [Qt.DecorationRole, Qt.SizeHintRole]
        )

    # ------------------------------------------------------------------
    # Event filter for Ctrl+Wheel zoom ---------------------------------
    # def eventFilter(self, obj, event):
    #     if event.type() == QEvent.Wheel and QApplication.keyboardModifiers() & Qt.ControlModifier:
    #         if event.angleDelta().y() > 0:
    #             self.zoom_in()
    #         else:
    #             self.zoom_out()
    #         return True
    #     return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Click / double‑click behaviour (unchanged) ------------------------
    def _on_cell_clicked(self, index: QModelIndex):
        if not self._model._session or not index.isValid():
            return
        edges = self._model._edges
        r_name = edges[index.row()]
        c_name = edges[index.column()]
        idxs = sorted(
            self._model._session.hyperedges[r_name] & self._model._session.hyperedges[c_name]
        )
        self.bus.set_images(idxs)

    def _on_header_double_clicked(self, section: int):
        if not self._model._session:
            return
        edges = self._model._edges
        if section >= len(edges):
            return
        name = edges[section]
        idxs = sorted(self._model._session.hyperedges.get(name, []))
        if idxs:
            show_image_metadata(self._model._session, idxs[0], self)

    # ------------------------------------------------------------------
    # Tooltip handling -------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self._view.viewport():
            if event.type() == QEvent.MouseMove:
                self._update_tooltip(event)
            elif event.type() == QEvent.Leave:
                self.tooltip_manager.hide()

        if event.type() == QEvent.Wheel and QApplication.keyboardModifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        return super().eventFilter(obj, event)

    def _update_tooltip(self, event):
        session = self._model._session
        if session is None:
            self.tooltip_manager.hide()
            self._last_index = QModelIndex()
            return

        idx = self._view.indexAt(event.pos())
        if not idx.isValid():
            self.tooltip_manager.hide()
            self._last_index = QModelIndex()
            return

        if idx == self._last_index:
            # just reposition
            global_pos = self._view.viewport().mapToGlobal(event.pos())
            self.tooltip_manager.tooltip.move(global_pos + QPoint(15, 10))
            return

        self._last_index = idx
        edges = self._model._edges
        r_name = edges[idx.row()]
        c_name = edges[idx.column()]
        html = self._build_cell_tooltip(r_name, c_name)
        if html:
            global_pos = self._view.viewport().mapToGlobal(event.pos())
            self.tooltip_manager.show(global_pos, html)
        else:
            self.tooltip_manager.hide()

    def _build_cell_tooltip(self, row_edge: str, col_edge: str) -> str:
        session = self._model._session
        if session is None:
            return ""

        if self._overview_triplets is None:
            self._overview_triplets = session.compute_overview_triplets()

        row_imgs = self._overview_triplets.get(row_edge, ())
        col_imgs = self._overview_triplets.get(col_edge, ())
        size = self._model._thumb_size

        def _img_tag(idx: int) -> str:
            url = QUrl.fromLocalFile(session.im_list[idx]).toString()
            return f'<img src="{url}" width="{size}" height="{size}" style="margin:2px;">'

        row_html = "".join(_img_tag(i) for i in row_imgs if i is not None)
        col_html = "<br>".join(_img_tag(i) for i in col_imgs if i is not None)

        if not row_html and not col_html:
            return ""

        return (
            f"<table><tr><td valign='top'><b>{col_edge}</b><br>{col_html}</td>"
            f"<td valign='top'><b>{row_edge}</b><br>{row_html}</td></tr></table>"
        )