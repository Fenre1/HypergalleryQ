from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple
from pathlib import Path
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSize,
    QEvent,
    QPoint,
    QUrl,
    QTimer,
    QRect,
)

from PyQt5.QtGui import (
    QPixmap,
    QColor,
    QKeySequence,
    QPainter,
    QIcon,
    QPalette,
    QFont,
)

from PyQt5.QtWidgets import (
    QApplication,
    QDockWidget,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QShortcut,
    QStyleOptionHeader,
    QStyle,
)

from .selection_bus import SelectionBus
from .session_model import SessionModel
from .image_popup import show_image_metadata
from .image_utils import (
    qimage_from_data,
    load_thumbnail,
)



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

class HyperedgeHeaderView(QHeaderView):
    """
    Paints the header section as:
      - Horizontal header: image on top, wrapped name below.
      - Vertical header:   image on left, wrapped name on right.
    """

    _MIN_FONT_SIZE = 5.0   # Minimum font size in points
    _MAX_FONT_SIZE = 12.0  # Maximum font size
    _BASE_SECTION_SIZE = 64.0 # The section size at which _BASE_FONT_SIZE is used
    _BASE_FONT_SIZE = 9.0     # The font size at 100% zoom (64px)

    def __init__(self, orientation: Qt.Orientation, parent=None):
        super().__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.setDefaultAlignment(Qt.AlignCenter)
        if hasattr(self, "setTextElideMode"):
            self.setTextElideMode(Qt.ElideNone)
        self._text_lines = 3
        self._margin = 6

    def sizeHint(self):
        sz = super().sizeHint()
        s = self.defaultSectionSize()
        fm = self.fontMetrics()
        if self.orientation() == Qt.Horizontal:
            txt_h = min(fm.lineSpacing() * self._text_lines, int(s * 0.75))
            sz.setHeight(int(s + txt_h + 2 * self._margin))
        else:
            txt_w = int(fm.averageCharWidth() * 12)  # room for a few words
            sz.setWidth(int(s + txt_w + 2 * self._margin))
        return sz

    def updateGeometryForZoom(self):
        sz = self.sizeHint()
        if self.orientation() == Qt.Horizontal:
            self.setFixedHeight(sz.height())
        else:
            self.setFixedWidth(sz.width())
        self.updateGeometry()


    def paintSection(self, painter: QPainter, rect: QRect, logicalIndex: int):
        if not rect.isValid():
            return

        opt = QStyleOptionHeader()
        self.initStyleOption(opt)
        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""
        opt.icon = QIcon()

        style = self.style()
        style.drawControl(QStyle.CE_HeaderSection, opt, painter, self) 

        model = self.model()
        if not model:
            return

        orientation = self.orientation()
        name = model.headerData(logicalIndex, orientation, Qt.DisplayRole) or ""
        pixdata = model.headerData(logicalIndex, orientation, Qt.DecorationRole)

        pix: QPixmap | None = None
        if isinstance(pixdata, QPixmap):
            pix = pixdata
        elif isinstance(pixdata, QIcon):
            s = pixdata.actualSize(rect.size())
            pix = pixdata.pixmap(s)

        margin = self._margin
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(self.palette().color(QPalette.ButtonText))  # visible text color

        reference_dim = min(rect.width(), rect.height())
        scaled_font_size = self._BASE_FONT_SIZE * (reference_dim / self._BASE_SECTION_SIZE)
        final_font_size = max(self._MIN_FONT_SIZE, min(scaled_font_size, self._MAX_FONT_SIZE))
        font = QFont(painter.font())
        font.setPointSizeF(final_font_size)
        painter.setFont(font)

        
        if orientation == Qt.Horizontal:
            fm = self.fontMetrics()
            text_h = min(fm.lineSpacing() * self._text_lines, int(rect.height() * 0.6))
            max_side = max(1, min(rect.width() - 2 * margin, rect.height() - text_h - 2 * margin))
            img_rect = QRect(
                rect.x() + (rect.width() - max_side) // 2,
                rect.y() + margin,
                max_side,
                max_side,
            )
            if pix and not pix.isNull():
                painter.drawPixmap(
                    img_rect,
                    pix.scaled(img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation),
                )
            text_rect = QRect(
                rect.x() + margin,
                rect.bottom() - text_h - margin + 1,
                rect.width() - 2 * margin,
                text_h,
            )
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, str(name))

        else:
            max_side = max(1, min(rect.height() - 2 * margin, int(rect.width() * 0.6)))
            img_rect = QRect(rect.x() + margin, rect.y() + (rect.height() - max_side) // 2, max_side, max_side)
            if pix and not pix.isNull():
                painter.drawPixmap(
                    img_rect,
                    pix.scaled(img_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation),
                )
            text_rect = QRect(
                img_rect.right() + margin,
                rect.y() + margin,
                max(rect.right() - img_rect.right() - 2 * margin, 1),
                max(rect.height() - 2 * margin, 1),
            )
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, str(name))

        painter.restore()





# ──────────────────────────────────────────────────────────────────────
# QAbstractTableModel implementation
# ──────────────────────────────────────────────────────────────────────
class HyperedgeMatrixModel(QAbstractTableModel):

    def __init__(
        self,
        session: SessionModel | None,
        thumb_size: int = 64,
        parent=None,
        use_full_images: bool = True,
    ):
        super().__init__(parent)
        self._session = session
        self._thumb_size = thumb_size
        self._use_full = use_full_images
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
        self._load_thumb.cache_clear()

    def set_use_full_images(self, flag: bool) -> None:
        if self._use_full == flag:
            return
        self._use_full = flag
        self._load_thumb.cache_clear()
        self.headerDataChanged.emit(Qt.Horizontal, 0, len(self._edges) - 1)
        self.headerDataChanged.emit(Qt.Vertical, 0, len(self._edges) - 1)

    def rowCount(self, parent=QModelIndex()) -> int:     
        return 0 if parent.isValid() else len(self._edges)

    def columnCount(self, parent=QModelIndex()) -> int:  
        return 0 if parent.isValid() else len(self._edges)

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

    def headerData(self, section: int, orient: Qt.Orientation, role: int = Qt.DisplayRole):
        if not self._session or section >= len(self._edges):
            return None
        name = self._edges[section]

        if role == Qt.DisplayRole:                          # ### NEW (show names)
            return name

        if role == Qt.ToolTipRole:
            return name

        if role == Qt.DecorationRole and orient in (Qt.Horizontal, Qt.Vertical):
            return self._load_thumb(name)

        if role == Qt.SizeHintRole:
            if orient == Qt.Horizontal:
                return QSize(self._thumb_size, int(self._thumb_size * 1.6))
            else:
                return QSize(int(self._thumb_size * 1.8), self._thumb_size)
        return None


    @lru_cache(maxsize=1024)
    def _load_thumb(self, edge_name: str) -> QPixmap:
        """Load & scale the first image of the hyperedge."""
        if not self._session:
            return QPixmap()
        idxs = sorted(self._session.hyperedges[edge_name])
        if not idxs:
            return QPixmap()
        idx = idxs[0]
        if not self._use_full and self._session.thumbnail_data:
            if self._session.thumbnails_are_embedded:
                data = self._session.thumbnail_data[idx]
                img = qimage_from_data(data)
                pix = QPixmap.fromImage(img)
            else:
                tpath = Path(self._session.h5_path).parent / self._session.thumbnail_data[idx]
                pix = load_thumbnail(str(tpath), self._thumb_size, self._thumb_size)
                return pix
            return pix.scaled(
                self._thumb_size,
                self._thumb_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        path = self._session.im_list[idx]
        return load_thumbnail(path, self._thumb_size, self._thumb_size)


    def _build_matrix(self):
        """Pre‑compute overlaps + F1 scores to speed up delegate drawing."""
        edges = self._edges
        sz = len(edges)
        self._overlap = [[0] * sz for _ in range(sz)]
        self._scores = [[0.0] * sz for _ in range(sz)]
        self._max_score = 0.0

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
    _ZOOM_STEP = 1.15

    def __init__(
        self,
        bus: SelectionBus,
        parent=None,
        thumb_size: int = 64,
        use_full_images: bool = True,
    ):
        super().__init__("Hyperedge Overlap", parent)
        self.bus = bus
        self._base_thumb = thumb_size
        self._zoom = 1.0
        self._overview_triplets: Dict[str, tuple[int | None, ...]] | None = None
        self._last_index = QModelIndex()
        self._use_full = use_full_images
        
        
        self._view = QTableView(self)
        self._view.setSelectionMode(QAbstractItemView.NoSelection)
        self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._view.verticalHeader().setSectionsClickable(True)
        self._view.horizontalHeader().setSectionsClickable(True)
        self._view.setAlternatingRowColors(False)

        self._view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        self._view.setHorizontalHeader(HyperedgeHeaderView(Qt.Horizontal, self._view))
        self._view.setVerticalHeader(HyperedgeHeaderView(Qt.Vertical, self._view))


        self._model = HyperedgeMatrixModel(None, thumb_size, use_full_images=use_full_images)
        self._view.setModel(self._model)
        self.setWidget(self._view)
        
        thumb_qsize = QSize(thumb_size, thumb_size)
        for hdr in (self._view.horizontalHeader(), self._view.verticalHeader()):
            hdr.setSectionResizeMode(QHeaderView.Fixed)
            hdr.setDefaultSectionSize(thumb_size)
            hdr.setMinimumSectionSize(self._MIN_SIZE)
            hdr.setIconSize(thumb_qsize) 

        hh = self._view.horizontalHeader()
        vh = self._view.verticalHeader()
        if isinstance(hh, HyperedgeHeaderView):
            hh.updateGeometryForZoom()
        if isinstance(vh, HyperedgeHeaderView):
            vh.updateGeometryForZoom()


        self._view.clicked.connect(self._on_cell_clicked)
        self._view.horizontalHeader().sectionDoubleClicked.connect(
            lambda s: self._on_header_double_clicked(s)
        )
        self._view.verticalHeader().sectionDoubleClicked.connect(
            lambda s: self._on_header_double_clicked(s)
        )

        QShortcut(QKeySequence.ZoomIn, self, self.zoom_in)
        QShortcut(QKeySequence.ZoomOut, self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.zoom_reset)

        # --- Ctrl+Wheel event filter ----------------------------------
        self._view.viewport().installEventFilter(self)
        self._view.viewport().setMouseTracking(True)

        self.tooltip_manager = TooltipManager(self._view)
        self._tooltip_timer = QTimer(self)                 # ### NEW
        self._tooltip_timer.setSingleShot(True)
        self._tooltip_timer.setInterval(500)               # 0.5s debounce
        self._tooltip_timer.timeout.connect(self._show_pending_tooltip)
        self._pending_index = QModelIndex()
        self._pending_pos = QPoint()
        self._tooltip_html_cache: Dict[Tuple[str, str, int], str] = {}  # ### NEW



    def set_model(self, session: SessionModel | None):
        """Load / clear the matrix."""
        self._model.set_session(session)
        self._overview_triplets = None
        self._tooltip_html_cache.clear()
        self.zoom_reset()  # ensures headers match current _base_thumb

    def update_matrix(self):
        """Compatibility wrapper used by the main window to refresh data."""
        self._model.set_session(self._model._session)

    def set_use_full_images(self, flag: bool) -> None:
        """Forward the image mode to the underlying model."""
        self._use_full = flag
        self._model.set_use_full_images(flag)


    # Zoom stuff
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

        self._model.set_thumb_size(size)

        hh = self._view.horizontalHeader()
        vh = self._view.verticalHeader()
        hh.setDefaultSectionSize(size)   # column width
        vh.setDefaultSectionSize(size)   # row height

        # Provide space for wrapped names
        if isinstance(hh, HyperedgeHeaderView):
            hh.updateGeometryForZoom()
        if isinstance(vh, HyperedgeHeaderView):
            vh.updateGeometryForZoom()

        # repaint everything
        self._model.dataChanged.emit(QModelIndex(), QModelIndex(), [Qt.DecorationRole, Qt.SizeHintRole])
        self._view.viewport().update()




    def _on_cell_clicked(self, index: QModelIndex):
        if not self._model._session or not index.isValid():
            return
        edges = self._model._edges
        r_name = edges[index.row()]
        c_name = edges[index.column()]
        idxs = sorted(self._model._session.hyperedges[r_name] & self._model._session.hyperedges[c_name])
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



    # Tooltip handling 
    def eventFilter(self, obj, event):
        if obj is self._view.viewport():
            if event.type() == QEvent.MouseMove:
                self._on_mouse_move(event)
            elif event.type() == QEvent.Leave:
                self._tooltip_timer.stop()
                self.tooltip_manager.hide()
                self._pending_index = QModelIndex()

        if event.type() == QEvent.Wheel and QApplication.keyboardModifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        return super().eventFilter(obj, event)

    def _on_mouse_move(self, event):
        session = self._model._session
        if session is None:
            return
        idx = self._view.indexAt(event.pos())
        if not idx.isValid():
            self.tooltip_manager.hide()
            self._tooltip_timer.stop()
            self._pending_index = QModelIndex()
            return

        if idx != self._pending_index:
            # Hovering a new cell → (re)start debounce
            self._pending_index = idx
            self._pending_pos = event.pos()
            self.tooltip_manager.hide()
            self._tooltip_timer.start()
        else:
            # Same cell: if tooltip is visible, just reposition smoothly
            if self.tooltip_manager.tooltip.isVisible():
                global_pos = self._view.viewport().mapToGlobal(event.pos())
                self.tooltip_manager.tooltip.move(global_pos + QPoint(15, 10))
            else:
                self._pending_pos = event.pos()
                self._tooltip_timer.start()

    def _show_pending_tooltip(self):
        if not self._pending_index.isValid() or self._model._session is None:
            return
        edges = self._model._edges
        r_name = edges[self._pending_index.row()]
        c_name = edges[self._pending_index.column()]

        key = (r_name, c_name, self._model._thumb_size)
        html = self._tooltip_html_cache.get(key)
        if html is None:
            html = self._build_cell_tooltip(r_name, c_name)
            self._tooltip_html_cache[key] = html

        if html:
            global_pos = self._view.viewport().mapToGlobal(self._pending_pos)
            self.tooltip_manager.show(global_pos, html)

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
            f"<table><tr>"
            f"<td valign='top'><b>{col_edge}</b><br>{col_html}</td>"
            f"<td valign='top'><b>{row_edge}</b><br>{row_html}</td>"
            f"</tr></table>"
        )
