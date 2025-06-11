from __future__ import annotations

from typing import List
from functools import lru_cache

from PyQt5.QtWidgets import (QDockWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QWidget, QGridLayout, QLabel,
                             QHBoxLayout, QVBoxLayout, QScrollArea, QApplication, QAbstractItemView) # Added QScrollArea, QApplication
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, QSize,QObject 

from .selection_bus import SelectionBus
from .session_model import SessionModel

      
class ScrollBarSynchronizer:
    """
    A class to synchronize two scrollbars proportionally.
    Connects the valueChanged signals of two scrollbars to keep them in sync.
    """
    def __init__(self, sb1, sb2):
        self.scrollbars = (sb1, sb2)
        
        # --- KEY CHANGE: Use lambdas to pass the sender explicitly ---
        sb1.valueChanged.connect(lambda value: self._sync_other(sb1, value))
        sb2.valueChanged.connect(lambda value: self._sync_other(sb2, value))
        # ---
        
        self._is_syncing = False # Prevents recursion

    def _sync_other(self, sender, value): # The sender is now an explicit argument
        if self._is_syncing:
            return

        self._is_syncing = True
        
        # Determine the other scrollbar
        other = self.scrollbars[1] if sender is self.scrollbars[0] else self.scrollbars[0]
        
        # Calculate the proportional position
        sender_max = sender.maximum()
        if sender_max == 0:
            proportion = 0
        else:
            proportion = value / sender_max

        # Apply the proportional position to the other scrollbar
        other_max = other.maximum()
        other.setValue(int(proportion * other_max))
        
        self._is_syncing = False

    

class HyperedgeMatrixDock(QDockWidget):
    """Dock widget showing overlap between hyperedges using a grid layout."""

    def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 256):
        super().__init__("Hyperedge Overlap", parent)
        self.bus = bus
        self.thumb_size = thumb_size
        self.session: SessionModel | None = None

        main_widget = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(grid_layout)

        # 1. Corner Widget (top-left)
        corner_widget = QWidget()
        corner_widget.setFixedSize(self.thumb_size, self.thumb_size)
        grid_layout.addWidget(corner_widget, 0, 0)

        # 2. Horizontal Thumbnail Area (top-right)
        self.h_thumb_container = QWidget()
        self.h_thumb_layout = QHBoxLayout()
        self.h_thumb_layout.setSpacing(0)
        self.h_thumb_layout.setContentsMargins(0, 0, 0, 0)
        # self.h_thumb_layout.addStretch() # Remove stretch from here initially
        self.h_thumb_container.setLayout(self.h_thumb_layout)

        self.h_scroll_area = QScrollArea()
        self.h_scroll_area.setWidget(self.h_thumb_container)
        self.h_scroll_area.setWidgetResizable(True)
        self.h_scroll_area.setFixedHeight(self.thumb_size)
        # self.h_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.h_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded) 
        self.h_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        grid_layout.addWidget(self.h_scroll_area, 0, 1)

        # 3. Vertical Thumbnail Area (bottom-left)
        self.v_thumb_container = QWidget()
        self.v_thumb_layout = QVBoxLayout()
        self.v_thumb_layout.setSpacing(0)
        self.v_thumb_layout.setContentsMargins(0, 0, 0, 0)
        # self.v_thumb_layout.addStretch() # Remove stretch from here initially
        self.v_thumb_container.setLayout(self.v_thumb_layout)

        self.v_scroll_area = QScrollArea()
        self.v_scroll_area.setWidget(self.v_thumb_container)
        self.v_scroll_area.setWidgetResizable(True)
        self.v_scroll_area.setFixedWidth(self.thumb_size)
        # self.v_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.v_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.v_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        grid_layout.addWidget(self.v_scroll_area, 1, 0)

        # 4. Main Data Table (bottom-right)
        self.table = QTableWidget()
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(self.thumb_size)
        self.table.horizontalHeader().setDefaultSectionSize(self.thumb_size)
        # Ensure cells cannot be resized by user dragging, they are fixed size
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)

        grid_layout.addWidget(self.table, 1, 1)
        
        # Stretch factors for the grid layout (important)
        grid_layout.setColumnStretch(1, 1) # Column 1 (table & h_scroll_area) takes priority for width
        grid_layout.setRowStretch(1, 1)    # Row 1 (table & v_scroll_area) takes priority for height

        # --- KEY CHANGE: Constrain the maximum size of main_widget ---
        # This prevents main_widget from requesting an enormous size from the QDockWidget.
        # Use available screen geometry for a sensible upper limit.
        if QApplication.instance(): # Ensure QApplication exists
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            main_widget.setMaximumSize(screen_geometry.width(), screen_geometry.height())
        else:
            # Fallback if no QApplication (e.g., during testing, though unlikely in a full app)
            main_widget.setMaximumSize(4096, 4096) 
        # --- End of KEY CHANGE ---

        self.setWidget(main_widget) # main_widget is the direct child of the dock

        # Connect scrollbars for synchronized scrolling
        # Table scrolls -> h_scroll_area (for horizontal thumbnails) scrolls
        self.h_scroll_sync = ScrollBarSynchronizer(
            self.table.horizontalScrollBar(),
            self.h_scroll_area.horizontalScrollBar()
        )
        self.v_scroll_sync = ScrollBarSynchronizer(
            self.table.verticalScrollBar(),
            self.v_scroll_area.verticalScrollBar()
        )

        self.table.cellClicked.connect(self._on_cell_clicked)

    # ------------------------------------------------------------------
    def set_model(self, session: SessionModel | None):
        self.session = session
        self._load_thumb.cache_clear()
        self.update_matrix()

    # ------------------------------------------------------------------
    @staticmethod
    def _clear_layout(layout):
        """Helper to remove all widgets from a layout."""
        if layout is None:
            return
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    # ------------------------------------------------------------------
    def update_matrix(self):
        # Clear previous state
        self._clear_layout(self.h_thumb_layout)
        self._clear_layout(self.v_thumb_layout)
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._data = {}

        if not self.session:
            return

        edges = list(self.session.hyperedges.keys())
        self.table.setRowCount(len(edges))
        self.table.setColumnCount(len(edges))
        
        # --- Populate Thumbnail Headers ---
        for name in edges:
            pixmap = self._load_thumb(name)
            
            h_label = QLabel()
            h_label.setPixmap(pixmap)
            h_label.setFixedSize(self.thumb_size, self.thumb_size)
            h_label.setAlignment(Qt.AlignCenter)
            h_label.setToolTip(name)
            self.h_thumb_layout.addWidget(h_label)
            
            v_label = QLabel()
            v_label.setPixmap(pixmap)
            v_label.setFixedSize(self.thumb_size, self.thumb_size)
            v_label.setAlignment(Qt.AlignCenter)
            v_label.setToolTip(name)
            self.v_thumb_layout.addWidget(v_label)
        
        # Add spacers to push thumbnails to the top/left
        self.h_thumb_layout.addStretch()
        self.v_thumb_layout.addStretch()

        # Fill matrix and store overlap counts
        self._data: dict[str, dict[str, int]] = {}
        for r, r_name in enumerate(edges):
            self._data[r_name] = {}
            r_imgs = self.session.hyperedges[r_name]
            for c, c_name in enumerate(edges):
                c_imgs = self.session.hyperedges[c_name]
                overlap = len(r_imgs & c_imgs)
                item = QTableWidgetItem(str(overlap))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
                self._data[r_name][c_name] = overlap

        self.apply_heatmap_coloring()

    # ... The rest of your methods (_on_cell_clicked, _load_thumb, apply_heatmap_coloring, get_heatmap_color) remain exactly the same ...
    # ------------------------------------------------------------------
    def _on_cell_clicked(self, row: int, col: int):
        """Send the intersection of the two selected hyperedges via the bus."""
        if not self.session:
            return
        edges = list(self.session.hyperedges.keys())
        if not (0 <= row < len(edges) and 0 <= col < len(edges)):
            return
        r_name = edges[row]
        c_name = edges[col]
        idxs = sorted(self.session.hyperedges[r_name] &
                      self.session.hyperedges[c_name])
        self.bus.set_images(idxs)

    # ------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def _load_thumb(self, edge_name: str) -> QPixmap:
        idxs = sorted(self.session.hyperedges.get(edge_name, [])) if self.session else []
        if not idxs:
            return QPixmap()
        path = self.session.im_list[idxs[0]]
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix

    # ------------------------------------------------------------------
    def apply_heatmap_coloring(self):
        if not self.session or not self._data:
            return
            
        hyperedge_keys = list(self._data)
        scores = {}
        max_score = 0.0

        # First pass: calculate all scores and find the maximum
        for i, rk in enumerate(hyperedge_keys):
            scores[rk] = {}
            for j, ck in enumerate(hyperedge_keys):
                if i == j:
                    scores[rk][ck] = -1  # Sentinel for diagonal
                    continue
                
                overlap = self._data[rk][ck]
                len_rk = len(self.session.hyperedges[rk])
                len_ck = len(self.session.hyperedges[ck])
                
                p1 = overlap / len_rk if len_rk else 0
                p2 = overlap / len_ck if len_ck else 0
                
                # Using F1 score as the metric
                score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0
                scores[rk][ck] = score
                if score > max_score:
                    max_score = score

        # Second pass: apply colors to the table items
        for i, rk in enumerate(hyperedge_keys):
            for j, ck in enumerate(hyperedge_keys):
                item = self.table.item(i, j)
                if item is None:
                    continue
                
                if i == j:
                    item.setBackground(QColor('#5555FF'))  # Diagonal color
                else:
                    score = scores[rk][ck]
                    color = QColor(self.get_heatmap_color(score, max_score))
                    item.setBackground(color)

    
    # ------------------------------------------------------------------
    @staticmethod
    def get_heatmap_color(score: float, max_score: float) -> str:
        if max_score == 0:
            return '#555555'
        start_color = (0x55, 0x55, 0x55)
        end_color = (0xFF, 0x55, 0x55)
        t = max(0.0, min(score, max_score)) / max_score
        r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        return f'#{r:02x}{g:02x}{b:02x}'


# from __future__ import annotations

# from typing import List
# from functools import lru_cache

# # from PySide6.QtWidgets import QDockWidget, QTableWidget, QTableWidgetItem
# # from PySide6.QtGui import QIcon, QPixmap, QColor
# # from PySide6.QtCore import Qt, QSize
# from PyQt5.QtWidgets import QDockWidget, QTableWidget, QTableWidgetItem, QHeaderView
# from PyQt5.QtGui import QIcon, QPixmap, QColor
# from PyQt5.QtCore import Qt, QSize


# from .selection_bus import SelectionBus

# from .session_model import SessionModel


# class HyperedgeMatrixDock(QDockWidget):
#     """Dock widget showing overlap between hyperedges."""

#     def __init__(self, bus: SelectionBus, parent=None, thumb_size: int = 256):
#         super().__init__("Hyperedge Overlap", parent)
#         self.bus = bus
#         self.thumb_size = thumb_size
#         icon_qsize = QSize(thumb_size, thumb_size)
#         self.session: SessionModel | None = None
#         self.table = QTableWidget(self)
#         self.table.verticalHeader().setDefaultSectionSize(self.thumb_size)
#         self.table.horizontalHeader().setDefaultSectionSize(self.thumb_size)
#         self.table.setIconSize(QSize(self.thumb_size, self.thumb_size))
#         self.setWidget(self.table)
#         self.table.cellClicked.connect(self._on_cell_clicked)

#         for hdr in (self.table.horizontalHeader(),
#                     self.table.verticalHeader()):
#             hdr.setIconSize(icon_qsize)                          # ⇐ key call
#             hdr.setSectionResizeMode(QHeaderView.Fixed)          # keep size constant

#         # give them enough room
#         self.table.horizontalHeader().setFixedHeight(thumb_size)
#         # self.table.verticalHeader().setMinimumWidth(thumb_size)
#         self.table.verticalHeader().setFixedWidth(thumb_size)
#         self.table.cellClicked.connect(self._on_cell_clicked)




#     # ------------------------------------------------------------------
#     def set_model(self, session: SessionModel | None):
#         self.session = session
#         # Clear cached thumbnails so icons match the currently loaded session
#         self._load_thumb.cache_clear()
#         self.update_matrix()

#     # ------------------------------------------------------------------
#     def update_matrix(self):
#         if not self.session:
#             self.table.clear()
#             self.table.setRowCount(0)
#             self.table.setColumnCount(0)
#             self._data = {}
#             return

#         edges = list(self.session.hyperedges.keys())
#         self.table.setRowCount(len(edges))
#         self.table.setColumnCount(len(edges))

#         # Set headers with thumbnails
#         for i, name in enumerate(edges):
#             icon = QIcon(self._load_thumb(name))
            
#             h_item = QTableWidgetItem()
#             h_item.setIcon(icon)
#             h_item.setToolTip(name)
#             # h_item.setText(name)
#             v_item = QTableWidgetItem()
#             v_item.setIcon(icon)
#             v_item.setToolTip(name)
#             # v_item.setText(name)
#             self.table.setHorizontalHeaderItem(i, h_item)
#             self.table.setVerticalHeaderItem(i, v_item)

#         # Fill matrix and store overlap counts
#         self._data: dict[str, dict[str, int]] = {}
#         for r, r_name in enumerate(edges):
#             self._data[r_name] = {}
#             r_imgs = self.session.hyperedges[r_name]
#             for c, c_name in enumerate(edges):
#                 c_imgs = self.session.hyperedges[c_name]
#                 overlap = len(r_imgs & c_imgs)
#                 item = QTableWidgetItem(str(overlap))
#                 item.setTextAlignment(Qt.AlignCenter)
#                 self.table.setItem(r, c, item)
#                 self._data[r_name][c_name] = overlap

#         self.apply_heatmap_coloring()

#     # ------------------------------------------------------------------
#     def _on_cell_clicked(self, row: int, col: int):
#         """Send the intersection of the two selected hyperedges via the bus."""
#         if not self.session:
#             return
#         edges = list(self.session.hyperedges.keys())
#         if not (0 <= row < len(edges) and 0 <= col < len(edges)):
#             return
#         r_name = edges[row]
#         c_name = edges[col]
#         idxs = sorted(self.session.hyperedges[r_name] &
#                       self.session.hyperedges[c_name])
#         self.bus.set_images(idxs)

#     # ------------------------------------------------------------------
#     @lru_cache(maxsize=256)
#     def _load_thumb(self, edge_name: str) -> QPixmap:
#         idxs = sorted(self.session.hyperedges.get(edge_name, [])) if self.session else []
#         if not idxs:
#             return QPixmap()
#         path = self.session.im_list[idxs[0]]
#         pix = QPixmap(path)
#         if not pix.isNull():
#             pix = pix.scaled(self.thumb_size, self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         return pix

#     # ------------------------------------------------------------------
      
#     def apply_heatmap_coloring(self):
#         if not self.session or not self._data:
#             return
            
#         hyperedge_keys = list(self._data)
#         scores = {}
#         max_score = 0.0

#         # First pass: calculate all scores and find the maximum
#         for i, rk in enumerate(hyperedge_keys):
#             scores[rk] = {}
#             for j, ck in enumerate(hyperedge_keys):
#                 if i == j:
#                     scores[rk][ck] = -1  # Sentinel for diagonal
#                     continue
                
#                 overlap = self._data[rk][ck]
#                 len_rk = len(self.session.hyperedges[rk])
#                 len_ck = len(self.session.hyperedges[ck])
                
#                 p1 = overlap / len_rk if len_rk else 0
#                 p2 = overlap / len_ck if len_ck else 0
                
#                 # Using F1 score as the metric
#                 score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0
#                 scores[rk][ck] = score
#                 if score > max_score:
#                     max_score = score

#         # Second pass: apply colors to the table items
#         for i, rk in enumerate(hyperedge_keys):
#             for j, ck in enumerate(hyperedge_keys):
#                 item = self.table.item(i, j)
#                 if item is None:
#                     continue
                
#                 if i == j:
#                     item.setBackground(QColor('#5555FF'))  # Diagonal color
#                 else:
#                     score = scores[rk][ck]
#                     # This was missing the "self" prefix in your code
#                     color = QColor(self.get_heatmap_color(score, max_score))
#                     item.setBackground(color)

    
#     # ------------------------------------------------------------------
#     @staticmethod
#     def get_heatmap_color(score: float, max_score: float) -> str:
#         if max_score == 0:
#             return '#555555'
#         start_color = (0x55, 0x55, 0x55)
#         end_color = (0xFF, 0x55, 0x55)
#         t = max(0.0, min(score, max_score)) / max_score
#         r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
#         g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
#         b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
#         return f'#{r:02x}{g:02x}{b:02x}'